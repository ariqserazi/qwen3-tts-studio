"""OpenAI-powered outline generator for podcast episodes."""

# pyright: reportImplicitRelativeImport=false, reportMissingImports=false, reportDeprecated=false
# pyright: reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportAny=false, reportImplicitStringConcatenation=false
# pyright: reportAssignmentType=false

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Mapping, Protocol, cast

from pydantic import ValidationError

from storage.persona_models import Persona

_config_module = import_module("config")
get_openai_api_key = cast(
    Callable[[], str], getattr(_config_module, "get_openai_api_key")
)
_models_module = import_module("podcast.models")

MODEL_NAME = "gpt-5.2"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = (1, 2, 4)


class _Speaker(Protocol):
    name: str
    voice_id: str
    role: str
    type: str


class _SpeakerProfile(Protocol):
    speakers: Sequence[_Speaker]


class _Segment(Protocol):
    size: str


class _Outline(Protocol):
    segments: Sequence[_Segment]

    @classmethod
    def model_validate(cls, data: object) -> "_Outline":
        ...


OutlineClass = cast(type[_Outline], getattr(_models_module, "Outline"))
SpeakerProfileClass = cast(
    type[_SpeakerProfile], getattr(_models_module, "SpeakerProfile")
)


class _OpenAIMessage(Protocol):
    content: str


class _OpenAIChoice(Protocol):
    message: _OpenAIMessage


class _OpenAIResponse(Protocol):
    choices: Sequence[_OpenAIChoice]


class _OpenAIChatCompletions(Protocol):
    def create(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, str]],
        response_format: dict[str, str],
        temperature: float,
    ) -> _OpenAIResponse:
        ...


class _OpenAIChat(Protocol):
    completions: _OpenAIChatCompletions


class _OpenAIClient(Protocol):
    chat: _OpenAIChat


class _OpenAIConstructor(Protocol):
    def __call__(self, *, api_key: str) -> _OpenAIClient:
        ...


def _load_openai() -> tuple[
    _OpenAIConstructor, type[Exception], type[Exception], type[Exception]
]:
    try:
        module = import_module("openai")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "openai package is required. Install it in a virtual environment."
        ) from exc

    try:
        openai_client = cast(_OpenAIConstructor, getattr(module, "OpenAI"))
        rate_limit_error = cast(type[Exception], getattr(module, "RateLimitError"))
        api_timeout_error = cast(type[Exception], getattr(module, "APITimeoutError"))
        api_error = cast(type[Exception], getattr(module, "APIError"))
    except AttributeError as exc:
        raise RuntimeError("openai package missing expected symbols.") from exc

    return openai_client, rate_limit_error, api_timeout_error, api_error


def _coerce_speakers(
    speakers: Sequence[_Speaker] | _SpeakerProfile,
) -> list[_Speaker]:
    if isinstance(speakers, SpeakerProfileClass):
        return list(speakers.speakers)
    return list(cast(Sequence[_Speaker], speakers))


def _format_key_points(key_points: Iterable[str]) -> str:
    cleaned = [point.strip() for point in key_points if point.strip()]
    if not cleaned:
        return "- None provided"
    return "\n".join(f"- {point}" for point in cleaned)


def _format_speakers(speakers: Sequence[_Speaker]) -> str:
    if not speakers:
        return "- None provided"
    lines: list[str] = []
    for speaker in speakers:
        lines.append(
            "- {name} (role: {role}, voice_id: {voice_id}, type: {type})".format(
                name=speaker.name,
                role=speaker.role,
                voice_id=speaker.voice_id,
                type=speaker.type,
            )
        )
    return "\n".join(lines)


def _format_persona_context(personas: dict[str, Persona] | None) -> str:
    if not personas:
        return ""
    lines = []
    for _, persona in personas.items():
        # Combine background and bio for full character context
        full_context = f"{persona.background} {persona.bio}".strip()
        context_truncated = full_context[:200] + "..." if len(full_context) > 200 else full_context
        expertise_str = ", ".join(persona.expertise[:3]) if persona.expertise else "General"
        line = (
            f"- {persona.character_name}: {persona.personality}, {persona.speaking_style} speaker. "
            f"Expertise: {expertise_str}. {context_truncated}"
        )
        lines.append(line)
    return "\n".join(lines)


def _segment_size_targets(num_segments: int) -> dict[str, int]:
    targets = {"short": 0.3, "medium": 0.5, "long": 0.2}
    raw = {key: num_segments * ratio for key, ratio in targets.items()}
    counts = {key: int(raw[key]) for key in targets}
    remainder = num_segments - sum(counts.values())
    priority = {"medium": 2, "short": 1, "long": 0}
    ordering = sorted(
        targets.keys(),
        key=lambda key: (raw[key] - counts[key], priority[key]),
        reverse=True,
    )
    for index in range(remainder):
        counts[ordering[index % len(ordering)]] += 1
    return counts


def _format_size_targets(size_targets: dict[str, int]) -> str:
    return "\n".join(
        f"- {size}: {count}" for size, count in size_targets.items()
    )


def _build_outline_prompt(
    topic: str,
    key_points: list[str],
    briefing: str,
    num_segments: int,
    speakers: Sequence[_Speaker],
    size_targets: dict[str, int],
    personas: dict[str, Persona] | None = None,
) -> str:
    outline_schema = {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "size": {
                            "type": "string",
                            "enum": ["short", "medium", "long"],
                        },
                    },
                    "required": ["title", "description", "size"],
                },
                "minItems": num_segments,
                "maxItems": num_segments,
            }
        },
        "required": ["segments"],
    }

    persona_section = ""
    if personas:
        persona_context = _format_persona_context(personas)
        if persona_context:
            persona_section = f"""
SPEAKER PERSONAS:
{persona_context}
"""

    return f"""You are an expert podcast producer. Build a clear, engaging episode outline.

TOPIC:
{topic}

KEY POINTS:
{_format_key_points(key_points)}

STYLE BRIEFING:
{briefing}

SPEAKERS:
{_format_speakers(speakers)}
{persona_section}
REQUIREMENTS:
1. Create exactly {num_segments} segments.
2. Ensure every key point is covered across the outline.
3. Segment sizes must follow this distribution (use each size exactly this many times):
{_format_size_targets(size_targets)}
4. Segment order should flow logically and build momentum.
5. Include speaker roles when describing the segment focus.
6. Design segments that leverage each speaker's expertise and personality.

OUTPUT FORMAT:
Return ONLY valid JSON matching this schema:
{json.dumps(outline_schema, indent=2)}
"""


def _extract_response_content(response: _OpenAIResponse) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError) as exc:
        raise ValueError("OpenAI response missing message content.") from exc
    if not content:
        raise ValueError("OpenAI response returned empty content.")
    return content


def _validate_distribution(outline: _Outline, size_targets: dict[str, int]) -> None:
    counts = {"short": 0, "medium": 0, "long": 0}
    for segment in outline.segments:
        if segment.size not in counts:
            raise ValueError(f"Unexpected segment size: {segment.size}")
        counts[segment.size] += 1
    if counts != size_targets:
        raise ValueError(
            f"Segment sizes do not match distribution. Expected {size_targets}, got {counts}."
        )


def _parse_outline_response(
    response_text: str,
    num_segments: int,
    size_targets: dict[str, int],
) -> _Outline:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON response from OpenAI.") from exc

    if not isinstance(payload, dict):
        raise ValueError("OpenAI response must be a JSON object.")
    payload = cast(dict[str, object], payload)

    try:
        outline = OutlineClass.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("OpenAI response does not match Outline schema.") from exc

    if len(outline.segments) != num_segments:
        raise ValueError(
            f"Segment count mismatch. Expected {num_segments}, got {len(outline.segments)}."
        )

    _validate_distribution(outline, size_targets)
    return outline


def generate_outline(
    topic: str,
    key_points: list[str],
    briefing: str,
    num_segments: int,
    speakers: Sequence[_Speaker] | _SpeakerProfile,
    personas: dict[str, Persona] | None = None,
) -> _Outline:
    """
    Generate a podcast outline using OpenAI.

    Args:
        topic: Main topic for the episode.
        key_points: Key points that must be covered.
        briefing: Style or background briefing.
        num_segments: Exact number of segments.
        speakers: List of Speaker instances or a SpeakerProfile.
        personas: Optional dict mapping voice_id to Persona for enhanced context.

    Returns:
        Outline Pydantic model.
    """
    if num_segments < 1:
        raise ValueError("num_segments must be at least 1.")

    speaker_list = _coerce_speakers(speakers)
    size_targets = _segment_size_targets(num_segments)
    prompt = _build_outline_prompt(
        topic=topic,
        key_points=key_points,
        briefing=briefing,
        num_segments=num_segments,
        speakers=speaker_list,
        size_targets=size_targets,
        personas=personas,
    )

    OpenAI, RateLimitError, APITimeoutError, APIError = _load_openai()
    client = OpenAI(api_key=get_openai_api_key())
    last_error: Exception | None = None
    total_attempts = MAX_RETRIES + 1

    for attempt in range(total_attempts):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You create podcast outlines that are structured, "
                            "engaging, and JSON-only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
            )
            content = _extract_response_content(response)
            return _parse_outline_response(content, num_segments, size_targets)
        except (RateLimitError, APITimeoutError, APIError) as exc:
            last_error = exc
        except ValueError as exc:
            last_error = exc

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS[attempt])

    raise RuntimeError(
        f"Failed to generate outline after {total_attempts} attempts."
    ) from last_error


@dataclass
class _MockMessage:
    content: str


@dataclass
class _MockChoice:
    message: _OpenAIMessage


@dataclass
class _MockResponse:
    choices: Sequence[_OpenAIChoice]

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "_MockResponse":
        content = json.dumps(payload)
        return cls(choices=(_MockChoice(message=_MockMessage(content=content)),))


def _build_mock_segments(size_targets: dict[str, int]) -> list[dict[str, str]]:
    segments: list[dict[str, str]] = []
    index = 1
    for size in ("short", "medium", "long"):
        for _ in range(size_targets.get(size, 0)):
            segments.append(
                {
                    "title": f"{size.title()} Segment {index}",
                    "description": f"Description for {size} segment {index}.",
                    "size": size,
                }
            )
            index += 1
    return segments


if __name__ == "__main__":
    test_segments = 5
    targets = _segment_size_targets(test_segments)
    mock_payload = {"segments": _build_mock_segments(targets)}
    mock_response = _MockResponse.from_payload(mock_payload)
    mock_content = _extract_response_content(mock_response)
    outline = _parse_outline_response(mock_content, test_segments, targets)
    assert len(outline.segments) == test_segments
    print("Mock OpenAI response parsed successfully.")

"""OpenAI-powered transcript generator for podcast episodes."""

# pyright: reportImplicitRelativeImport=false, reportMissingImports=false, reportDeprecated=false
# pyright: reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportAny=false, reportImplicitStringConcatenation=false
# pyright: reportAssignmentType=false
# pyright: reportUnusedParameter=false
# pyright: reportCallIssue=false

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Mapping, Protocol, Sequence, cast

from pydantic import ValidationError

logger = logging.getLogger(__name__)

from storage.persona_models import Persona
from podcast.prompts import get_transcript_prompt

_config_module = import_module("config")
get_openai_api_key = cast(
    Callable[[], str], getattr(_config_module, "get_openai_api_key")
)
_models_module = import_module("podcast.models")

MODEL_NAME = "gpt-5.2"
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (1, 2, 4)
TURN_TARGETS = {"short": 3, "medium": 6, "long": 10}


class _Speaker(Protocol):
    name: str
    voice_id: str
    role: str
    type: str


class _SpeakerProfile(Protocol):
    speakers: Sequence[_Speaker]


class _Segment(Protocol):
    title: str
    description: str
    size: str


class _Outline(Protocol):
    segments: Sequence[_Segment]

    @classmethod
    def model_validate(cls, data: object) -> "_Outline":
        ...


class _Dialogue(Protocol):
    speaker: str
    text: str


class _Transcript(Protocol):
    dialogues: Sequence[_Dialogue]

    @classmethod
    def model_validate(cls, data: object) -> "_Transcript":
        ...


OutlineClass = cast(type[_Outline], getattr(_models_module, "Outline"))
TranscriptClass = cast(type[_Transcript], getattr(_models_module, "Transcript"))
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


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _speaker_name_map(speakers: Sequence[_Speaker]) -> dict[str, str]:
    name_map: dict[str, str] = {}
    for speaker in speakers:
        normalized = _normalize_name(speaker.name)
        name_map[normalized] = speaker.name
    return name_map


def _validate_speaker_names(dialogues: Sequence[_Dialogue], speakers: Sequence[_Speaker]) -> None:
    allowed = _speaker_name_map(speakers)
    for dialogue in dialogues:
        normalized = _normalize_name(dialogue.speaker)
        if normalized not in allowed:
            raise ValueError(f"Unexpected speaker name: {dialogue.speaker}")


def _canonicalize_speaker_names(
    dialogues: Sequence[_Dialogue], speakers: Sequence[_Speaker]
) -> list[dict[str, str]]:
    allowed = _speaker_name_map(speakers)
    canonicalized: list[dict[str, str]] = []
    for dialogue in dialogues:
        normalized = _normalize_name(dialogue.speaker)
        if normalized not in allowed:
            raise ValueError(f"Unexpected speaker name: {dialogue.speaker}")
        canonicalized.append(
            {"speaker": allowed[normalized], "text": dialogue.text}
        )
    return canonicalized


def _format_outline_for_prompt(outline: _Outline, topic: str) -> str:
    lines = [f"Topic: {topic}", "Segments:"]
    for index, segment in enumerate(outline.segments, start=1):
        lines.append(
            f"{index}. {segment.title} ({segment.size}) - {segment.description}"
        )
    return "\n".join(lines)


def _format_speaker_roles(speakers: Sequence[_Speaker]) -> str:
    if not speakers:
        return "- None provided"
    lines = []
    for speaker in speakers:
        lines.append(f"- {speaker.name}: {speaker.role}")
    return "\n".join(lines)


def _format_persona_context(personas: dict[str, Persona]) -> str:
    """Format persona information for prompt injection.

    Args:
        personas: Dict mapping voice_id to Persona object

    Returns:
        Formatted persona context string
    """
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


def _segment_turns(size: str) -> int:
    normalized = size.strip().lower()
    return TURN_TARGETS.get(normalized, TURN_TARGETS["medium"])


def _format_segment_for_prompt(segment: _Segment) -> str:
    return (
        f"{segment.title} ({segment.size})\n"
        f"Summary: {segment.description}"
    )


def _extract_response_content(response: _OpenAIResponse) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError) as exc:
        raise ValueError("OpenAI response missing message content.") from exc
    if not content:
        raise ValueError("OpenAI response returned empty content.")
    return content


def _parse_transcript_response(
    response_text: str, speakers: Sequence[_Speaker]
) -> list[dict[str, str]]:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON response from OpenAI.") from exc

    if not isinstance(payload, dict):
        raise ValueError("OpenAI response must be a JSON object.")
    payload = cast(dict[str, object], payload)

    try:
        transcript = TranscriptClass.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("OpenAI response does not match Transcript schema.") from exc

    _validate_speaker_names(transcript.dialogues, speakers)
    return _canonicalize_speaker_names(transcript.dialogues, speakers)


def _fallback_dialogue(
    speakers: Sequence[_Speaker], segment: _Segment, is_final: bool
) -> list[dict[str, str]]:
    if not speakers:
        raise ValueError("At least one speaker is required for fallback dialogue.")
    host = next((speaker for speaker in speakers if speaker.role == "Host"), None)
    chosen = host or speakers[0]
    if is_final:
        text = (
            "We hit a technical snag on this segment, but that's all for today. "
            "Thanks for listening, and we'll see you next time."
        )
    else:
        text = (
            f"We ran into a technical snag on {segment.title}. "
            "Let's move to the next part of the conversation."
        )
    return [{"speaker": chosen.name, "text": text}]


def generate_transcript(
     outline: _Outline,
     topic: str,
     briefing: str,
     speakers: Sequence[_Speaker] | _SpeakerProfile,
     personas: dict[str, Persona] | None = None,
     language: str = "English",
  ) -> _Transcript:
      """
      Generate a podcast transcript using OpenAI.

      Args:
          outline: Outline Pydantic model for the episode.
          topic: Main topic for the episode.
          briefing: Style or background briefing.
          speakers: List of Speaker instances or a SpeakerProfile.
          personas: Optional dict mapping voice_id to Persona objects for enhanced context.
          language: Target language for the transcript (default: English).

      Returns:
          Transcript Pydantic model.
      """
      speaker_list = _coerce_speakers(speakers)
      if not speaker_list:
          raise ValueError("At least one speaker is required to generate a transcript.")

      outline_text = _format_outline_for_prompt(outline, topic)
      role_context = _format_speaker_roles(speaker_list)
      enriched_briefing = f"{briefing}\n\nSPEAKER ROLES:\n{role_context}"

      if personas:
          persona_context = _format_persona_context(personas)
          if persona_context:
              enriched_briefing += f"\n\nSPEAKER PERSONAS:\n{persona_context}"

      OpenAI, RateLimitError, APITimeoutError, APIError = _load_openai()
      client = OpenAI(api_key=get_openai_api_key())

      dialogues: list[dict[str, str]] = []
      total_segments = len(outline.segments)

      for index, segment in enumerate(outline.segments):
          is_final = index == total_segments - 1
          turns = _segment_turns(segment.size)
          logger.info(f"[LANG] Transcript generation: {language}")
          prompt = get_transcript_prompt(
              outline=outline_text,
              segment=_format_segment_for_prompt(segment),
              briefing=enriched_briefing,
              speakers=[speaker.name for speaker in speaker_list],
              is_final=is_final,
              turns=turns,
              language=language,
          )

          last_error: Exception | None = None
          segment_dialogues: list[dict[str, str]] | None = None
          for attempt in range(MAX_ATTEMPTS):
              try:
                  response = client.chat.completions.create(
                      model=MODEL_NAME,
                      messages=[
                          {
                              "role": "system",
                              "content": (
                                  "You create podcast transcripts that are natural, "
                                  "role-aware, and JSON-only."
                              ),
                          },
                          {"role": "user", "content": prompt},
                      ],
                      response_format={"type": "json_object"},
                      temperature=0.5,
                  )
                  content = _extract_response_content(response)
                  segment_dialogues = _parse_transcript_response(content, speaker_list)
                  break
              except (RateLimitError, APITimeoutError, APIError) as exc:
                  last_error = exc
              except ValueError as exc:
                  last_error = exc

              if attempt < MAX_ATTEMPTS - 1:
                  time.sleep(RETRY_BACKOFF_SECONDS[attempt])

          if segment_dialogues is None:
              _ = last_error
              segment_dialogues = _fallback_dialogue(speaker_list, segment, is_final)

          dialogues.extend(segment_dialogues)

      return TranscriptClass.model_validate({"dialogues": dialogues})


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


if __name__ == "__main__":
    Speaker = cast(type, getattr(_models_module, "Speaker"))
    Segment = cast(type, getattr(_models_module, "Segment"))

    speakers = [
        Speaker(name="Alex", voice_id="voice_a", role="Host", type="preset"),
        Speaker(name="Riley", voice_id="voice_b", role="Expert", type="saved"),
    ]
    outline_data = {
        "segments": [
            Segment(
                title="Welcome",
                description="Introduce the topic and guest.",
                size="short",
            ),
            Segment(
                title="Deep Dive",
                description="Explore the main points with examples.",
                size="medium",
            ),
            Segment(
                title="Wrap Up",
                description="Summarize and close the episode.",
                size="short",
            ),
    ]
    }
    outline = OutlineClass.model_validate(outline_data)

    mock_payload = {
    "dialogues": [
    {"speaker": "Alex", "text": "Welcome to the show."},
    {"speaker": "Riley", "text": "Glad to be here."},
    ]
    }
    mock_response = _MockResponse.from_payload(mock_payload)
    mock_content = _extract_response_content(mock_response)
    parsed = _parse_transcript_response(mock_content, speakers)
    transcript = TranscriptClass.model_validate({"dialogues": parsed})
    assert len(transcript.dialogues) == 2
    logger.info("Mock transcript parsed successfully.")


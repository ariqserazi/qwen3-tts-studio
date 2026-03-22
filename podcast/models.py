"""Pydantic models for podcast generation."""

from __future__ import annotations

from typing import Callable, ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


ALLOWED_SPEAKER_ROLES: Final[frozenset[str]] = frozenset(
    {"Host", "Expert", "Guest", "Narrator"}
)
ALLOWED_SPEAKER_TYPES: Final[frozenset[str]] = frozenset({"preset", "saved"})
ALLOWED_SEGMENT_SIZES: Final[frozenset[str]] = frozenset({"short", "medium", "long"})


class PodcastBaseModel(BaseModel):
    """Base class for podcast models."""

    model_config: ClassVar[ConfigDict] = ConfigDict(str_strip_whitespace=True)

    def to_json(self) -> str:
        """Serialize the model to JSON."""

        return self.model_dump_json()


class Speaker(PodcastBaseModel):
    """A podcast speaker with voice and role metadata."""

    name: str = Field(..., description="Display name for the speaker.")
    voice_id: str = Field(..., description="Voice identifier for TTS.")
    role: str = Field(..., description="Role: Host, Expert, Guest, or Narrator.")
    type: str = Field(..., description="Voice type: preset or saved.")

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Speaker name cannot be empty.")
        return cleaned

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        cleaned = value.strip()
        normalized = cleaned.title()
        if normalized not in ALLOWED_SPEAKER_ROLES:
            allowed = ", ".join(sorted(ALLOWED_SPEAKER_ROLES))
            raise ValueError(f"Speaker role must be one of: {allowed}.")
        return normalized

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in ALLOWED_SPEAKER_TYPES:
            raise ValueError("Speaker type must be 'preset' or 'saved'.")
        return cleaned


class SpeakerProfile(PodcastBaseModel):
    """A collection of speakers participating in the podcast."""

    speakers: list[Speaker] = Field(..., description="List of 1 to 4 speakers.")

    @field_validator("speakers")
    @classmethod
    def validate_speakers(cls, value: list[Speaker]) -> list[Speaker]:
        count = len(value)
        if count < 1 or count > 4:
            raise ValueError("SpeakerProfile must include 1 to 4 speakers.")

        seen: set[str] = set()
        for speaker in value:
            normalized = speaker.name.strip().lower()
            if normalized in seen:
                raise ValueError("Speaker names must be unique.")
            seen.add(normalized)
        return value


class Segment(PodcastBaseModel):
    """A single outline segment for a podcast episode."""

    title: str = Field(..., description="Segment title.")
    description: str = Field(..., description="Segment summary or notes.")
    size: str = Field(..., description="Segment length: short, medium, or long.")

    @field_validator("size")
    @classmethod
    def validate_size(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in ALLOWED_SEGMENT_SIZES:
            allowed = ", ".join(sorted(ALLOWED_SEGMENT_SIZES))
            raise ValueError(f"Segment size must be one of: {allowed}.")
        return cleaned


class Outline(PodcastBaseModel):
    """A structured outline consisting of segments."""

    segments: list[Segment] = Field(..., description="Ordered list of segments.")


class Dialogue(PodcastBaseModel):
    """A single line of dialogue in the transcript."""

    speaker: str = Field(..., description="Speaker name.")
    text: str = Field(..., description="Spoken text.")

    @field_validator("speaker")
    @classmethod
    def validate_speaker(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Speaker name cannot be empty.")
        return cleaned

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Dialogue text cannot be empty.")
        return cleaned


class Transcript(PodcastBaseModel):
    """A transcript containing ordered dialogue lines."""

    dialogues: list[Dialogue] = Field(..., description="Ordered dialogue list.")


class PodcastMetadata(PodcastBaseModel):
    """Metadata describing a podcast episode."""

    title: str = Field(..., description="Episode title.")
    description: str | None = Field(None, description="Episode summary.")
    language: str | None = Field(None, description="Language tag (e.g., en, es).")
    tags: list[str] = Field(default_factory=list, description="Keyword tags.")


if __name__ == "__main__":
    def expect_validation_error(factory: Callable[[], object]) -> None:
        try:
            _ = factory()
        except ValidationError:
            return
        raise AssertionError("Expected ValidationError")

    speakers = [
        Speaker(name="Alex", voice_id="voice_a", role="Host", type="preset"),
        Speaker(name="Riley", voice_id="voice_b", role="Expert", type="saved"),
    ]
    profile = SpeakerProfile(speakers=speakers)

    outline = Outline(
        segments=[
            Segment(title="Intro", description="Welcome and setup.", size="short"),
            Segment(title="Main", description="Deep dive discussion.", size="medium"),
        ]
    )

    transcript = Transcript(
        dialogues=[
            Dialogue(speaker="Alex", text="Welcome to the show."),
            Dialogue(speaker="Riley", text="Thanks for having me."),
        ]
    )

    metadata = PodcastMetadata(
        title="Example Episode",
        description="A short demo episode.",
        language="en",
        tags=["demo", "test"],
    )

    assert isinstance(profile.model_dump_json(), str)
    assert isinstance(outline.model_dump_json(), str)
    assert isinstance(transcript.model_dump_json(), str)
    assert "Example Episode" in metadata.model_dump_json()
    assert isinstance(metadata.to_json(), str)

    expect_validation_error(
        lambda: Speaker(name="Alex", voice_id="v1", role="Leader", type="preset")
    )
    expect_validation_error(
        lambda: Segment(title="Bad", description="Invalid size", size="tiny")
    )
    expect_validation_error(lambda: Dialogue(speaker="Alex", text="   "))
    expect_validation_error(
        lambda: SpeakerProfile(
            speakers=[
                Speaker(name="Alex", voice_id="v1", role="Host", type="preset"),
                Speaker(name="alex", voice_id="v2", role="Guest", type="saved"),
            ]
        )
    )
    expect_validation_error(
        lambda: SpeakerProfile(
            speakers=[
                Speaker(name="A", voice_id="v1", role="Host", type="preset"),
                Speaker(name="B", voice_id="v2", role="Expert", type="preset"),
                Speaker(name="C", voice_id="v3", role="Guest", type="preset"),
                Speaker(name="D", voice_id="v4", role="Narrator", type="saved"),
                Speaker(name="E", voice_id="v5", role="Guest", type="saved"),
            ]
        )
    )

    print("All tests passed.")

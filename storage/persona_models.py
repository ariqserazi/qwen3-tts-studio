# -*- coding: utf-8 -*-
"""Pydantic models for podcast persona management."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Final

from pydantic import Field, field_validator

from podcast.models import PodcastBaseModel


# Allowed values for persona attributes
ALLOWED_PERSONALITIES: Final[frozenset[str]] = frozenset(
    {
        "Cheerful",
        "Serious",
        "Witty",
        "Thoughtful",
        "Analytical",
        "Energetic",
        "Calm",
        "Humorous",
    }
)

ALLOWED_SPEAKING_STYLES: Final[frozenset[str]] = frozenset(
    {
        "Formal",
        "Casual",
        "Conversational",
        "Professional",
        "Friendly",
        "Academic",
        "Storytelling",
    }
)

ALLOWED_VOICE_TYPES: Final[frozenset[str]] = frozenset({"preset", "saved"})


class Persona(PodcastBaseModel):
    """A character persona for a podcast voice.

    Personas define the personality, speaking style, and expertise of a voice,
    enabling richer and more consistent character representation in transcripts.
    """

    voice_id: str = Field(
        ..., description="Unique identifier for the voice (preset or saved)."
    )
    voice_type: str = Field(
        ..., description="Type of voice: 'preset' or 'saved'."
    )
    character_name: str = Field(
        ..., description="Display name for the character persona."
    )
    personality: str = Field(
        ...,
        description="Personality trait: Cheerful, Serious, Witty, Thoughtful, Analytical, Energetic, Calm, or Humorous.",
    )
    speaking_style: str = Field(
        ...,
        description="Speaking style: Formal, Casual, Conversational, Professional, Friendly, Academic, or Storytelling.",
    )
    expertise: list[str] = Field(
        default_factory=list,
        description="List of expertise domains (e.g., 'AI Ethics', 'Philosophy').",
    )
    background: str = Field(
        default="",
        description="Free-form background information about the character.",
    )
    bio: str = Field(
        default="",
        description="Free-form character description and personality notes.",
    )
    created: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp when the persona was created.",
    )

    @field_validator("voice_type")
    @classmethod
    def validate_voice_type(cls, value: str) -> str:
        """Validate that voice_type is 'preset' or 'saved'."""
        cleaned = value.strip().lower()
        if cleaned not in ALLOWED_VOICE_TYPES:
            raise ValueError("Voice type must be 'preset' or 'saved'.")
        return cleaned

    @field_validator("personality")
    @classmethod
    def validate_personality(cls, value: str) -> str:
        """Validate that personality is in the allowed list."""
        cleaned = value.strip()
        if cleaned not in ALLOWED_PERSONALITIES:
            allowed = ", ".join(sorted(ALLOWED_PERSONALITIES))
            raise ValueError(f"Personality must be one of: {allowed}.")
        return cleaned

    @field_validator("speaking_style")
    @classmethod
    def validate_speaking_style(cls, value: str) -> str:
        """Validate that speaking_style is in the allowed list."""
        cleaned = value.strip()
        if cleaned not in ALLOWED_SPEAKING_STYLES:
            allowed = ", ".join(sorted(ALLOWED_SPEAKING_STYLES))
            raise ValueError(f"Speaking style must be one of: {allowed}.")
        return cleaned

    @field_validator("expertise")
    @classmethod
    def validate_expertise(cls, value: list[str]) -> list[str]:
        """Trim whitespace and remove empty strings from expertise list."""
        cleaned = [item.strip() for item in value]
        return [item for item in cleaned if item]

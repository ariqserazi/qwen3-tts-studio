# -*- coding: utf-8 -*-
"""Storage operations for persona management.

Provides CRUD operations (Create, Read, Update, Delete) for saving and loading
persona data to/from the filesystem. Personas are stored in a directory structure:
    personas/{voice_id}_{voice_type}_default/persona.json

This module handles:
- Saving personas to JSON files with UTF-8 encoding
- Loading personas from disk with graceful error handling
- Listing all saved personas
- Deleting personas from disk
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from storage.persona_models import Persona


def _get_personas_dir() -> Path:
    """Get the personas directory, creating it if it doesn't exist.
    
    Returns:
        Path object pointing to the personas directory.
    """
    personas_dir = Path.cwd() / "personas"
    personas_dir.mkdir(exist_ok=True)
    return personas_dir


def _get_persona_path(voice_id: str, voice_type: str) -> Path:
    """Get the file path for a persona.
    
    Args:
        voice_id: Voice identifier (e.g., "serena", "my_voice")
        voice_type: Type of voice ("preset" or "saved")
        
    Returns:
        Path object pointing to the persona.json file.
    """
    personas_dir = _get_personas_dir()
    persona_dir = personas_dir / f"{voice_id}_{voice_type}_default"
    return persona_dir / "persona.json"


def save_persona(persona: Persona) -> None:
    """Save a persona to disk.
    
    Saves the persona as a JSON file in the directory structure:
        personas/{voice_id}_{voice_type}_default/persona.json
    
    The JSON file is formatted with UTF-8 encoding and 2-space indentation
    for readability.
    
    Args:
        persona: The Persona object to save.
        
    Raises:
        OSError: If the file cannot be written (e.g., permission denied).
        ValueError: If the persona data is invalid.
    """
    persona_path = _get_persona_path(persona.voice_id, persona.voice_type)
    
    persona_path.parent.mkdir(parents=True, exist_ok=True)
    persona_dict = persona.model_dump()
    with open(persona_path, "w", encoding="utf-8") as f:
        json.dump(persona_dict, f, indent=2, ensure_ascii=False)


def load_persona(voice_id: str, voice_type: str) -> Persona | None:
    """Load a persona from disk.
    
    Attempts to load a persona from the expected file path. If the file
    does not exist, returns None gracefully. If the file exists but is
    invalid JSON or fails validation, raises an exception.
    
    Args:
        voice_id: Voice identifier (e.g., "serena", "my_voice").
        voice_type: Type of voice ("preset" or "saved").
        
    Returns:
        Persona object if found and valid, None if file does not exist.
        
    Raises:
        json.JSONDecodeError: If the JSON file is malformed.
        ValueError: If the persona data fails Pydantic validation.
    """
    from storage.persona_models import Persona
    
    persona_path = _get_persona_path(voice_id, voice_type)
    
    if not persona_path.exists():
        return None
    
    with open(persona_path, "r", encoding="utf-8") as f:
        persona_dict = json.load(f)
    
    persona = Persona.model_validate(persona_dict)
    return persona


def list_personas() -> list[tuple[str, str, Persona]]:
    """List all saved personas.
    
    Scans the personas directory and loads all valid personas. Skips
    invalid or corrupted persona files with a warning.
    
    Returns:
        List of tuples (voice_id, voice_type, persona) for each saved persona.
        Returns empty list if no personas exist.
    """
    from storage.persona_models import Persona
    
    personas_dir = _get_personas_dir()
    result = []
    
    if not personas_dir.exists():
        return result
    
    for persona_dir in personas_dir.iterdir():
        if not persona_dir.is_dir():
            continue
        
        dir_name = persona_dir.name
        if not dir_name.endswith("_default"):
            continue
        
        parts = dir_name[:-8].rsplit("_", 1)
        if len(parts) != 2:
            continue
        
        voice_id, voice_type = parts
        persona_path = persona_dir / "persona.json"
        if not persona_path.exists():
            continue
        
        try:
            with open(persona_path, "r", encoding="utf-8") as f:
                persona_dict = json.load(f)
            persona = Persona.model_validate(persona_dict)
            result.append((voice_id, voice_type, persona))
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Skipping invalid persona at {persona_path}: {e}")
            continue
    
    return result


def delete_persona(voice_id: str, voice_type: str) -> bool:
    """Delete a persona from disk.
    
    Removes the persona directory and all its contents. If the persona
    does not exist, returns False without raising an error.
    
    Args:
        voice_id: Voice identifier (e.g., "serena", "my_voice").
        voice_type: Type of voice ("preset" or "saved").
        
    Returns:
        True if the persona was deleted, False if it did not exist.
        
    Raises:
        OSError: If the deletion fails (e.g., permission denied).
    """
    import shutil
    
    persona_path = _get_persona_path(voice_id, voice_type)
    persona_dir = persona_path.parent
    
    if not persona_dir.exists():
        return False
    
    shutil.rmtree(persona_dir)
    return True

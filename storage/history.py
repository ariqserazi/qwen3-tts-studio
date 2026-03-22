"""Podcast file storage management."""

import json
import re
from pathlib import Path

from podcast.models import Outline, PodcastMetadata, Transcript


PODCASTS_DIR = Path("podcasts")


def sanitize_podcast_name(name: str) -> str:
    """
    Sanitize podcast name for use as directory name.
    
    Removes invalid filename characters and prevents directory traversal.
    
    Args:
        name: Raw podcast name
        
    Returns:
        Sanitized name safe for filesystem
        
    Raises:
        ValueError: If name is empty after sanitization
    """
    sanitized = re.sub(r'[<>:"/\\|?*!@#$%^&()+=\[\]{};,]', '', name)
    sanitized = sanitized.replace(" ", "_")
    sanitized = sanitized.strip(". ")
    
    if not sanitized:
        raise ValueError("Podcast name cannot be empty or contain only invalid characters")
    
    return sanitized


def create_podcast_directory(podcast_name: str) -> Path:
    """
    Create podcast directory structure.
    
    Creates podcasts/{name}/ with clips/ subdirectory.
    
    Args:
        podcast_name: Name of the podcast
        
    Returns:
        Path to created podcast directory
        
    Raises:
        ValueError: If podcast name is invalid
        PermissionError: If directory creation fails due to permissions
    """
    sanitized_name = sanitize_podcast_name(podcast_name)
    podcast_dir = PODCASTS_DIR / sanitized_name
    
    try:
        podcast_dir.mkdir(parents=True, exist_ok=True)
        clips_dir = podcast_dir / "clips"
        clips_dir.mkdir(exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Cannot create directory {podcast_dir}: {e}") from e
    except OSError as e:
        raise OSError(f"Error creating podcast directory: {e}") from e
    
    return podcast_dir


def save_outline(outline: Outline, podcast_dir: Path) -> Path:
    """
    Save outline to outline.json.
    
    Args:
        outline: Outline model instance
        podcast_dir: Path to podcast directory
        
    Returns:
        Path to saved outline.json
        
    Raises:
        FileNotFoundError: If podcast directory doesn't exist
        PermissionError: If write fails due to permissions
    """
    if not podcast_dir.exists():
        raise FileNotFoundError(f"Podcast directory not found: {podcast_dir}")
    
    outline_path = podcast_dir / "outline.json"
    
    try:
        _ = outline_path.write_text(outline.model_dump_json(indent=2))
    except PermissionError as e:
        raise PermissionError(f"Cannot write to {outline_path}: {e}") from e
    except OSError as e:
        raise OSError(f"Error saving outline: {e}") from e
    
    return outline_path


def save_transcript(transcript: Transcript, podcast_dir: Path) -> Path:
    """
    Save transcript to transcript.json.
    
    Args:
        transcript: Transcript model instance
        podcast_dir: Path to podcast directory
        
    Returns:
        Path to saved transcript.json
        
    Raises:
        FileNotFoundError: If podcast directory doesn't exist
        PermissionError: If write fails due to permissions
    """
    if not podcast_dir.exists():
        raise FileNotFoundError(f"Podcast directory not found: {podcast_dir}")
    
    transcript_path = podcast_dir / "transcript.json"
    
    try:
        _ = transcript_path.write_text(transcript.model_dump_json(indent=2))
    except PermissionError as e:
        raise PermissionError(f"Cannot write to {transcript_path}: {e}") from e
    except OSError as e:
        raise OSError(f"Error saving transcript: {e}") from e
    
    return transcript_path


def save_metadata(metadata: PodcastMetadata, podcast_dir: Path) -> Path:
    """
    Save metadata to metadata.json.
    
    Args:
        metadata: PodcastMetadata model instance
        podcast_dir: Path to podcast directory
        
    Returns:
        Path to saved metadata.json
        
    Raises:
        FileNotFoundError: If podcast directory doesn't exist
        PermissionError: If write fails due to permissions
    """
    if not podcast_dir.exists():
        raise FileNotFoundError(f"Podcast directory not found: {podcast_dir}")
    
    metadata_path = podcast_dir / "metadata.json"
    
    try:
        _ = metadata_path.write_text(metadata.model_dump_json(indent=2))
    except PermissionError as e:
        raise PermissionError(f"Cannot write to {metadata_path}: {e}") from e
    except OSError as e:
        raise OSError(f"Error saving metadata: {e}") from e
    
    return metadata_path


def get_podcast_list() -> list[str]:
    """
    List all podcasts in podcasts/ directory.
    
    Returns:
        List of podcast names (directory names)
        
    Raises:
        FileNotFoundError: If podcasts directory doesn't exist
    """
    if not PODCASTS_DIR.exists():
        return []
    
    try:
        podcasts = [
            d.name for d in PODCASTS_DIR.iterdir()
            if d.is_dir()
        ]
        return sorted(podcasts)
    except PermissionError as e:
        raise PermissionError(f"Cannot read podcasts directory: {e}") from e


def load_podcast_artifacts(podcast_name: str) -> dict[str, dict | None]:
    """
    Load all JSON artifacts for a podcast.
    
    Loads outline.json, transcript.json, and metadata.json.
    
    Args:
        podcast_name: Name of the podcast
        
    Returns:
        Dictionary with keys: outline, transcript, metadata
        Each value is parsed JSON or None if file doesn't exist
        
    Raises:
        FileNotFoundError: If podcast directory doesn't exist
        json.JSONDecodeError: If JSON files are malformed
        PermissionError: If read fails due to permissions
    """
    sanitized_name = sanitize_podcast_name(podcast_name)
    podcast_dir = PODCASTS_DIR / sanitized_name
    
    if not podcast_dir.exists():
        raise FileNotFoundError(f"Podcast directory not found: {podcast_dir}")
    
    artifacts = {
        "outline": None,
        "transcript": None,
        "metadata": None,
    }
    
    try:
        outline_path = podcast_dir / "outline.json"
        if outline_path.exists():
            artifacts["outline"] = json.loads(outline_path.read_text())
        
        transcript_path = podcast_dir / "transcript.json"
        if transcript_path.exists():
            artifacts["transcript"] = json.loads(transcript_path.read_text())
        
        metadata_path = podcast_dir / "metadata.json"
        if metadata_path.exists():
            artifacts["metadata"] = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Malformed JSON in podcast artifacts: {e.msg}",
            e.doc,
            e.pos,
        ) from e
    except PermissionError as e:
        raise PermissionError(f"Cannot read podcast artifacts: {e}") from e
    except OSError as e:
        raise OSError(f"Error loading podcast artifacts: {e}") from e
    
    return artifacts


if __name__ == "__main__":
    print("Testing storage.py...")
    
    print("\n1. Testing sanitize_podcast_name...")
    _ = sanitize_podcast_name("My Podcast")
    assert sanitize_podcast_name("My Podcast") == "My_Podcast"
    assert sanitize_podcast_name("Tech Talk!") == "Tech_Talk"
    assert sanitize_podcast_name("Pod/Cast") == "PodCast"
    print("   ✓ Name sanitization works")
    
    print("\n2. Testing invalid name handling...")
    try:
        sanitize_podcast_name("///")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("   ✓ Invalid names rejected")
    
    print("\n3. Testing create_podcast_directory...")
    test_podcast_dir = create_podcast_directory("Test_Podcast")
    assert test_podcast_dir.exists()
    assert (test_podcast_dir / "clips").exists()
    print(f"   ✓ Created directory: {test_podcast_dir}")
    
    print("\n4. Testing save/load artifacts...")
    from podcast_models import Dialogue, Outline, Segment, PodcastMetadata
    
    outline = Outline(
        segments=[
            Segment(title="Intro", description="Welcome", size="short"),
            Segment(title="Main", description="Discussion", size="medium"),
        ]
    )
    transcript = Transcript(
        dialogues=[
            Dialogue(speaker="Host", text="Welcome to the show."),
            Dialogue(speaker="Guest", text="Thanks for having me."),
        ]
    )
    metadata = PodcastMetadata(
        title="Test Episode",
        description="A test episode",
        language="en",
        tags=["test"],
    )
    
    _ = save_outline(outline, test_podcast_dir)
    _ = save_transcript(transcript, test_podcast_dir)
    _ = save_metadata(metadata, test_podcast_dir)
    print("   ✓ Saved outline, transcript, metadata")
    
    artifacts = load_podcast_artifacts("Test_Podcast")
    assert artifacts["outline"] is not None
    assert artifacts["transcript"] is not None
    assert artifacts["metadata"] is not None
    assert artifacts["metadata"]["title"] == "Test Episode"
    print("   ✓ Loaded all artifacts successfully")
    
    # Test 5: Get podcast list
    print("\n5. Testing get_podcast_list...")
    podcasts = get_podcast_list()
    assert "Test_Podcast" in podcasts
    print(f"   ✓ Found podcasts: {podcasts}")
    
    print("\n6. Testing error handling...")
    try:
        _ = load_podcast_artifacts("NonExistent_Podcast")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("   ✓ FileNotFoundError raised for missing podcast")
    
    try:
        _ = save_outline(outline, Path("nonexistent/path"))
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("   ✓ FileNotFoundError raised for missing directory")
    
    import shutil
    shutil.rmtree(test_podcast_dir)
    print("\n   ✓ Cleaned up test directory")
    
    print("\n✅ All tests passed!")

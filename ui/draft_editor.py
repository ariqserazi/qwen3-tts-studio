"""Functions to apply user edits to podcast outlines and transcripts."""

from typing import Any

from pydantic import ValidationError

from podcast.models import Dialogue, Outline, Segment, Transcript


class EditValidationError(Exception):
    """Raised when an edit fails validation."""

    pass


def apply_outline_edits(original_outline: Outline, edited_data: dict) -> Outline:
    """
    Apply user edits to an outline.

    Args:
        original_outline: The original Outline model.
        edited_data: Dictionary with structure:
            {
                "segments": [
                    {
                        "index": 0,
                        "title": "New Title",  # optional
                        "description": "New Description",  # optional
                        "size": "medium"  # optional
                    },
                    ...
                ]
            }

    Returns:
        Updated Outline model with edits applied.

    Raises:
        EditValidationError: If edits are invalid or incomplete.
    """
    if not isinstance(edited_data, dict):
        raise EditValidationError("edited_data must be a dictionary.")

    if "segments" not in edited_data:
        raise EditValidationError("edited_data must contain 'segments' key.")

    segments_edits = edited_data["segments"]
    if not isinstance(segments_edits, list):
        raise EditValidationError("'segments' must be a list.")

    # Create a mutable copy of segments
    updated_segments = [
        Segment(title=seg.title, description=seg.description, size=seg.size)
        for seg in original_outline.segments
    ]

    # Apply edits
    for edit in segments_edits:
        if not isinstance(edit, dict):
            raise EditValidationError("Each segment edit must be a dictionary.")

        if "index" not in edit:
            raise EditValidationError("Each segment edit must have an 'index' field.")

        index = edit["index"]
        if not isinstance(index, int) or index < 0 or index >= len(updated_segments):
            raise EditValidationError(
                f"Segment index {index} out of range [0, {len(updated_segments) - 1}]."
            )

        segment = updated_segments[index]

        # Update title if provided
        if "title" in edit:
            title = edit["title"]
            if not isinstance(title, str):
                raise EditValidationError(f"Segment title must be a string, got {type(title).__name__}.")
            segment.title = title.strip()

        # Update description if provided
        if "description" in edit:
            description = edit["description"]
            if not isinstance(description, str):
                raise EditValidationError(
                    f"Segment description must be a string, got {type(description).__name__}."
                )
            segment.description = description.strip()

        # Update size if provided
        if "size" in edit:
            size = edit["size"]
            if not isinstance(size, str):
                raise EditValidationError(f"Segment size must be a string, got {type(size).__name__}.")
            segment.size = size.strip().lower()

        # Validate the updated segment
        try:
            updated_segments[index] = Segment(
                title=segment.title, description=segment.description, size=segment.size
            )
        except ValidationError as e:
            raise EditValidationError(f"Segment {index} validation failed: {e}") from e

    # Create and return the updated outline
    try:
        return Outline(segments=updated_segments)
    except ValidationError as e:
        raise EditValidationError(f"Outline validation failed: {e}") from e


def apply_transcript_edits(original_transcript: Transcript, edited_data: dict) -> Transcript:
    """
    Apply user edits to a transcript.

    Args:
        original_transcript: The original Transcript model.
        edited_data: Dictionary with structure:
            {
                "dialogues": [
                    {
                        "index": 0,
                        "text": "New dialogue text"  # optional, speaker cannot change
                    },
                    ...
                ]
            }

    Returns:
        Updated Transcript model with edits applied.

    Raises:
        EditValidationError: If edits are invalid, speaker changes attempted, or text is unreasonable.
    """
    if not isinstance(edited_data, dict):
        raise EditValidationError("edited_data must be a dictionary.")

    if "dialogues" not in edited_data:
        raise EditValidationError("edited_data must contain 'dialogues' key.")

    dialogues_edits = edited_data["dialogues"]
    if not isinstance(dialogues_edits, list):
        raise EditValidationError("'dialogues' must be a list.")

    # Create a mutable copy of dialogues
    updated_dialogues = [
        Dialogue(speaker=dlg.speaker, text=dlg.text) for dlg in original_transcript.dialogues
    ]

    # Apply edits
    for edit in dialogues_edits:
        if not isinstance(edit, dict):
            raise EditValidationError("Each dialogue edit must be a dictionary.")

        if "index" not in edit:
            raise EditValidationError("Each dialogue edit must have an 'index' field.")

        index = edit["index"]
        if not isinstance(index, int) or index < 0 or index >= len(updated_dialogues):
            raise EditValidationError(
                f"Dialogue index {index} out of range [0, {len(updated_dialogues) - 1}]."
            )

        dialogue = updated_dialogues[index]

        # Check if speaker is being changed (not allowed)
        if "speaker" in edit:
            new_speaker = edit["speaker"]
            if new_speaker.strip() != dialogue.speaker:
                raise EditValidationError(
                    f"Cannot change speaker for dialogue {index}. "
                    f"Original: '{dialogue.speaker}', Attempted: '{new_speaker}'."
                )

        # Update text if provided
        if "text" in edit:
            text = edit["text"]
            if not isinstance(text, str):
                raise EditValidationError(
                    f"Dialogue text must be a string, got {type(text).__name__}."
                )

            text = text.strip()

            # Validate dialogue length is reasonable (not empty, not excessively long)
            if not text:
                raise EditValidationError(f"Dialogue {index} text cannot be empty.")

            if len(text) > 5000:
                raise EditValidationError(
                    f"Dialogue {index} text is too long ({len(text)} chars, max 5000)."
                )

            dialogue.text = text

        # Validate the updated dialogue
        try:
            updated_dialogues[index] = Dialogue(speaker=dialogue.speaker, text=dialogue.text)
        except ValidationError as e:
            raise EditValidationError(f"Dialogue {index} validation failed: {e}") from e

    # Create and return the updated transcript
    try:
        return Transcript(dialogues=updated_dialogues)
    except ValidationError as e:
        raise EditValidationError(f"Transcript validation failed: {e}") from e


if __name__ == "__main__":
    from podcast_models import Outline, Segment, Transcript, Dialogue

    # Create sample outline
    original_outline = Outline(
        segments=[
            Segment(title="Intro", description="Welcome and setup.", size="short"),
            Segment(title="Main Topic", description="Deep dive discussion.", size="medium"),
            Segment(title="Outro", description="Closing remarks.", size="short"),
        ]
    )

    # Test 1: Update outline segment title and description
    print("Test 1: Update outline segment title and description")
    outline_edits = {
        "segments": [
            {"index": 0, "title": "Introduction"},
            {"index": 1, "description": "Comprehensive analysis of the topic.", "size": "long"},
        ]
    }
    updated_outline = apply_outline_edits(original_outline, outline_edits)
    assert updated_outline.segments[0].title == "Introduction"
    assert updated_outline.segments[1].description == "Comprehensive analysis of the topic."
    assert updated_outline.segments[1].size == "long"
    print("✓ Passed")

    # Test 2: Invalid outline edit (out of range index)
    print("\nTest 2: Invalid outline edit (out of range index)")
    try:
        bad_edits = {"segments": [{"index": 10, "title": "Bad"}]}
        _ = apply_outline_edits(original_outline, bad_edits)
        print("✗ Failed - should have raised EditValidationError")
    except EditValidationError as e:
        print(f"✓ Passed - caught error: {e}")

    # Test 3: Invalid outline edit (bad size)
    print("\nTest 3: Invalid outline edit (bad size)")
    try:
        bad_edits = {"segments": [{"index": 0, "size": "huge"}]}
        _ = apply_outline_edits(original_outline, bad_edits)
        print("✗ Failed - should have raised EditValidationError")
    except EditValidationError as e:
        print(f"✓ Passed - caught error: {e}")

    # Create sample transcript
    original_transcript = Transcript(
        dialogues=[
            Dialogue(speaker="Alex", text="Welcome to the show."),
            Dialogue(speaker="Riley", text="Thanks for having me."),
            Dialogue(speaker="Alex", text="Let's dive into the topic."),
        ]
    )

    # Test 4: Update transcript dialogue text
    print("\nTest 4: Update transcript dialogue text")
    transcript_edits = {
        "dialogues": [
            {"index": 0, "text": "Welcome to our podcast!"},
            {"index": 2, "text": "Now let's explore this fascinating topic."},
        ]
    }
    updated_transcript = apply_transcript_edits(original_transcript, transcript_edits)
    assert updated_transcript.dialogues[0].text == "Welcome to our podcast!"
    assert updated_transcript.dialogues[2].text == "Now let's explore this fascinating topic."
    assert updated_transcript.dialogues[0].speaker == "Alex"  # Speaker unchanged
    print("✓ Passed")

    # Test 5: Attempt to change speaker (should fail)
    print("\nTest 5: Attempt to change speaker (should fail)")
    try:
        bad_edits = {"dialogues": [{"index": 0, "speaker": "Jordan"}]}
        _ = apply_transcript_edits(original_transcript, bad_edits)
        print("✗ Failed - should have raised EditValidationError")
    except EditValidationError as e:
        print(f"✓ Passed - caught error: {e}")

    # Test 6: Invalid transcript edit (out of range index)
    print("\nTest 6: Invalid transcript edit (out of range index)")
    try:
        bad_edits = {"dialogues": [{"index": 100, "text": "Bad"}]}
        _ = apply_transcript_edits(original_transcript, bad_edits)
        print("✗ Failed - should have raised EditValidationError")
    except EditValidationError as e:
        print(f"✓ Passed - caught error: {e}")

    # Test 7: Empty dialogue text (should fail)
    print("\nTest 7: Empty dialogue text (should fail)")
    try:
        bad_edits = {"dialogues": [{"index": 0, "text": "   "}]}
        _ = apply_transcript_edits(original_transcript, bad_edits)
        print("✗ Failed - should have raised EditValidationError")
    except EditValidationError as e:
        print(f"✓ Passed - caught error: {e}")

    # Test 8: Dialogue text too long (should fail)
    print("\nTest 8: Dialogue text too long (should fail)")
    try:
        bad_edits = {"dialogues": [{"index": 0, "text": "x" * 6000}]}
        _ = apply_transcript_edits(original_transcript, bad_edits)
        print("✗ Failed - should have raised EditValidationError")
    except EditValidationError as e:
        print(f"✓ Passed - caught error: {e}")

    # Test 9: Missing required fields
    print("\nTest 9: Missing required fields in edits")
    try:
        bad_edits = {"segments": [{"title": "No index"}]}
        _ = apply_outline_edits(original_outline, bad_edits)
        print("✗ Failed - should have raised EditValidationError")
    except EditValidationError as e:
        print(f"✓ Passed - caught error: {e}")

    # Test 10: Complex multi-edit scenario
    print("\nTest 10: Complex multi-edit scenario")
    complex_edits = {
        "segments": [
            {"index": 0, "title": "Opening", "description": "Warm welcome"},
            {"index": 2, "size": "medium"},
        ]
    }
    updated_outline = apply_outline_edits(original_outline, complex_edits)
    assert updated_outline.segments[0].title == "Opening"
    assert updated_outline.segments[0].description == "Warm welcome"
    assert updated_outline.segments[2].size == "medium"
    print("✓ Passed")

    print("\n" + "=" * 50)
    print("All tests passed!")

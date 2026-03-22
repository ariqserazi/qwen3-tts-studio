"""Batch audio processing with progress callbacks and error recovery."""

import gc
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from audio.generator import generate_dialogue_audio
from podcast.models import Dialogue, SpeakerProfile, Transcript

# Retry configuration for TTS generation
MAX_RETRIES = 3  # Maximum retry attempts (total 4 attempts including initial)
RETRY_BACKOFF = (5, 10, 20)  # Exponential backoff delays in seconds


def generate_all_clips(
    transcript: Transcript,
    speaker_profile: SpeakerProfile,
    params: dict[str, Any],
    clips_dir: str | Path,
    progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> list[str]:
    """
    Generate audio clips for all dialogues in a transcript sequentially.

    Processes dialogues one at a time, respecting MPS device limits. Calls
    progress_callback after each clip generation. Saves clips with zero-padded
    filenames (0000.wav, 0001.wav, etc.).

    Error handling policy:
    - Individual clip generation failures are logged and skipped
    - Failed clips are tracked and reported in summary
    - Returns list of successfully generated clip paths
    - Raises RuntimeError only if ALL clips fail or output directory cannot be created

    Args:
        transcript: Transcript with dialogue list to process.
        speaker_profile: SpeakerProfile with voice mappings for speakers.
        params: TTS parameters dict with keys:
            - model_name: str (e.g., "Qwen3-TTS-12Hz-1.7B-Base")
            - temperature: float
            - top_k: int
            - top_p: float
            - repetition_penalty: float
            - max_new_tokens: int
            - subtalker_temperature: float
            - subtalker_top_k: int
            - subtalker_top_p: float
            - language: str (e.g., "en")
            - instruct: str | None (optional instruction)
        clips_dir: Directory where audio files will be saved.
        progress_callback: Optional callback function called after each clip.
            Signature: progress_callback(current: int, total: int, segment_info: dict)
            segment_info contains: {
                "index": int,
                "speaker": str,
                "text": str,
                "filename": str,
                "status": "success" | "error",
                "error": str | None,
                "path": str | None,
            }

    Returns:
        List of paths to successfully generated audio files.

    Raises:
        ValueError: If transcript is empty or speaker_profile is invalid.
        RuntimeError: If all clips fail to generate or output directory cannot be created.
    """
    # Validate inputs
    if not transcript.dialogues:
        raise ValueError("Transcript must contain at least one dialogue.")

    clips_dir = Path(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)

    total_clips = len(transcript.dialogues)
    clip_paths = []
    failed_indices = []

    print(f"\n{'='*60}")
    print(f"Batch Audio Generation: {total_clips} clips")
    print(f"Output directory: {clips_dir}")
    print(f"{'='*60}\n")

    for idx, dialogue in enumerate(transcript.dialogues):
        current = idx + 1
        filename = f"{idx:04d}.wav"
        output_path = clips_dir / filename

        segment_info = {
            "index": idx,
            "speaker": dialogue.speaker,
            "text": dialogue.text[:50] + "..." if len(dialogue.text) > 50 else dialogue.text,
            "filename": filename,
            "status": None,
            "error": None,
            "path": None,
        }

        print(f"[{current:3d}/{total_clips}] Generating: {dialogue.speaker}")
        print(f"         Text: {segment_info['text']}")

        segment_info["status"] = "started"
        if progress_callback is not None:
            progress_callback(current, total_clips, segment_info)

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                path = generate_dialogue_audio(dialogue, speaker_profile, params, output_path)

                segment_info["status"] = "success"
                segment_info["path"] = path
                clip_paths.append(path)

                print(f"         ✓ Saved to: {filename}\n")
                break

            except (TimeoutError, RuntimeError) as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_BACKOFF[attempt]
                    print(f"         ⚠ Clip {idx} failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}")
                    print(f"         Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    # All retries exhausted - log and continue to next clip
                    error_msg = f"Failed after {MAX_RETRIES + 1} attempts: {last_error}"
                    segment_info["status"] = "error"
                    segment_info["error"] = error_msg
                    failed_indices.append(idx)

                    # Create crash report with diagnostic info
                    error_report = {
                        "clip_index": idx,
                        "speaker": dialogue.speaker,
                        "text_length": len(dialogue.text),
                        "error": str(last_error),
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.now().isoformat(),
                        "retry_attempts": MAX_RETRIES + 1,
                    }

                    # Add memory stats if psutil available
                    try:
                        import psutil

                        error_report["memory_percent"] = psutil.virtual_memory().percent
                    except ImportError:
                        pass  # psutil not available, skip memory stats

                    # Save crash report to clips directory
                    crash_log = clips_dir / f"crash_clip_{idx:04d}.json"
                    crash_log.write_text(json.dumps(error_report, indent=2))
                    print(f"         ✗ Crash report saved to: {crash_log}")
                    print(f"         ✗ Skipping clip {idx}\n")
                    break

            except Exception as e:
                error_msg = str(e)
                segment_info["status"] = "error"
                segment_info["error"] = error_msg
                failed_indices.append(idx)

                print(f"         ✗ Error: {error_msg}\n")
                break

        # Memory cleanup after clip generation (success or failure)
        # Prevents memory exhaustion on long podcasts with large TTS models
        if segment_info["status"] == "success":
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except ImportError:
                pass  # torch not available, skip GPU cleanup

        # Call progress callback if provided
        if progress_callback is not None:
            progress_callback(current, total_clips, segment_info)

    # Summary
    print(f"{'='*60}")
    print(f"Batch Generation Complete")
    print(f"  Total clips: {total_clips}")
    print(f"  Successful: {len(clip_paths)}")
    print(f"  Failed: {len(failed_indices)}")
    if failed_indices:
        print(f"  Failed indices: {failed_indices}")
    print(f"{'='*60}\n")

    if not clip_paths:
        raise RuntimeError(
            f"All {total_clips} audio clips failed to generate. "
            f"Check model configuration and speaker profiles."
        )

    return clip_paths


if __name__ == "__main__":
    """Test batch processor with mock transcript."""
    from podcast_models import Dialogue, Speaker, SpeakerProfile, Transcript

    print("=== Batch Processor Test ===\n")

    # Create test speaker profile
    speakers = [
        Speaker(name="Alice", voice_id="male_1", role="Host", type="preset"),
        Speaker(name="Bob", voice_id="female_1", role="Guest", type="preset"),
    ]
    profile = SpeakerProfile(speakers=speakers)

    # Create test transcript with multiple dialogues
    dialogues = [
        Dialogue(speaker="Alice", text="Welcome to the podcast."),
        Dialogue(speaker="Bob", text="Thanks for having me."),
        Dialogue(speaker="Alice", text="Let's dive into the topic."),
        Dialogue(speaker="Bob", text="I'm excited to discuss this."),
        Dialogue(speaker="Alice", text="Great! Let's begin."),
    ]
    transcript = Transcript(dialogues=dialogues)

    # TTS parameters
    tts_params = {
        "model_name": "Qwen3-TTS-12Hz-1.7B-Base",
        "temperature": 0.3,
        "top_k": 50,
        "top_p": 0.85,
        "repetition_penalty": 1.0,
        "max_new_tokens": 1024,
        "subtalker_temperature": 0.3,
        "subtalker_top_k": 50,
        "subtalker_top_p": 0.85,
        "language": "en",
        "instruct": None,
    }

    # Define progress callback
    def progress_callback(current: int, total: int, segment_info: dict[str, Any]) -> None:
        """Log progress to console."""
        status_icon = "✓" if segment_info["status"] == "success" else "✗"
        print(f"  [{status_icon}] Progress: {current}/{total} - {segment_info['speaker']}")
        if segment_info["error"]:
            print(f"      Error: {segment_info['error']}")

    # Test 1: Batch generation with progress callback
    print("Test 1: Batch generation with progress callback")
    try:
        output_dir = Path("test_batch_output")
        clip_paths = generate_all_clips(
            transcript, profile, tts_params, output_dir, progress_callback
        )
        print(f"✓ Generated {len(clip_paths)} clips")
        for path in clip_paths:
            print(f"  - {path}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 2: Empty transcript error handling
    print("\nTest 2: Empty transcript error handling")
    try:
        empty_transcript = Transcript(dialogues=[])
        clip_paths = generate_all_clips(empty_transcript, profile, tts_params, "test_output")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")

    # Test 3: Batch generation without callback
    print("\nTest 3: Batch generation without callback")
    try:
        output_dir = Path("test_batch_output_no_callback")
        clip_paths = generate_all_clips(transcript, profile, tts_params, output_dir)
        print(f"✓ Generated {len(clip_paths)} clips without callback")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\n=== Tests completed ===")

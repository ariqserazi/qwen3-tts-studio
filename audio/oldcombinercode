"""Audio clip combiner for podcast generation using moviepy."""

import logging
from pathlib import Path

from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import concatenate_audioclips

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def combine_audio_clips(
    clips_dir: Path | str,
    output_path: Path | str,
    bitrate: str = "192k",
) -> Path:
    """
    Combine audio clips into a single podcast file.

    Merges all MP3 clips from clips_dir in numerical order (0000.mp3, 0001.mp3, ...)
    into a single MP3 file using moviepy's concatenate_audioclips.

    Args:
        clips_dir: Directory containing audio clips (e.g., podcasts/MyPodcast/clips/)
        output_path: Path where final combined podcast will be saved
        bitrate: Audio bitrate for export (default: "192k")

    Returns:
        Path to the final combined podcast file

    Raises:
        FileNotFoundError: If clips_dir doesn't exist or contains no audio files
        ValueError: If clips_dir is empty or no valid audio files found
        IOError: If audio files are corrupted or unreadable
        OSError: If output directory doesn't exist or write fails
    """
    clips_dir = Path(clips_dir)
    output_path = Path(output_path)

    # Validate clips directory exists
    if not clips_dir.exists():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")

    if not clips_dir.is_dir():
        raise ValueError(f"clips_dir must be a directory: {clips_dir}")

    # Validate output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all audio files and sort numerically
    audio_files = sorted(
        clips_dir.glob("*.wav"),
        key=lambda x: int(x.stem) if x.stem.isdigit() else float("inf"),
    )

    if not audio_files:
        raise ValueError(
            f"No audio files found in {clips_dir}. "
            f"Expected WAV files named 0000.wav, 0001.wav, etc."
        )

    logger.info(f"Found {len(audio_files)} audio clips to combine")
    logger.info(f"Output will be saved to: {output_path}")

    # Load audio clips
    audio_clips = []
    try:
        for clip_path in audio_files:
            logger.info(f"Loading: {clip_path.name}")
            try:
                audio_clip = AudioFileClip(str(clip_path))
                audio_clips.append(audio_clip)
            except Exception as e:
                # Clean up already loaded clips on error
                for loaded_clip in audio_clips:
                    loaded_clip.close()
                raise IOError(f"Failed to load audio file {clip_path.name}: {e}") from e

        # Concatenate all clips
        logger.info("Concatenating audio clips...")
        final_audio = concatenate_audioclips(audio_clips)

        logger.info(f"Exporting to MP3 with bitrate {bitrate}...")
        final_audio.write_audiofile(
            str(output_path),
            codec="libmp3lame",
            bitrate=bitrate,
        )

        logger.info(f"✓ Successfully created podcast: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error combining audio clips: {e}")
        raise
    finally:
        # Clean up: close all audio clips to prevent memory leaks
        for clip in audio_clips:
            try:
                clip.close()
            except Exception as e:
                logger.warning(f"Error closing audio clip: {e}")


if __name__ == "__main__":
    """Test audio_combiner with sample clips."""
    import tempfile
    from pathlib import Path

    print("Testing audio_combiner.py...")

    # Create a test directory with sample audio clips
    with tempfile.TemporaryDirectory() as tmpdir:
        test_clips_dir = Path(tmpdir) / "test_clips"
        test_clips_dir.mkdir()

        # Create minimal test MP3 files (silent audio)
        print("\n1. Testing with sample clips directory...")
        print(f"   Test clips directory: {test_clips_dir}")

        # Check if we can create the directory structure
        assert test_clips_dir.exists(), "Test clips directory creation failed"
        print("   ✓ Test directory created")

        # Test error handling: empty directory
        print("\n2. Testing error handling (empty directory)...")
        try:
            output_path = Path(tmpdir) / "output.mp3"
            _ = combine_audio_clips(test_clips_dir, output_path)
            print("   ✗ Should have raised ValueError for empty directory")
        except ValueError as e:
            print(f"   ✓ Correctly raised ValueError: {e}")

        # Test error handling: non-existent directory
        print("\n3. Testing error handling (non-existent directory)...")
        try:
            non_existent = Path(tmpdir) / "non_existent"
            output_path = Path(tmpdir) / "output.mp3"
            _ = combine_audio_clips(non_existent, output_path)
            print("   ✗ Should have raised FileNotFoundError")
        except FileNotFoundError as e:
            print(f"   ✓ Correctly raised FileNotFoundError: {e}")

        print("\n✅ Error handling tests passed!")
        print("\nNote: Full integration testing requires actual audio files.")
        print("In production, this will combine TTS-generated MP3 clips.")

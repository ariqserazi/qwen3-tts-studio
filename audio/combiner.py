"""Audio clip combiner for podcast generation.

This version reduces clicks and glitches by:
1, forcing a consistent sample rate
2, applying short fades on every clip boundary
3, peak limiting each clip before concatenation
"""

import logging
from pathlib import Path

import numpy as np
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import AudioArrayClip, concatenate_audioclips

try:
    from moviepy.audio.fx.all import audio_fadein, audio_fadeout
except Exception:
    audio_fadein = None
    audio_fadeout = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s , %(name)s , %(levelname)s , %(message)s",
)
logger = logging.getLogger(__name__)


def _safe_int_stem(p: Path) -> int:
    try:
        return int(p.stem)
    except Exception:
        return 10**12


def _to_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == np.int16:
        return (x.astype(np.float32) / 32768.0).copy()
    if x.dtype == np.int32:
        return (x.astype(np.float32) / 2147483648.0).copy()
    return x.astype(np.float32, copy=True)


def _remove_dc(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    return x - float(np.mean(x))


def _peak_limit(x: np.ndarray, target_peak: float) -> np.ndarray:
    if x.size == 0:
        return x
    peak = float(np.max(np.abs(x)))
    if peak <= 0.0:
        return x
    if peak > target_peak:
        x = x * (target_peak / peak)
    return x


def _fade_edges_array(x: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
    if x.size == 0:
        return x
    n = int(sr * (fade_ms / 1000.0))
    n = max(1, min(n, x.shape[0] // 2))
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)

    y = x.copy()
    if y.ndim == 1:
        y[:n] *= ramp
        y[-n:] *= ramp[::-1]
        return y

    y[:n, :] *= ramp[:, None]
    y[-n:, :] *= ramp[::-1][:, None]
    return y


def _sanitize_clip_to_array(
    clip: AudioFileClip,
    target_sr: int,
    fade_ms: float,
    target_peak: float,
) -> AudioArrayClip:

    # resample to consistent sample rate
    arr = clip.to_soundarray(fps=target_sr)

    arr = _to_float32(arr)
    arr = _remove_dc(arr)
    arr = _peak_limit(arr, target_peak)
    arr = _fade_edges_array(arr, target_sr, fade_ms)

    return AudioArrayClip(arr, fps=target_sr)


def combine_audio_clips(
    clips_dir: Path | str,
    output_path: Path | str,
    bitrate: str = "192k",
    target_sr: int = 24000,
    fade_ms: float = 10.0,
    target_peak: float = 0.95,
) -> Path:
    """
    Combine WAV clips into one MP3.

    clips_dir expects files named 0000.wav, 0001.wav, etc.

    Args:
        clips_dir: folder containing wav clips
        output_path: output mp3 pathß
        bitrate: mp3 bitrate string, example 192k
        target_sr: sample rate for all clips
        fade_ms: fade in and fade out length per clip boundary
        target_peak: peak limit per clip

    Returns:
        Path to output file
    """
    clips_dir = Path(clips_dir)
    output_path = Path(output_path)

    if not clips_dir.exists():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")
    if not clips_dir.is_dir():
        raise ValueError(f"clips_dir must be a directory: {clips_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(clips_dir.glob("*.wav"), key=_safe_int_stem)
    if not audio_files:
        raise ValueError(
            f"No audio files found in {clips_dir}. Expected 0000.wav, 0001.wav, etc."
        )

    logger.info(f"Found {len(audio_files)} audio clips")
    logger.info(f"Target sample rate: {target_sr}")
    logger.info(f"Fade per clip edge: {fade_ms} ms")
    logger.info(f"Peak limit: {target_peak}")
    logger.info(f"Output path: {output_path}")

    raw_clips: list[AudioFileClip] = []
    clean_clips: list[AudioArrayClip] = []

    try:
        for p in audio_files:
            logger.info(f"Loading: {p.name}")
            raw = AudioFileClip(str(p))
            raw_clips.append(raw)

            clean = _sanitize_clip_to_array(
                raw,
                target_sr=target_sr,
                fade_ms=fade_ms,
                target_peak=target_peak,
            )
            clean_clips.append(clean)

        logger.info("Concatenating clips...")
        final_audio = concatenate_audioclips(clean_clips)

        logger.info(f"Exporting mp3, bitrate {bitrate}...")
        final_audio.write_audiofile(
            str(output_path),
            codec="libmp3lame",
            bitrate=bitrate,
            fps=target_sr,
        )

        logger.info(f"Done: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error combining clips: {e}")
        raise

    finally:
        for c in clean_clips:
            try:
                c.close()
            except Exception:
                pass
        for c in raw_clips:
            try:
                c.close()
            except Exception:
                pass
        try:
            final_audio.close()  # type: ignore[name-defined]
        except Exception:
            pass


if __name__ == "__main__":
    import tempfile

    print("Testing combiner.py")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_clips_dir = Path(tmpdir) / "test_clips"
        test_clips_dir.mkdir()

        try:
            output_path = Path(tmpdir) / "output.mp3"
            _ = combine_audio_clips(test_clips_dir, output_path)
            print("Expected error for empty directory, but did not get one")
        except ValueError as e:
            print(f"Got expected ValueError: {e}")

        try:
            missing_dir = Path(tmpdir) / "missing"
            output_path = Path(tmpdir) / "output.mp3"
            _ = combine_audio_clips(missing_dir, output_path)
            print("Expected error for missing directory, but did not get one")
        except FileNotFoundError as e:
            print(f"Got expected FileNotFoundError: {e}")

        print("Done")
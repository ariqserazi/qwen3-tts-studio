"""Audio generation for podcast dialogue using Qwen3-TTS."""

import copy
import json
import pickle
import threading
from contextlib import contextmanager
from math import ceil
from pathlib import Path
from typing import Any, Generator

import numpy as np
import soundfile as sf
import torch

# Timeout for TTS generation (10 minutes per clip for MPS/MacBook)
TTS_TIMEOUT_SECONDS = 600


@contextmanager
def timeout_handler(
    seconds: int, error_context: str = ""
) -> Generator[None, None, None]:
    """Context manager for timeout protection.

    Uses threading.Timer for thread-safe timeout support across all platforms.
    Note: This implementation sets a flag on timeout but cannot interrupt
    blocking operations. The timeout is checked after the operation completes.
    For true interruption of long-running TTS, the model itself would need
    timeout support.

    Args:
        seconds: Timeout duration in seconds.
        error_context: Additional context for error message (e.g., clip index, speaker).

    Yields:
        None

    Raises:
        TimeoutError: If the operation exceeds the timeout duration.
    """
    # Thread-safe timeout using threading.Timer (works in any thread)
    timeout_occurred = threading.Event()

    def timeout_trigger() -> None:
        timeout_occurred.set()

    timer = threading.Timer(seconds, timeout_trigger)
    timer.start()
    try:
        yield
        # Check if timeout occurred during the operation
        if timeout_occurred.is_set():
            raise TimeoutError(
                f"TTS generation timed out after {seconds}s. {error_context}"
            )
    finally:
        timer.cancel()


from podcast.models import Dialogue, SpeakerProfile, Transcript

SAVED_VOICES_DIR = Path("saved_voices")


def _get_model_dtype_device(model: Any) -> tuple[torch.dtype, torch.device]:
    """Get model's dtype and device from talker module (most reliable for Qwen3-TTS)."""
    hf = getattr(model, "model", model)
    talker = getattr(hf, "talker", None)

    target = talker if talker is not None else hf

    try:
        param = next(target.parameters())
        return param.dtype, param.device
    except (StopIteration, AttributeError):
        pass

    if torch.backends.mps.is_available():
        return torch.float16, torch.device("mps")
    elif torch.cuda.is_available():
        return torch.bfloat16, torch.device("cuda")
    return torch.float32, torch.device("cpu")


def _prepare_voice_clone_prompt(voice_clone_prompt: Any, model: Any) -> Any:
    """Normalize voice clone prompt dtype/device to match model."""
    model_dtype, model_device = _get_model_dtype_device(model)

    def convert_item(item: Any) -> Any:
        if hasattr(item, "ref_spk_embedding") and isinstance(
            item.ref_spk_embedding, torch.Tensor
        ):
            if (
                item.ref_spk_embedding.dtype != model_dtype
                or item.ref_spk_embedding.device != model_device
            ):
                item.ref_spk_embedding = item.ref_spk_embedding.to(
                    dtype=model_dtype, device=model_device
                )
        if hasattr(item, "ref_code") and isinstance(item.ref_code, torch.Tensor):
            if item.ref_code.device != model_device:
                item.ref_code = item.ref_code.to(device=model_device)
        return item

    if isinstance(voice_clone_prompt, list):
        return [convert_item(copy.copy(item)) for item in voice_clone_prompt]
    elif isinstance(voice_clone_prompt, dict):
        result = voice_clone_prompt.copy()
        if "ref_spk_embedding" in result:
            emb = result["ref_spk_embedding"]
            if isinstance(emb, list):
                result["ref_spk_embedding"] = [
                    e.to(dtype=model_dtype, device=model_device)
                    if isinstance(e, torch.Tensor)
                    else e
                    for e in emb
                ]
            elif isinstance(emb, torch.Tensor):
                result["ref_spk_embedding"] = emb.to(
                    dtype=model_dtype, device=model_device
                )
        if "ref_code" in result:
            code = result["ref_code"]
            if isinstance(code, list):
                result["ref_code"] = [
                    c.to(device=model_device) if isinstance(c, torch.Tensor) else c
                    for c in code
                ]
            elif isinstance(code, torch.Tensor):
                result["ref_code"] = code.to(device=model_device)
        return result
    else:
        return convert_item(copy.copy(voice_clone_prompt))


LANGUAGE_MAP = {
    "en": "english",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "es": "spanish",
}


def _normalize_language(lang: str) -> str:
    if lang in LANGUAGE_MAP:
        return LANGUAGE_MAP[lang]
    return lang


CHUNK_TARGET = 120
CHUNK_MAX = 150
CHUNK_MIN = 50


def _split_text_into_chunks(text: str) -> list[str]:
    """Split long text into sentence-based chunks for TTS generation."""
    text = text.strip()
    if len(text) <= CHUNK_MAX:
        return [text]

    import re

    sentences = re.split(r"(?<=[.!?。！？])\s*", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text]

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(sentence) > CHUNK_MAX:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            words = sentence.split()
            temp = ""
            for word in words:
                if len(temp) + len(word) + 1 <= CHUNK_TARGET:
                    temp = f"{temp} {word}".strip()
                else:
                    if temp:
                        chunks.append(temp)
                    temp = word
            if temp:
                chunks.append(temp)
            continue

        if len(current_chunk) + len(sentence) + 1 <= CHUNK_TARGET:
            current_chunk = f"{current_chunk} {sentence}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    merged = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        while i + 1 < len(chunks) and len(chunk) < CHUNK_MIN:
            i += 1
            chunk = f"{chunk} {chunks[i]}"
        merged.append(chunk.strip())
        i += 1

    return merged if merged else [text]


def _check_trailing_silence(
    audio: np.ndarray,
    sr: int,
    segment_sec: float = 0.5,
    min_speech_segments: int = 2,
    min_trailing_silent_sec: float = 2.0,
) -> tuple[bool, float]:
    """
    Detect audio truncation: speech followed by sustained trailing silence.

    More accurate than global silent ratio - targets the specific truncation pattern
    where audio starts normally then becomes silent due to premature EOS.

    Args:
        audio: Audio array (float32, normalized to [-1, 1])
        sr: Sample rate
        segment_sec: Window size for analysis (0.5s is responsive yet stable)
        min_speech_segments: Minimum non-silent segments required at start
        min_trailing_silent_sec: Minimum trailing silence duration to flag as truncated

    Returns:
        Tuple of (is_truncated, trailing_silence_ratio)
    """
    segment_samples = int(sr * segment_sec)
    min_trailing_segments = int(min_trailing_silent_sec / segment_sec)

    if len(audio) < segment_samples * (min_speech_segments + min_trailing_segments):
        return False, 0.0

    num_segments = len(audio) // segment_samples
    segment_rms = np.array(
        [
            np.sqrt(
                np.mean(audio[i * segment_samples : (i + 1) * segment_samples] ** 2)
            )
            for i in range(num_segments)
        ]
    )

    if len(segment_rms) == 0:
        return False, 0.0

    p95_rms = float(np.percentile(segment_rms, 95))
    abs_floor = 0.001
    adaptive_threshold = max(abs_floor, 0.02 * p95_rms)

    is_silent = segment_rms < adaptive_threshold

    speech_found = np.sum(~is_silent[:min_speech_segments]) >= min_speech_segments
    if not speech_found:
        return False, 0.0

    trailing_silent_count = 0
    for i in range(num_segments - 1, -1, -1):
        if is_silent[i]:
            trailing_silent_count += 1
        else:
            break

    trailing_silence_ratio = trailing_silent_count / num_segments
    is_truncated = trailing_silent_count >= min_trailing_segments

    return is_truncated, trailing_silence_ratio


def _crossfade_audio(
    audio1: np.ndarray, audio2: np.ndarray, sr: int, fade_ms: int = 30
) -> np.ndarray:
    """Concatenate two audio arrays with crossfade."""
    fade_samples = int(sr * fade_ms / 1000)

    if len(audio1) < fade_samples or len(audio2) < fade_samples:
        return np.concatenate([audio1, audio2])

    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in = np.linspace(0.0, 1.0, fade_samples)

    audio1_end = audio1[-fade_samples:] * fade_out
    audio2_start = audio2[:fade_samples] * fade_in
    crossfaded = audio1_end + audio2_start

    return np.concatenate([audio1[:-fade_samples], crossfaded, audio2[fade_samples:]])


def _calculate_dynamic_max_tokens(text: str, preset_max: int) -> int:
    """Calculate dynamic max_new_tokens based on text length."""
    MIN_TOKENS = 256
    MAX_TOKENS = 768

    char_count = len(text)
    estimated = ceil(char_count * 2.5)
    dynamic_max = ceil(estimated * 1.3)

    max_new = max(MIN_TOKENS, min(dynamic_max, MAX_TOKENS))

    print(
        f"[TTS] max_tokens: chars={char_count}, dynamic={dynamic_max}, final={max_new}",
        flush=True,
    )

    return max_new


def generate_dialogue_audio(
    dialogue: Dialogue,
    speaker_profile: SpeakerProfile,
    params: dict[str, Any],
    output_path: str | Path,
) -> str:
    """
    Generate audio for a single dialogue line using Qwen3-TTS.

    Args:
        dialogue: Dialogue instance with speaker name and text.
        speaker_profile: SpeakerProfile containing speaker voice mappings.
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
        output_path: Path where audio file will be saved.

    Returns:
        Path to the generated audio file.

    Raises:
        ValueError: If speaker not found in profile or voice type is invalid.
        RuntimeError: If TTS generation fails or device issues occur.
    """
    # Find speaker in profile
    speaker = None
    for s in speaker_profile.speakers:
        if s.name.lower() == dialogue.speaker.lower():
            speaker = s
            break

    if speaker is None:
        raise ValueError(
            f"Speaker '{dialogue.speaker}' not found in profile. "
            f"Available: {', '.join(s.name for s in speaker_profile.speakers)}"
        )

    # Validate voice type
    if speaker.type not in ("preset", "saved"):
        raise ValueError(
            f"Invalid voice type: {speaker.type}. Must be 'preset' or 'saved'."
        )

    base_model_name = params.get("model_name", "1.7B-CustomVoice")

    if speaker.type == "saved":
        voice_meta_path = SAVED_VOICES_DIR / speaker.voice_id / "metadata.json"
        if voice_meta_path.exists():
            with open(voice_meta_path) as f:
                voice_meta = json.load(f)
                model_name = voice_meta.get("model", "1.7B-Base")
        else:
            model_name = base_model_name.replace("CustomVoice", "Base")
    else:
        model_name = base_model_name

    try:
        from audio.model_loader import get_model

        model = get_model(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {e}")

    try:
        if speaker.type == "preset":
            wavs, sr = _generate_preset_voice(
                model, dialogue.text, speaker.voice_id, params
            )
        else:
            wavs, sr = _generate_saved_voice(
                model, dialogue.text, speaker.voice_id, params
            )
    except Exception as e:
        raise RuntimeError(f"TTS generation failed for speaker '{speaker.name}': {e}")

    # Save audio to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        sf.write(str(output_path), wavs[0], sr)
    except Exception as e:
        raise RuntimeError(f"Failed to save audio to {output_path}: {e}")

    return str(output_path)


def _generate_preset_voice(
    model: Any, text: str, speaker: str, params: dict[str, Any]
) -> tuple[Any, int]:
    """
    Generate audio using a preset voice.

    Args:
        model: Qwen3-TTS model instance.
        text: Text to synthesize.
        speaker: Preset speaker name.
        params: TTS parameters.

    Returns:
        Tuple of (wavs, sample_rate).
    """
    lang = _normalize_language(params.get("language", "english"))
    print(f"[LANG] TTS normalized: {lang}", flush=True)

    # Split long text into chunks to avoid timeouts
    chunks = _split_text_into_chunks(text)

    if len(chunks) > 1:
        print(
            f"[TTS] Splitting text into {len(chunks)} chunks for speaker {speaker}",
            flush=True,
        )

    all_audio: list[np.ndarray] = []
    sr: int = 0

    for i, chunk in enumerate(chunks):
        preset_max = int(params.get("max_new_tokens", 1024))
        dynamic_max = _calculate_dynamic_max_tokens(chunk, preset_max)

        error_context = f"Speaker: {speaker}, Chunk {i + 1}/{len(chunks)}, Text length: {len(chunk)} chars"
        with timeout_handler(TTS_TIMEOUT_SECONDS, error_context):
            wavs, chunk_sr = model.generate_custom_voice(
                text=chunk,
                speaker=speaker,
                language=lang,
                instruct=params.get("instruct"),
                temperature=params.get("temperature", 0.3),
                top_k=int(params.get("top_k", 50)),
                top_p=params.get("top_p", 0.85),
                repetition_penalty=params.get("repetition_penalty", 1.0),
                max_new_tokens=dynamic_max,
                subtalker_temperature=params.get("subtalker_temperature", 0.3),
                subtalker_top_k=int(params.get("subtalker_top_k", 50)),
                subtalker_top_p=params.get("subtalker_top_p", 0.85),
            )

        if sr == 0:
            sr = int(chunk_sr)

        audio_data = wavs[0]
        if audio_data.size == 0:
            raise RuntimeError(
                f"Empty audio for preset voice {speaker}, chunk {i + 1}/{len(chunks)}"
            )

        audio_f = audio_data.astype(np.float32)
        if np.issubdtype(audio_data.dtype, np.integer):
            audio_f = audio_f / np.iinfo(audio_data.dtype).max
        audio_rms = float(np.sqrt(np.mean(audio_f * audio_f)))
        audio_peak = float(np.max(np.abs(audio_f)))

        print(
            f"[TTS] Preset chunk {i + 1}/{len(chunks)}: RMS={audio_rms:.4f}, peak={audio_peak:.4f}",
            flush=True,
        )

        if audio_peak < 0.003 or audio_rms < 0.001:
            raise RuntimeError(
                f"Silent audio for preset voice {speaker}, chunk {i + 1}/{len(chunks)}. "
                f"RMS={audio_rms:.6f}, peak={audio_peak:.6f}."
            )

        is_truncated, trailing_ratio = _check_trailing_silence(audio_f, sr)
        if is_truncated:
            raise RuntimeError(
                f"Audio truncation detected for preset voice {speaker}, chunk {i + 1}/{len(chunks)}. "
                f"Trailing silence: {trailing_ratio:.1%}. "
                f"This indicates premature EOS token generation."
            )

        all_audio.append(wavs[0])

    if len(all_audio) == 1:
        merged = all_audio[0]
    else:
        merged = all_audio[0]
        for audio in all_audio[1:]:
            merged = _crossfade_audio(merged, audio, sr)
        print(f"[TTS] Merged {len(all_audio)} chunks into single audio", flush=True)

    return [merged], sr


def _generate_saved_voice(
    model: Any, text: str, voice_id: str, params: dict[str, Any]
) -> tuple[Any, int]:
    """
    Generate audio using a saved voice clone.

    Args:
        model: Qwen3-TTS model instance.
        text: Text to synthesize.
        voice_id: Saved voice identifier.
        params: TTS parameters.

    Returns:
        Tuple of (wavs, sample_rate).

    Raises:
        FileNotFoundError: If saved voice not found.
    """
    voice_dir = SAVED_VOICES_DIR / voice_id
    prompt_path = voice_dir / "prompt.pkl"
    meta_path = voice_dir / "metadata.json"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Saved voice not found: {voice_id}")

    with open(meta_path) as f:
        meta = json.load(f)

    with open(prompt_path, "rb") as f:
        raw_prompt = pickle.load(f)

    voice_clone_prompt = _prepare_voice_clone_prompt(raw_prompt, model)
    print(
        f"[TTS] Prepared voice clone prompt for {voice_id} (dtype/device normalized)",
        flush=True,
    )

    lang = _normalize_language(params.get("language", "english"))
    print(f"[LANG] TTS normalized (voice clone): {lang}", flush=True)

    chunks = _split_text_into_chunks(text)

    if len(chunks) > 1:
        print(
            f"[TTS] Splitting text into {len(chunks)} chunks for voice {voice_id}",
            flush=True,
        )

    all_audio: list[np.ndarray] = []
    sr: int = 0

    for i, chunk in enumerate(chunks):
        preset_max = int(params.get("max_new_tokens", 1024))
        dynamic_max = _calculate_dynamic_max_tokens(chunk, preset_max)

        error_context = f"Voice: {voice_id}, Chunk {i + 1}/{len(chunks)}, Text length: {len(chunk)} chars"
        with timeout_handler(TTS_TIMEOUT_SECONDS, error_context):
            wavs, chunk_sr = model.generate_voice_clone(
                text=chunk,
                language=lang,
                voice_clone_prompt=voice_clone_prompt,
                temperature=params.get("temperature", 0.3),
                top_k=int(params.get("top_k", 50)),
                top_p=params.get("top_p", 0.85),
                repetition_penalty=params.get("repetition_penalty", 1.0),
                max_new_tokens=dynamic_max,
                subtalker_temperature=params.get("subtalker_temperature", 0.3),
                subtalker_top_k=int(params.get("subtalker_top_k", 50)),
                subtalker_top_p=params.get("subtalker_top_p", 0.85),
            )

        if sr == 0:
            sr = int(chunk_sr)

        audio_data = wavs[0]
        if audio_data.size == 0:
            raise RuntimeError(
                f"Empty audio returned for voice {voice_id}, chunk {i + 1}/{len(chunks)}"
            )

        audio_f = audio_data.astype(np.float32)
        if np.issubdtype(audio_data.dtype, np.integer):
            audio_f = audio_f / np.iinfo(audio_data.dtype).max
        audio_rms = float(np.sqrt(np.mean(audio_f * audio_f)))
        audio_peak = float(np.max(np.abs(audio_f)))

        print(
            f"[TTS] Voice clone chunk {i + 1}/{len(chunks)}: RMS={audio_rms:.4f}, peak={audio_peak:.4f}",
            flush=True,
        )

        if audio_peak < 0.003 or audio_rms < 0.001:
            raise RuntimeError(
                f"Silent audio detected for voice {voice_id}, chunk {i + 1}/{len(chunks)}. "
                f"RMS={audio_rms:.6f}, peak={audio_peak:.6f}."
            )

        is_truncated, trailing_ratio = _check_trailing_silence(audio_f, sr)
        if is_truncated:
            raise RuntimeError(
                f"Audio truncation detected for voice {voice_id}, chunk {i + 1}/{len(chunks)}. "
                f"Trailing silence: {trailing_ratio:.1%}. "
                f"This indicates premature EOS token generation."
            )

        all_audio.append(wavs[0])

    if len(all_audio) == 1:
        merged = all_audio[0]
    else:
        merged = all_audio[0]
        for audio in all_audio[1:]:
            merged = _crossfade_audio(merged, audio, sr)
        print(f"[TTS] Merged {len(all_audio)} chunks into single audio", flush=True)

    return [merged], sr


def generate_transcript_audio(
    transcript: Transcript,
    speaker_profile: SpeakerProfile,
    params: dict[str, Any],
    output_dir: str | Path,
) -> list[str]:
    """
    Generate audio for all dialogues in a transcript.

    Args:
        transcript: Transcript with dialogue list.
        speaker_profile: SpeakerProfile with voice mappings.
        params: TTS parameters.
        output_dir: Directory to save audio files.

    Returns:
        List of paths to generated audio files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = []
    for i, dialogue in enumerate(transcript.dialogues):
        # Sanitize speaker name for safe filename
        safe_speaker = (
            "".join(c for c in dialogue.speaker if c.isalnum() or c in "_-")[:30]
            or "unknown"
        )
        output_file = output_dir / f"dialogue_{i:03d}_{safe_speaker}.wav"
        try:
            path = generate_dialogue_audio(
                dialogue, speaker_profile, params, output_file
            )
            audio_paths.append(path)
        except Exception as e:
            print(f"Warning: Failed to generate audio for dialogue {i}: {e}")

    return audio_paths


if __name__ == "__main__":
    # Test with mock data
    from podcast_models import Dialogue, Speaker, SpeakerProfile, Transcript

    print("=== Audio Generator Test ===\n")

    # Create test speaker profile
    speakers = [
        Speaker(name="Alice", voice_id="male_1", role="Host", type="preset"),
        Speaker(name="Bob", voice_id="female_1", role="Guest", type="preset"),
    ]
    profile = SpeakerProfile(speakers=speakers)

    # Create test transcript
    dialogues = [
        Dialogue(speaker="Alice", text="Welcome to the podcast."),
        Dialogue(speaker="Bob", text="Thanks for having me."),
        Dialogue(speaker="Alice", text="Let's dive into the topic."),
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

    # Test 1: Single dialogue generation
    print("Test 1: Generate single dialogue audio")
    try:
        output_file = Path("test_output") / "test_dialogue.wav"
        path = generate_dialogue_audio(dialogues[0], profile, tts_params, output_file)
        print(f"✓ Generated: {path}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 2: Missing speaker error handling
    print("\nTest 2: Missing speaker error handling")
    try:
        bad_dialogue = Dialogue(speaker="Unknown", text="This should fail.")
        path = generate_dialogue_audio(bad_dialogue, profile, tts_params, "test.wav")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")

    # Test 3: Invalid voice type error handling
    print("\nTest 3: Invalid voice type error handling")
    try:
        bad_speaker = Speaker(
            name="Charlie", voice_id="v1", role="Guest", type="invalid"
        )
        bad_profile = SpeakerProfile(speakers=[bad_speaker])
        dialogue = Dialogue(speaker="Charlie", text="Test")
        path = generate_dialogue_audio(dialogue, bad_profile, tts_params, "test.wav")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")

    print("\n=== Tests completed ===")

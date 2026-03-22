"""
Batch + per-dialogue audio generation for podcast dialogue using Qwen3-TTS.

macOS-focused upgrades
- Model caching to avoid reload for every clip
- torch.inference_mode() to reduce overhead
- MPS cleanup after each chunk (helps long runs)
- Device-aware chunk sizing (smaller chunks on MPS)
- Optional hard-timeout mode (runs generation in a separate process so it can be killed)
- Retry + crash report JSON for batch mode
"""

from __future__ import annotations

import copy
import gc
import json
import os
import pickle
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any, Callable, Generator

import numpy as np
import soundfile as sf
import torch

from podcast.models import Dialogue, SpeakerProfile, Transcript

# --- macOS / MPS behavior knobs (safe defaults) ---
# Must be set before heavy model use. Setting after torch import may still help some runs.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# You can experiment with this if you see MPS memory pressure.
# os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

SAVED_VOICES_DIR = Path("saved_voices")

# Timeout for a single chunk generation (soft timeout by default)
TTS_TIMEOUT_SECONDS = 600

# Hard timeout kills the generation process if it stalls.
# This is the most reliable anti-hang option on macOS, but it is slower
# because the child process must load/access the model.
USE_HARD_TIMEOUT = False

# Retry configuration for batch generation
MAX_RETRIES = 3
RETRY_BACKOFF = (5, 10, 20)

MODEL_CACHE: dict[str, Any] = {}

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


# -------------------------
# Utilities
# -------------------------
def _normalize_language(lang: str) -> str:
    return LANGUAGE_MAP.get(lang, lang)


def _device_cleanup() -> None:
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def get_model_cached(model_name: str) -> Any:
    """
    Cache model instances by name so we do not reload them per dialogue.
    """
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    from audio.model_loader import get_model

    model = get_model(model_name)
    MODEL_CACHE[model_name] = model
    return model


def _get_model_dtype_device(model: Any) -> tuple[torch.dtype, torch.device]:
    """
    Get model dtype and device from Qwen3-TTS talker module when available.
    """
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
    if torch.cuda.is_available():
        return torch.bfloat16, torch.device("cuda")
    return torch.float32, torch.device("cpu")


def _prepare_voice_clone_prompt(voice_clone_prompt: Any, model: Any) -> Any:
    """
    Normalize voice-clone prompt dtype/device to match model.
    """
    model_dtype, model_device = _get_model_dtype_device(model)

    def convert_item(item: Any) -> Any:
        if hasattr(item, "ref_spk_embedding") and isinstance(item.ref_spk_embedding, torch.Tensor):
            if item.ref_spk_embedding.dtype != model_dtype or item.ref_spk_embedding.device != model_device:
                item.ref_spk_embedding = item.ref_spk_embedding.to(dtype=model_dtype, device=model_device)

        if hasattr(item, "ref_code") and isinstance(item.ref_code, torch.Tensor):
            if item.ref_code.device != model_device:
                item.ref_code = item.ref_code.to(device=model_device)

        return item

    if isinstance(voice_clone_prompt, list):
        return [convert_item(copy.copy(x)) for x in voice_clone_prompt]

    if isinstance(voice_clone_prompt, dict):
        result = voice_clone_prompt.copy()

        if "ref_spk_embedding" in result:
            emb = result["ref_spk_embedding"]
            if isinstance(emb, list):
                result["ref_spk_embedding"] = [
                    e.to(dtype=model_dtype, device=model_device) if isinstance(e, torch.Tensor) else e
                    for e in emb
                ]
            elif isinstance(emb, torch.Tensor):
                result["ref_spk_embedding"] = emb.to(dtype=model_dtype, device=model_device)

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

    return convert_item(copy.copy(voice_clone_prompt))


@contextmanager
def timeout_handler(seconds: int, error_context: str = "") -> Generator[None, None, None]:
    """
    Soft timeout guard that raises after control returns to Python.
    It cannot interrupt a blocking model call.
    """
    timeout_occurred = threading.Event()

    def trigger() -> None:
        timeout_occurred.set()

    timer = threading.Timer(seconds, trigger)
    timer.start()
    try:
        yield
        if timeout_occurred.is_set():
            raise TimeoutError(f"TTS generation timed out after {seconds}s. {error_context}")
    finally:
        timer.cancel()


def _chunk_config() -> tuple[int, int, int]:
    """
    Smaller chunks are safer on MPS to reduce stalls and truncation.
    """
    if torch.backends.mps.is_available():
        return 120, 150, 50
    return 300, 340, 100


def _split_text_into_chunks(text: str) -> list[str]:
    """
    Split long text into sentence-based chunks for TTS generation.
    """
    chunk_target, chunk_max, chunk_min = _chunk_config()

    text = text.strip()
    if len(text) <= chunk_max:
        return [text]

    import re

    sentences = re.split(r"(?<=[.!?。！？])\s*", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text]

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > chunk_max:
            if current:
                chunks.append(current.strip())
                current = ""

            words = sentence.split()
            temp = ""
            for word in words:
                if len(temp) + len(word) + 1 <= chunk_target:
                    temp = f"{temp} {word}".strip()
                else:
                    if temp:
                        chunks.append(temp)
                    temp = word
            if temp:
                chunks.append(temp)
            continue

        if len(current) + len(sentence) + 1 <= chunk_target:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    merged: list[str] = []
    i = 0
    while i < len(chunks):
        c = chunks[i]
        while i + 1 < len(chunks) and len(c) < chunk_min:
            i += 1
            c = f"{c} {chunks[i]}"
        merged.append(c.strip())
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
    Detect truncation pattern: speech then sustained trailing silence.
    """
    segment_samples = int(sr * segment_sec)
    min_trailing_segments = int(min_trailing_silent_sec / segment_sec)

    if len(audio) < segment_samples * (min_speech_segments + min_trailing_segments):
        return False, 0.0

    num_segments = len(audio) // segment_samples
    if num_segments <= 0:
        return False, 0.0

    segment_rms = np.array(
        [
            np.sqrt(np.mean(audio[i * segment_samples : (i + 1) * segment_samples] ** 2))
            for i in range(num_segments)
        ],
        dtype=np.float32,
    )

    if segment_rms.size == 0:
        return False, 0.0

    p95 = float(np.percentile(segment_rms, 95))
    abs_floor = 0.001
    adaptive_threshold = max(abs_floor, 0.02 * p95)

    is_silent = segment_rms < adaptive_threshold

    speech_found = np.sum(~is_silent[:min_speech_segments]) >= min_speech_segments
    if not speech_found:
        return False, 0.0

    trailing = 0
    for i in range(num_segments - 1, -1, -1):
        if is_silent[i]:
            trailing += 1
        else:
            break

    trailing_ratio = trailing / num_segments
    truncated = trailing >= min_trailing_segments
    return truncated, trailing_ratio


def _crossfade_audio(audio1: np.ndarray, audio2: np.ndarray, sr: int, fade_ms: int = 30) -> np.ndarray:
    fade_samples = int(sr * fade_ms / 1000)
    if len(audio1) < fade_samples or len(audio2) < fade_samples:
        return np.concatenate([audio1, audio2])

    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)

    a1_end = audio1[-fade_samples:] * fade_out
    a2_start = audio2[:fade_samples] * fade_in
    cross = a1_end + a2_start

    return np.concatenate([audio1[:-fade_samples], cross, audio2[fade_samples:]])


def _calculate_dynamic_max_tokens(text: str, preset_max: int) -> int:
    min_tokens = 256
    max_tokens = 768

    char_count = len(text)
    estimated = ceil(char_count * 2.5)
    dynamic_max = ceil(estimated * 1.3)

    final_max = max(min_tokens, min(dynamic_max, max_tokens, int(preset_max) if preset_max else max_tokens))
    print(f"[TTS] max_tokens: chars={char_count}, dynamic={dynamic_max}, final={final_max}", flush=True)
    return final_max


# -------------------------
# Optional hard-timeout execution (process-based)
# -------------------------
def _hard_timeout_generate_custom_voice(model_name: str, kwargs: dict[str, Any], timeout_seconds: int) -> tuple[Any, int]:
    """
    Runs generate_custom_voice in a child process so it can be terminated.
    Slower on macOS, but stops true hangs.
    """
    import multiprocessing as mp

    def _worker(q: mp.Queue) -> None:
        try:
            from audio.model_loader import get_model

            model = get_model(model_name)
            with torch.inference_mode():
                wavs, sr = model.generate_custom_voice(**kwargs)
            q.put(("ok", wavs, sr))
        except Exception as e:
            q.put(("err", f"{type(e).__name__}: {e}", traceback.format_exc()))

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_worker, args=(q,))
    p.start()
    p.join(timeout_seconds)

    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"TTS generation exceeded {timeout_seconds}s")

    if q.empty():
        raise RuntimeError("TTS process ended without returning audio")

    tag = q.get()
    if tag[0] == "ok":
        return tag[1], int(tag[2])

    raise RuntimeError(f"TTS worker failed: {tag[1]}\n{tag[2]}")


def _hard_timeout_generate_voice_clone(model_name: str, kwargs: dict[str, Any], timeout_seconds: int) -> tuple[Any, int]:
    import multiprocessing as mp

    def _worker(q: mp.Queue) -> None:
        try:
            from audio.model_loader import get_model

            model = get_model(model_name)
            with torch.inference_mode():
                wavs, sr = model.generate_voice_clone(**kwargs)
            q.put(("ok", wavs, sr))
        except Exception as e:
            q.put(("err", f"{type(e).__name__}: {e}", traceback.format_exc()))

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_worker, args=(q,))
    p.start()
    p.join(timeout_seconds)

    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"TTS generation exceeded {timeout_seconds}s")

    if q.empty():
        raise RuntimeError("TTS process ended without returning audio")

    tag = q.get()
    if tag[0] == "ok":
        return tag[1], int(tag[2])

    raise RuntimeError(f"TTS worker failed: {tag[1]}\n{tag[2]}")


# -------------------------
# Public API
# -------------------------
def generate_dialogue_audio(
    dialogue: Dialogue,
    speaker_profile: SpeakerProfile,
    params: dict[str, Any],
    output_path: str | Path,
) -> str:
    """
    Generate audio for a single dialogue line.

    Supports preset voices and saved voice clones.
    """
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

    if speaker.type not in ("preset", "saved"):
        raise ValueError(f"Invalid voice type: {speaker.type}. Must be 'preset' or 'saved'.")

    base_model_name = params.get("model_name", "1.7B-CustomVoice")

    if speaker.type == "saved":
        voice_meta_path = SAVED_VOICES_DIR / speaker.voice_id / "metadata.json"
        if voice_meta_path.exists():
            voice_meta = json.loads(voice_meta_path.read_text())
            model_name = voice_meta.get("model", "1.7B-Base")
        else:
            model_name = base_model_name.replace("CustomVoice", "Base")
    else:
        model_name = base_model_name

    model = get_model_cached(model_name)

    try:
        if speaker.type == "preset":
            wavs, sr = _generate_preset_voice(model_name, model, dialogue.text, speaker.voice_id, params)
        else:
            wavs, sr = _generate_saved_voice(model_name, model, dialogue.text, speaker.voice_id, params)
    except Exception as e:
        raise RuntimeError(f"TTS generation failed for speaker '{speaker.name}': {e}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        sf.write(str(output_path), wavs[0], sr)
    except Exception as e:
        raise RuntimeError(f"Failed to save audio to {output_path}: {e}")

    return str(output_path)


def _generate_preset_voice(
    model_name: str,
    model: Any,
    text: str,
    speaker: str,
    params: dict[str, Any],
) -> tuple[list[np.ndarray], int]:
    lang = _normalize_language(params.get("language", "english"))
    print(f"[LANG] TTS normalized: {lang}", flush=True)

    chunks = _split_text_into_chunks(text)
    if len(chunks) > 1:
        print(f"[TTS] Splitting text into {len(chunks)} chunks for speaker {speaker}", flush=True)

    all_audio: list[np.ndarray] = []
    sr = 0

    for i, chunk in enumerate(chunks):
        preset_max = int(params.get("max_new_tokens", 1024))
        dynamic_max = _calculate_dynamic_max_tokens(chunk, preset_max)

        error_context = f"Speaker: {speaker}, Chunk {i + 1}/{len(chunks)}, Text length: {len(chunk)} chars"

        kwargs = dict(
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

        with timeout_handler(TTS_TIMEOUT_SECONDS, error_context):
            if USE_HARD_TIMEOUT:
                wavs, chunk_sr = _hard_timeout_generate_custom_voice(model_name, kwargs, TTS_TIMEOUT_SECONDS)
            else:
                with torch.inference_mode():
                    wavs, chunk_sr = model.generate_custom_voice(**kwargs)

        if sr == 0:
            sr = int(chunk_sr)

        audio_data = wavs[0]
        if audio_data.size == 0:
            raise RuntimeError(f"Empty audio for preset voice {speaker}, chunk {i + 1}/{len(chunks)}")

        audio_f = audio_data.astype(np.float32)
        if np.issubdtype(audio_data.dtype, np.integer):
            audio_f = audio_f / np.iinfo(audio_data.dtype).max

        audio_rms = float(np.sqrt(np.mean(audio_f * audio_f)))
        audio_peak = float(np.max(np.abs(audio_f)))
        print(f"[TTS] Preset chunk {i + 1}/{len(chunks)}: RMS={audio_rms:.4f}, peak={audio_peak:.4f}", flush=True)

        if audio_peak < 0.003 or audio_rms < 0.001:
            raise RuntimeError(
                f"Silent audio for preset voice {speaker}, chunk {i + 1}/{len(chunks)}. "
                f"RMS={audio_rms:.6f}, peak={audio_peak:.6f}."
            )

        truncated, trailing_ratio = _check_trailing_silence(audio_f, sr)
        if truncated:
            raise RuntimeError(
                f"Audio truncation detected for preset voice {speaker}, chunk {i + 1}/{len(chunks)}. "
                f"Trailing silence: {trailing_ratio:.1%}."
            )

        all_audio.append(audio_data)
        _device_cleanup()

    if len(all_audio) == 1:
        merged = all_audio[0]
    else:
        merged = all_audio[0]
        for nxt in all_audio[1:]:
            merged = _crossfade_audio(merged, nxt, sr)
        print(f"[TTS] Merged {len(all_audio)} chunks into single audio", flush=True)

    return [merged], sr


def _generate_saved_voice(
    model_name: str,
    model: Any,
    text: str,
    voice_id: str,
    params: dict[str, Any],
) -> tuple[list[np.ndarray], int]:
    voice_dir = SAVED_VOICES_DIR / voice_id
    prompt_path = voice_dir / "prompt.pkl"
    meta_path = voice_dir / "metadata.json"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Saved voice not found: {voice_id}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Saved voice metadata not found: {voice_id}")

    meta = json.loads(meta_path.read_text())
    _ = meta  # reserved in case you want per-voice configs later

    raw_prompt = pickle.loads(prompt_path.read_bytes())
    voice_clone_prompt = _prepare_voice_clone_prompt(raw_prompt, model)

    print(f"[TTS] Prepared voice clone prompt for {voice_id} (dtype/device normalized)", flush=True)

    lang = _normalize_language(params.get("language", "english"))
    print(f"[LANG] TTS normalized (voice clone): {lang}", flush=True)

    chunks = _split_text_into_chunks(text)
    if len(chunks) > 1:
        print(f"[TTS] Splitting text into {len(chunks)} chunks for voice {voice_id}", flush=True)

    all_audio: list[np.ndarray] = []
    sr = 0

    for i, chunk in enumerate(chunks):
        preset_max = int(params.get("max_new_tokens", 1024))
        dynamic_max = _calculate_dynamic_max_tokens(chunk, preset_max)

        error_context = f"Voice: {voice_id}, Chunk {i + 1}/{len(chunks)}, Text length: {len(chunk)} chars"

        kwargs = dict(
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

        with timeout_handler(TTS_TIMEOUT_SECONDS, error_context):
            if USE_HARD_TIMEOUT:
                wavs, chunk_sr = _hard_timeout_generate_voice_clone(model_name, kwargs, TTS_TIMEOUT_SECONDS)
            else:
                with torch.inference_mode():
                    wavs, chunk_sr = model.generate_voice_clone(**kwargs)

        if sr == 0:
            sr = int(chunk_sr)

        audio_data = wavs[0]
        if audio_data.size == 0:
            raise RuntimeError(f"Empty audio returned for voice {voice_id}, chunk {i + 1}/{len(chunks)}")

        audio_f = audio_data.astype(np.float32)
        if np.issubdtype(audio_data.dtype, np.integer):
            audio_f = audio_f / np.iinfo(audio_data.dtype).max

        audio_rms = float(np.sqrt(np.mean(audio_f * audio_f)))
        audio_peak = float(np.max(np.abs(audio_f)))
        print(f"[TTS] Voice clone chunk {i + 1}/{len(chunks)}: RMS={audio_rms:.4f}, peak={audio_peak:.4f}", flush=True)

        if audio_peak < 0.003 or audio_rms < 0.001:
            raise RuntimeError(
                f"Silent audio detected for voice {voice_id}, chunk {i + 1}/{len(chunks)}. "
                f"RMS={audio_rms:.6f}, peak={audio_peak:.6f}."
            )

        truncated, trailing_ratio = _check_trailing_silence(audio_f, sr)
        if truncated:
            raise RuntimeError(
                f"Audio truncation detected for voice {voice_id}, chunk {i + 1}/{len(chunks)}. "
                f"Trailing silence: {trailing_ratio:.1%}."
            )

        all_audio.append(audio_data)
        _device_cleanup()

    if len(all_audio) == 1:
        merged = all_audio[0]
    else:
        merged = all_audio[0]
        for nxt in all_audio[1:]:
            merged = _crossfade_audio(merged, nxt, sr)
        print(f"[TTS] Merged {len(all_audio)} chunks into single audio", flush=True)

    return [merged], sr


def generate_transcript_audio(
    transcript: Transcript,
    speaker_profile: SpeakerProfile,
    params: dict[str, Any],
    output_dir: str | Path,
) -> list[str]:
    """
    Generate audio for all dialogues, writing one wav per dialogue.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_paths: list[str] = []
    for i, dialogue in enumerate(transcript.dialogues):
        safe_speaker = ("".join(c for c in dialogue.speaker if c.isalnum() or c in "_-")[:30] or "unknown")
        output_file = output_dir / f"dialogue_{i:03d}_{safe_speaker}.wav"
        try:
            path = generate_dialogue_audio(dialogue, speaker_profile, params, output_file)
            audio_paths.append(path)
        except Exception as e:
            print(f"Warning: Failed to generate audio for dialogue {i}: {e}")

    return audio_paths


def generate_all_clips(
    transcript: Transcript,
    speaker_profile: SpeakerProfile,
    params: dict[str, Any],
    clips_dir: str | Path,
    progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> list[str]:
    """
    Batch generator with retries, progress callback, crash logs, and summary.
    Saves clips as 0000.wav, 0001.wav, etc.
    """
    if not transcript.dialogues:
        raise ValueError("Transcript must contain at least one dialogue.")

    clips_dir = Path(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)

    total = len(transcript.dialogues)
    clip_paths: list[str] = []
    failed_indices: list[int] = []

    print("\n" + "=" * 60)
    print(f"Batch Audio Generation: {total} clips")
    print(f"Output directory: {clips_dir}")
    print("=" * 60 + "\n")

    for idx, dialogue in enumerate(transcript.dialogues):
        current = idx + 1
        filename = f"{idx:04d}.wav"
        output_path = clips_dir / filename

        segment_info: dict[str, Any] = {
            "index": idx,
            "speaker": dialogue.speaker,
            "text": dialogue.text[:50] + "..." if len(dialogue.text) > 50 else dialogue.text,
            "filename": filename,
            "status": None,
            "error": None,
            "path": None,
        }

        print(f"[{current:3d}/{total}] Generating: {dialogue.speaker}")
        print(f"         Text: {segment_info['text']}")

        segment_info["status"] = "started"
        if progress_callback is not None:
            progress_callback(current, total, segment_info)

        last_error: Exception | None = None

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
                    delay = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                    print(f"         ⚠ Clip {idx} failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}")
                    print(f"         Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    error_msg = f"Failed after {MAX_RETRIES + 1} attempts: {last_error}"
                    segment_info["status"] = "error"
                    segment_info["error"] = error_msg
                    failed_indices.append(idx)

                    error_report = {
                        "clip_index": idx,
                        "speaker": dialogue.speaker,
                        "text_length": len(dialogue.text),
                        "error": str(last_error),
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.now().isoformat(),
                        "retry_attempts": MAX_RETRIES + 1,
                        "device": "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
                    }

                    crash_log = clips_dir / f"crash_clip_{idx:04d}.json"
                    crash_log.write_text(json.dumps(error_report, indent=2))
                    print(f"         ✗ Crash report saved to: {crash_log}")
                    print(f"         ✗ Skipping clip {idx}\n")
                    break

            except Exception as e:
                segment_info["status"] = "error"
                segment_info["error"] = str(e)
                failed_indices.append(idx)

                print(f"         ✗ Error: {e}\n")
                break

        _device_cleanup()

        if progress_callback is not None:
            progress_callback(current, total, segment_info)

    print("=" * 60)
    print("Batch Generation Complete")
    print(f"  Total clips: {total}")
    print(f"  Successful: {len(clip_paths)}")
    print(f"  Failed: {len(failed_indices)}")
    if failed_indices:
        print(f"  Failed indices: {failed_indices}")
    print("=" * 60 + "\n")

    if not clip_paths:
        raise RuntimeError(
            f"All {total} audio clips failed to generate. Check model config and speaker profiles."
        )

    return clip_paths


# -------------------------
# Simple test harness
# -------------------------
if __name__ == "__main__":
    from podcast_models import Dialogue as TestDialogue
    from podcast_models import Speaker as TestSpeaker
    from podcast_models import SpeakerProfile as TestSpeakerProfile
    from podcast_models import Transcript as TestTranscript

    print("=== Qwen3-TTS Batch + Generator Test ===\n")

    speakers = [
        TestSpeaker(name="Alice", voice_id="male_1", role="Host", type="preset"),
        TestSpeaker(name="Bob", voice_id="female_1", role="Guest", type="preset"),
    ]
    profile = TestSpeakerProfile(speakers=speakers)

    dialogues = [
        TestDialogue(speaker="Alice", text="Welcome to the podcast."),
        TestDialogue(speaker="Bob", text="Thanks for having me."),
        TestDialogue(speaker="Alice", text="Let's dive into the topic."),
        TestDialogue(speaker="Bob", text="I am excited to discuss this."),
        TestDialogue(speaker="Alice", text="Great. Let us begin."),
    ]
    transcript = TestTranscript(dialogues=dialogues)

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

    def progress_callback(current: int, total: int, segment_info: dict[str, Any]) -> None:
        icon = "✓" if segment_info["status"] == "success" else ("…" if segment_info["status"] == "started" else "✗")
        print(f"  [{icon}] {current}/{total} - {segment_info['speaker']}")
        if segment_info.get("error"):
            print(f"      Error: {segment_info['error']}")

    # Single dialogue test
    print("Test 1 - Single dialogue")
    try:
        out = Path("test_output") / "single.wav"
        p = generate_dialogue_audio(dialogues[0], profile, tts_params, out)
        print(f"✓ Generated: {p}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")

    # Batch test
    print("Test 2 - Batch generation")
    try:
        out_dir = Path("test_batch_output")
        paths = generate_all_clips(transcript, profile, tts_params, out_dir, progress_callback)
        print(f"✓ Generated {len(paths)} clips\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")

    print("=== Tests completed ===")
"""Utilities for combining multiple speaker embeddings for improved voice cloning."""

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class AudioSampleInfo:
    """Information about a single audio sample for voice cloning."""

    path: str
    duration: float  # seconds
    transcript: Optional[str] = None
    snr_estimate: Optional[float] = None  # dB, higher is better
    is_primary: bool = False  # Use this sample's ref_code for ICL mode
    weight: float = 1.0  # Computed weight for embedding combination


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds."""
    try:
        data, sr = sf.read(audio_path)
        return len(data) / sr
    except Exception:
        return 0.0


def estimate_snr(audio_path: str) -> float:
    """
    Estimate Signal-to-Noise Ratio of an audio file.

    Uses a simple heuristic based on:
    - RMS energy of the full signal vs silent portions
    - Assumes first/last 5% might contain silence

    Returns:
        Estimated SNR in dB (higher is better), or 0.0 on error
    """
    try:
        data, sr = sf.read(audio_path)
        if len(data) == 0 or sr <= 0:
            return 0.0

        audio = data.astype(np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        rms_total = np.sqrt(np.mean(audio**2))
        if rms_total < 1e-10:
            return 0.0

        frame_size = max(1, int(sr * 0.02))
        n_frames = len(audio) // frame_size
        if n_frames < 10:
            return 20.0

        frame_rms = []
        for i in range(n_frames):
            frame = audio[i * frame_size : (i + 1) * frame_size]
            frame_rms.append(np.sqrt(np.mean(frame**2)))

        frame_rms = np.array(frame_rms)
        noise_floor = np.percentile(frame_rms, 10)

        if noise_floor < 1e-10:
            return 40.0

        snr = 20 * np.log10(rms_total / noise_floor)
        return max(0.0, min(60.0, snr))

    except Exception:
        return 20.0


def analyze_audio_samples(
    audio_paths: list[str],
    transcripts: Optional[list[Optional[str]]] = None,
) -> list[AudioSampleInfo]:
    """
    Analyze multiple audio samples and compute quality metrics.

    Args:
        audio_paths: List of paths to audio files
        transcripts: Optional list of transcripts (same length as audio_paths)

    Returns:
        List of AudioSampleInfo with computed metrics and weights
    """
    transcript_list: list[Optional[str]] = (
        [None] * len(audio_paths) if transcripts is None else list(transcripts)
    )
    if len(transcript_list) < len(audio_paths):
        transcript_list.extend([None] * (len(audio_paths) - len(transcript_list)))

    samples = []
    for i, (path, transcript) in enumerate(zip(audio_paths, transcript_list)):
        duration = get_audio_duration(path)
        snr = estimate_snr(path)

        samples.append(
            AudioSampleInfo(
                path=path,
                duration=duration,
                transcript=transcript,
                snr_estimate=snr,
                is_primary=(i == 0),  # First sample is primary by default
            )
        )

    # Compute weights based on duration and quality
    total_duration = sum(s.duration for s in samples)
    if total_duration > 0:
        for s in samples:
            # Duration weight (normalized)
            duration_weight = s.duration / total_duration

            # Quality weight (SNR normalized to 0-1 range, assuming 0-40dB)
            snr = s.snr_estimate or 20.0
            quality_weight = min(1.0, snr / 40.0)

            # Combined weight: 70% duration, 30% quality
            s.weight = 0.7 * duration_weight + 0.3 * quality_weight

    # Normalize weights to sum to 1
    total_weight = sum(s.weight for s in samples)
    if total_weight > 0:
        for s in samples:
            s.weight = s.weight / total_weight

    # Select best sample as primary
    # Prefer samples with transcripts (needed for ICL mode), fall back to best quality
    if samples:
        samples_with_transcript = [
            i for i, s in enumerate(samples) if s.transcript and s.transcript.strip()
        ]
        candidates = (
            samples_with_transcript if samples_with_transcript else range(len(samples))
        )
        best_idx = max(
            candidates,
            key=lambda i: samples[i].duration * (samples[i].snr_estimate or 20.0),
        )
        for i, s in enumerate(samples):
            s.is_primary = i == best_idx

    return samples


def combine_speaker_embeddings(
    embeddings: list[torch.Tensor],
    weights: Optional[list[float]] = None,
    outlier_threshold: float = 0.7,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Combine multiple speaker embeddings into one using weighted averaging.

    This is the core function for multi-sample voice cloning.
    Uses L2-normalized weighted mean with optional outlier rejection.

    Args:
        embeddings: List of speaker embedding tensors, each of shape (D,)
        weights: Optional weights for each embedding (will be normalized)
        outlier_threshold: Cosine similarity threshold for outlier rejection.
                          Embeddings with similarity to centroid below this
                          will be excluded. Set to 0 to disable.
        device: Target device for output tensor

    Returns:
        Combined speaker embedding tensor of shape (D,)
    """
    if not embeddings:
        raise ValueError("At least one embedding is required")

    if len(embeddings) == 1:
        # Single embedding: normalize and return
        return F.normalize(embeddings[0].to(device).float(), dim=-1)

    # Determine device
    if device is None:
        device = embeddings[0].device

    # Move all embeddings to same device and normalize
    normed = []
    for e in embeddings:
        e_device = e.to(device).float()
        normed.append(F.normalize(e_device, dim=-1))

    # Default to equal weights
    if weights is None:
        weights = [1.0] * len(normed)

    # Compute centroid for outlier detection
    stacked = torch.stack(normed)
    centroid = stacked.mean(dim=0)
    centroid = F.normalize(centroid, dim=-1)

    # Filter outliers based on cosine similarity to centroid
    valid_embeddings = []
    valid_weights = []

    for i, (emb, w) in enumerate(zip(normed, weights)):
        if outlier_threshold > 0:
            sim = (emb.flatten() * centroid.flatten()).sum().item()
            if sim < outlier_threshold:
                print(
                    f"[Embedding] Sample {i} excluded: cosine similarity {sim:.3f} < {outlier_threshold}"
                )
                continue
        valid_embeddings.append(emb)
        valid_weights.append(w)

    # If all samples were filtered, fall back to using all
    if not valid_embeddings:
        print("[Embedding] Warning: All samples below threshold, using all")
        valid_embeddings = normed
        valid_weights = weights

    # Normalize weights
    total_w = sum(valid_weights)
    valid_weights = [w / total_w for w in valid_weights]

    # Compute weighted mean
    combined = torch.zeros_like(valid_embeddings[0])
    for emb, w in zip(valid_embeddings, valid_weights):
        combined = combined + w * emb

    # Final L2 normalization
    combined = F.normalize(combined, dim=-1)

    return combined


def create_combined_voice_clone_prompt(
    model: Any,
    sample_infos: list[AudioSampleInfo],
    x_vector_only_mode: bool = False,
) -> list[Any]:
    """
    Create a voice clone prompt from multiple audio samples.

    This function:
    1. Creates individual prompts for each sample
    2. Combines speaker embeddings using weighted averaging
    3. Uses the primary sample's ref_code for ICL mode

    Args:
        model: Qwen3-TTS model instance
        sample_infos: List of AudioSampleInfo with paths and transcripts
        x_vector_only_mode: If True, only use speaker embedding (no ref_code)

    Returns:
        List containing single VoiceClonePromptItem with combined embedding
    """
    if not sample_infos:
        raise ValueError("At least one audio sample is required")

    # Create individual prompts for each sample
    all_prompts = []
    for info in sample_infos:
        prompt = model.create_voice_clone_prompt(
            ref_audio=info.path,
            ref_text=info.transcript,
            x_vector_only_mode=x_vector_only_mode,
        )
        all_prompts.append((info, prompt[0]))  # prompt is a list, take first item

    # Extract embeddings and weights
    embeddings = [p[1].ref_spk_embedding for p in all_prompts]
    weights = [p[0].weight for p in all_prompts]

    # Combine embeddings
    combined_embedding = combine_speaker_embeddings(
        embeddings=embeddings,
        weights=weights,
        outlier_threshold=0.7,
    )

    # Find primary sample for ref_code (only used in ICL mode)
    # ICL mode needs ref_text, so prefer samples with transcripts
    primary_prompt = None
    if not x_vector_only_mode:
        for info, prompt in all_prompts:
            if info.is_primary and info.transcript and info.transcript.strip():
                primary_prompt = prompt
                break
        if primary_prompt is None:
            for info, prompt in all_prompts:
                if info.transcript and info.transcript.strip():
                    primary_prompt = prompt
                    break
    if primary_prompt is None:
        for info, prompt in all_prompts:
            if info.is_primary:
                primary_prompt = prompt
                break
    if primary_prompt is None:
        primary_prompt = all_prompts[0][1]

    # Create combined prompt item
    # Use the same type as the original prompt items
    prompt_type = type(primary_prompt)
    combined_prompt = prompt_type(
        ref_code=None if x_vector_only_mode else primary_prompt.ref_code,
        ref_spk_embedding=combined_embedding,
        x_vector_only_mode=x_vector_only_mode,
        icl_mode=not x_vector_only_mode,
        ref_text=primary_prompt.ref_text if not x_vector_only_mode else None,
    )

    return [combined_prompt]


def format_samples_summary(samples: list[AudioSampleInfo]) -> str:
    """Format a summary of audio samples for display."""
    if not samples:
        return "No samples"

    total_duration = sum(s.duration for s in samples)
    lines = []

    for i, s in enumerate(samples):
        name = Path(s.path).name if s.path else f"Sample {i + 1}"
        primary_marker = " ★" if s.is_primary else ""
        snr_text = f"SNR: {s.snr_estimate:.0f}dB" if s.snr_estimate else ""
        lines.append(f"• {name}: {s.duration:.1f}s{primary_marker} {snr_text}")

    lines.append(f"\nTotal: {total_duration:.1f}s ({len(samples)} samples)")

    # Add recommendations
    if total_duration < 10:
        lines.append("⚠️ Consider adding more samples for better quality")
    elif total_duration > 60:
        lines.append("⚠️ Total duration is long, consider removing some samples")

    return "\n".join(lines)


def get_sample_warnings(samples: list[AudioSampleInfo]) -> list[str]:
    """Get warnings about sample quality issues."""
    warnings = []

    for i, s in enumerate(samples):
        name = Path(s.path).name if s.path else f"Sample {i + 1}"

        if s.duration < 3:
            warnings.append(
                f"⚠️ {name}: Too short ({s.duration:.1f}s). Min 3s recommended."
            )

        if s.snr_estimate and s.snr_estimate < 15:
            warnings.append(
                f"⚠️ {name}: Low audio quality detected (SNR: {s.snr_estimate:.0f}dB)"
            )

    # Check for potential speaker mismatch
    if len(samples) >= 2:
        # This would require embedding comparison, simplified check here
        durations = [s.duration for s in samples]
        if max(durations) > 5 * min(durations):
            warnings.append(
                "⚠️ Large duration variance between samples. Ensure all samples are from the same speaker."
            )

    return warnings

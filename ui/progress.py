#!/usr/bin/env python3
"""
Progress UI Components for Multi-Step Voice Generation

Provides visual feedback for podcast generation pipeline:
Outline -> Transcript -> Audio -> Combine

Can run standalone for testing with mock progress updates.
"""

import gradio as gr
import time
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum


class GenerationStep(Enum):
    """Steps in the podcast generation pipeline."""
    OUTLINE = "outline"
    TRANSCRIPT = "transcript"
    AUDIO = "audio"
    COMBINE = "combine"


@dataclass
class ProgressState:
    """Current state of generation progress."""
    current_step: GenerationStep
    step_progress: float  # 0.0 - 1.0 within current step
    overall_progress: float  # 0.0 - 1.0 total
    status_text: str
    segment_current: int = 0
    segment_total: int = 0
    clip_current: int = 0
    clip_total: int = 0
    segment_title: str = ""
    elapsed_seconds: float = 0.0
    estimated_remaining: Optional[float] = None
    
    
# Step weights for overall progress calculation
STEP_WEIGHTS = {
    GenerationStep.OUTLINE: 0.05,
    GenerationStep.TRANSCRIPT: 0.15,
    GenerationStep.AUDIO: 0.70,
    GenerationStep.COMBINE: 0.10,
}


def get_step_order() -> list[GenerationStep]:
    """Return steps in execution order."""
    return [
        GenerationStep.OUTLINE,
        GenerationStep.TRANSCRIPT,
        GenerationStep.AUDIO,
        GenerationStep.COMBINE,
    ]


def calculate_overall_progress(step: GenerationStep, step_progress: float) -> float:
    """Calculate overall progress percentage based on step and progress within step."""
    steps = get_step_order()
    completed_weight = sum(
        STEP_WEIGHTS[s] for s in steps[:steps.index(step)]
    )
    current_weight = STEP_WEIGHTS[step] * step_progress
    return round((completed_weight + current_weight) * 100, 1)


def format_time_remaining(seconds: Optional[float]) -> str:
    """Format estimated time remaining as human-readable string."""
    if seconds is None or seconds <= 0:
        return ""
    
    if seconds < 60:
        return f"~{int(seconds)}s remaining"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"~{mins}m {secs}s remaining"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"~{hours}h {mins}m remaining"


def create_step_indicator_html(
    current_step: GenerationStep,
    step_progress: float = 0.0
) -> str:
    """
    Generate HTML for step indicator with checkmarks.
    
    Shows: Outline -> Transcript -> Audio -> Combine
    With checkmarks for completed, spinner for current, dimmed for pending.
    """
    steps = get_step_order()
    current_idx = steps.index(current_step)
    
    step_labels = {
        GenerationStep.OUTLINE: "Outline",
        GenerationStep.TRANSCRIPT: "Transcript",
        GenerationStep.AUDIO: "Audio",
        GenerationStep.COMBINE: "Combine",
    }
    
    html_parts = ['<div class="step-indicator">']
    
    for i, step in enumerate(steps):
        label = step_labels[step]
        
        if i < current_idx:
            # Completed step
            icon = '<span class="step-icon completed">&#10003;</span>'
            state_class = "completed"
        elif i == current_idx:
            # Current step
            progress_pct = int(step_progress * 100)
            icon = f'<span class="step-icon current">{progress_pct}%</span>'
            state_class = "current"
        else:
            # Pending step
            icon = f'<span class="step-icon pending">{i + 1}</span>'
            state_class = "pending"
        
        html_parts.append(f'''
            <div class="step-item {state_class}">
                {icon}
                <span class="step-label">{label}</span>
            </div>
        ''')
        
        # Add connector between steps (except after last)
        if i < len(steps) - 1:
            connector_class = "completed" if i < current_idx else "pending"
            html_parts.append(f'<div class="step-connector {connector_class}"></div>')
    
    html_parts.append('</div>')
    return ''.join(html_parts)


def create_status_text(state: ProgressState) -> str:
    """Generate detailed status text based on current state."""
    step_descriptions = {
        GenerationStep.OUTLINE: "Creating podcast outline...",
        GenerationStep.TRANSCRIPT: "Generating transcript...",
        GenerationStep.AUDIO: "Generating audio...",
        GenerationStep.COMBINE: "Combining audio segments...",
    }
    
    base_text = step_descriptions.get(state.current_step, "Processing...")
    
    # Add detail for audio generation
    if state.current_step == GenerationStep.AUDIO and state.segment_total > 0:
        segment_info = f"segment {state.segment_current}/{state.segment_total}"
        if state.clip_total > 0:
            clip_info = f"clip {state.clip_current}/{state.clip_total}"
            title_part = f": '{state.segment_title}'" if state.segment_title else ""
            base_text = f"Generating audio for {segment_info} ({clip_info}){title_part}"
        else:
            base_text = f"Generating audio for {segment_info}"
    
    return base_text


# CSS for the progress components - matches qwen_tts_ui.py monotone aesthetic
PROGRESS_CSS = """
/* ===== PROGRESS INDICATOR STYLES ===== */
:root {
    --gray-50: #f8f9fa;
    --gray-100: #f1f3f5;
    --gray-200: #e9ecef;
    --gray-300: #dee2e6;
    --gray-400: #ced4da;
    --gray-500: #adb5bd;
    --gray-600: #868e96;
    --gray-700: #495057;
    --gray-800: #343a40;
    --gray-900: #212529;
    --white: #ffffff;
    --radius: 4px;
}

/* Step indicator container */
.step-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1.5rem 1rem;
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius);
    margin-bottom: 1rem;
    gap: 0;
}

/* Individual step item */
.step-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    min-width: 80px;
}

/* Step icon (circle with number/check) */
.step-icon {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.875rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.step-icon.completed {
    background: var(--gray-800);
    color: var(--white);
    font-size: 1.1rem;
}

.step-icon.current {
    background: var(--gray-700);
    color: var(--white);
    font-size: 0.75rem;
    animation: pulse 2s ease-in-out infinite;
}

.step-icon.pending {
    background: var(--gray-200);
    color: var(--gray-500);
}

/* Step label */
.step-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--gray-600);
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.step-item.completed .step-label {
    color: var(--gray-800);
}

.step-item.current .step-label {
    color: var(--gray-800);
    font-weight: 600;
}

.step-item.pending .step-label {
    color: var(--gray-400);
}

/* Connector line between steps */
.step-connector {
    width: 60px;
    height: 2px;
    background: var(--gray-200);
    margin: 0 0.5rem;
    margin-bottom: 1.5rem;
    transition: background 0.3s ease;
}

.step-connector.completed {
    background: var(--gray-800);
}

/* Pulse animation for current step */
@keyframes pulse {
    0%, 100% {
        box-shadow: 0 0 0 0 rgba(52, 58, 64, 0.4);
    }
    50% {
        box-shadow: 0 0 0 8px rgba(52, 58, 64, 0);
    }
}

/* Progress panel container */
.progress-panel {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius);
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.progress-panel-header {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--gray-700);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
}

/* Status text styling */
.status-text {
    font-size: 0.9rem;
    color: var(--gray-700);
    padding: 0.75rem;
    background: var(--gray-50);
    border-radius: var(--radius);
    border-left: 3px solid var(--gray-600);
    margin-bottom: 1rem;
}

/* Time remaining */
.time-remaining {
    font-size: 0.8rem;
    color: var(--gray-500);
    text-align: right;
    margin-top: 0.5rem;
}

/* Overall progress section */
.overall-progress {
    margin-top: 1rem;
}

.progress-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.progress-label-text {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--gray-700);
}

.progress-label-percent {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--gray-800);
    font-variant-numeric: tabular-nums;
}

/* Responsive adjustments */
@media (max-width: 600px) {
    .step-indicator {
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    
    .step-connector {
        display: none;
    }
    
    .step-item {
        min-width: 60px;
    }
    
    .step-icon {
        width: 32px;
        height: 32px;
        font-size: 0.75rem;
    }
}
"""


def create_progress_components():
    """
    Create Gradio components for progress visualization.
    
    Returns tuple of (step_indicator, progress_bar, status_text, time_remaining)
    """
    step_indicator = gr.HTML(
        value=create_step_indicator_html(GenerationStep.OUTLINE, 0.0),
        elem_classes=["step-indicator-container"]
    )
    
    overall_progress = gr.Slider(
        minimum=0,
        maximum=100,
        value=0,
        label="Overall Progress",
        interactive=False,
        elem_classes=["progress-slider"]
    )
    
    status_text = gr.Textbox(
        value="Ready to generate...",
        label="Status",
        interactive=False,
        elem_classes=["status-text-box"]
    )
    
    time_remaining = gr.Textbox(
        value="",
        label="",
        interactive=False,
        show_label=False,
        elem_classes=["time-remaining-box"]
    )
    
    return step_indicator, overall_progress, status_text, time_remaining


def update_progress_display(state: ProgressState):
    """
    Update all progress components based on current state.
    
    Returns tuple of values for (step_indicator, progress_bar, status_text, time_remaining)
    """
    step_html = create_step_indicator_html(state.current_step, state.step_progress)
    overall_pct = calculate_overall_progress(state.current_step, state.step_progress)
    status = create_status_text(state)
    time_str = format_time_remaining(state.estimated_remaining)
    
    return step_html, overall_pct, status, time_str


class ProgressTracker:
    """
    Helper class to track and calculate progress during generation.
    
    Tracks timing to estimate remaining time based on average clip generation speed.
    """
    
    def __init__(self):
        self.start_time: float = 0
        self.clip_times: list[float] = []
        self.current_state: ProgressState = ProgressState(
            current_step=GenerationStep.OUTLINE,
            step_progress=0.0,
            overall_progress=0.0,
            status_text="Starting...",
        )
    
    def start(self):
        """Mark start of generation."""
        self.start_time = time.time()
        self.clip_times = []
    
    def record_clip_time(self, duration: float):
        """Record time taken for a single clip generation."""
        self.clip_times.append(duration)
    
    def get_average_clip_time(self) -> float:
        """Get average time per clip based on recorded times."""
        if not self.clip_times:
            return 0.0
        return sum(self.clip_times) / len(self.clip_times)
    
    def estimate_remaining(self, clips_remaining: int) -> Optional[float]:
        """Estimate time remaining based on clips left and average time."""
        avg_time = self.get_average_clip_time()
        if avg_time <= 0 or clips_remaining <= 0:
            return None
        return avg_time * clips_remaining
    
    def update(
        self,
        step: GenerationStep,
        step_progress: float,
        segment_current: int = 0,
        segment_total: int = 0,
        clip_current: int = 0,
        clip_total: int = 0,
        segment_title: str = "",
    ) -> ProgressState:
        """Update progress state and return new state."""
        elapsed = time.time() - self.start_time if self.start_time > 0 else 0.0
        
        # Estimate remaining time for audio generation
        estimated_remaining = None
        if step == GenerationStep.AUDIO and clip_total > 0:
            clips_remaining = clip_total - clip_current
            estimated_remaining = self.estimate_remaining(clips_remaining)
        
        self.current_state = ProgressState(
            current_step=step,
            step_progress=step_progress,
            overall_progress=calculate_overall_progress(step, step_progress),
            status_text=create_status_text(self.current_state),
            segment_current=segment_current,
            segment_total=segment_total,
            clip_current=clip_current,
            clip_total=clip_total,
            segment_title=segment_title,
            elapsed_seconds=elapsed,
            estimated_remaining=estimated_remaining,
        )
        
        return self.current_state


# ============================================================================
# STANDALONE TEST / DEMO APP
# ============================================================================

def create_demo_app():
    """Create a standalone demo app for testing progress components."""
    
    with gr.Blocks(title="Progress UI Demo") as demo:
        gr.HTML(f"""
        <style>{PROGRESS_CSS}</style>
        <div style="text-align: center; padding: 1rem; border-bottom: 1px solid #e9ecef; margin-bottom: 1rem;">
            <h1 style="font-size: 1.5rem; font-weight: 600; color: #343a40; margin: 0;">
                Progress UI Components
            </h1>
            <p style="color: #868e96; font-size: 0.875rem; margin: 0.25rem 0 0;">
                Visual feedback for podcast generation pipeline
            </p>
        </div>
        """)
        
        with gr.Column():
            gr.HTML('<div class="progress-panel-header">Generation Progress</div>')
            
            step_indicator, overall_progress, status_text, time_remaining = create_progress_components()
            
            gr.HTML("<hr style='margin: 1.5rem 0; border: 0; border-top: 1px solid #e9ecef;'>")
            gr.HTML('<div class="progress-panel-header">Demo Controls</div>')
            
            with gr.Row():
                start_demo_btn = gr.Button(
                    "Start Mock Generation",
                    variant="primary",
                    size="lg"
                )
                reset_btn = gr.Button(
                    "Reset",
                    variant="secondary",
                    size="lg"
                )
            
            demo_log = gr.Textbox(
                label="Demo Log",
                lines=6,
                interactive=False,
                placeholder="Click 'Start Mock Generation' to see progress animation..."
            )
        
        def reset_progress():
            """Reset all progress indicators to initial state."""
            return (
                create_step_indicator_html(GenerationStep.OUTLINE, 0.0),
                0,
                "Ready to generate...",
                "",
                ""
            )
        
        def run_mock_generation(progress=gr.Progress()):
            """Simulate a full generation cycle with mock progress updates."""
            log_lines = []
            tracker = ProgressTracker()
            tracker.start()
            
            segments: list[dict[str, str | int]] = [
                {"title": "Introduction", "clips": 3},
                {"title": "The Main Challenge", "clips": 5},
                {"title": "Solutions & Insights", "clips": 4},
                {"title": "Conclusion", "clips": 3},
            ]
            total_clips = sum(int(s["clips"]) for s in segments)
            
            # Step 1: Outline
            log_lines.append("Step 1: Creating outline...")
            progress(0.02, desc="Creating outline...")
            yield (
                create_step_indicator_html(GenerationStep.OUTLINE, 0.3),
                calculate_overall_progress(GenerationStep.OUTLINE, 0.3),
                "Creating podcast outline...",
                "",
                "\n".join(log_lines)
            )
            time.sleep(0.8)
            
            log_lines.append("  - Outline complete!")
            yield (
                create_step_indicator_html(GenerationStep.OUTLINE, 1.0),
                calculate_overall_progress(GenerationStep.OUTLINE, 1.0),
                "Outline created successfully",
                "",
                "\n".join(log_lines)
            )
            time.sleep(0.3)
            
            # Step 2: Transcript
            log_lines.append("Step 2: Generating transcript...")
            progress(0.1, desc="Generating transcript...")
            for i in range(4):
                segment_progress = (i + 1) / 4
                yield (
                    create_step_indicator_html(GenerationStep.TRANSCRIPT, segment_progress),
                    calculate_overall_progress(GenerationStep.TRANSCRIPT, segment_progress),
                    f"Generating transcript... segment {i+1}/4",
                    "",
                    "\n".join(log_lines)
                )
                time.sleep(0.5)
            
            log_lines.append("  - Transcript complete!")
            time.sleep(0.3)
            
            # Step 3: Audio generation
            log_lines.append("Step 3: Generating audio...")
            progress(0.2, desc="Generating audio...")
            
            clip_idx = 0
            for seg_idx, segment in enumerate(segments):
                num_clips = int(segment["clips"])
                title = str(segment["title"])
                for clip_num in range(num_clips):
                    clip_idx += 1
                    clip_start = time.time()
                    
                    step_progress = clip_idx / total_clips
                    state = ProgressState(
                        current_step=GenerationStep.AUDIO,
                        step_progress=step_progress,
                        overall_progress=calculate_overall_progress(GenerationStep.AUDIO, step_progress),
                        status_text="",
                        segment_current=seg_idx + 1,
                        segment_total=len(segments),
                        clip_current=clip_idx,
                        clip_total=total_clips,
                        segment_title=title,
                        estimated_remaining=tracker.estimate_remaining(total_clips - clip_idx),
                    )
                    
                    progress(0.2 + 0.7 * step_progress, desc=create_status_text(state))
                    yield (
                        create_step_indicator_html(GenerationStep.AUDIO, step_progress),
                        calculate_overall_progress(GenerationStep.AUDIO, step_progress),
                        create_status_text(state),
                        format_time_remaining(state.estimated_remaining),
                        "\n".join(log_lines + [f"  - Clip {clip_idx}/{total_clips}: {title}"])
                    )
                    
                    # Simulate variable generation time
                    time.sleep(0.3 + (clip_num % 3) * 0.1)
                    tracker.record_clip_time(time.time() - clip_start)
            
            log_lines.append("  - Audio generation complete!")
            time.sleep(0.3)
            
            # Step 4: Combine
            log_lines.append("Step 4: Combining audio segments...")
            progress(0.92, desc="Combining segments...")
            for i in range(3):
                step_progress = (i + 1) / 3
                yield (
                    create_step_indicator_html(GenerationStep.COMBINE, step_progress),
                    calculate_overall_progress(GenerationStep.COMBINE, step_progress),
                    f"Combining audio segments... ({i+1}/3)",
                    "~2s remaining" if i < 2 else "",
                    "\n".join(log_lines)
                )
                time.sleep(0.5)
            
            log_lines.append("  - Complete!")
            log_lines.append("")
            log_lines.append("Generation finished successfully!")
            
            progress(1.0, desc="Complete!")
            yield (
                create_step_indicator_html(GenerationStep.COMBINE, 1.0),
                100,
                "Generation complete!",
                "",
                "\n".join(log_lines)
            )
        
        start_demo_btn.click(
            fn=run_mock_generation,
            outputs=[step_indicator, overall_progress, status_text, time_remaining, demo_log],
        )
        
        reset_btn.click(
            fn=reset_progress,
            outputs=[step_indicator, overall_progress, status_text, time_remaining, demo_log],
        )
    
    return demo


if __name__ == "__main__":
    print("Starting Progress UI Demo...")
    print("Open http://localhost:7861 in your browser")
    demo = create_demo_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
    )

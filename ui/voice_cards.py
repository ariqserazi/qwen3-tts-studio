#!/usr/bin/env python3
"""
Voice Cards UI Component

Gradio-based visual voice selection with cards for podcast generation.
Each card displays: Voice name, Type badge, Role dropdown, Preview button.
Supports 2-4 voice selection with visual feedback.
"""

import json
import os
import tempfile
from typing import Any

import gradio as gr

os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

ROLES = ["Host", "Expert", "Guest", "Narrator"]
ROLE_COLORS = {
    "Host": "#e85d04",
    "Expert": "#0077b6",
    "Guest": "#38b000",
    "Narrator": "#9d4edd",
}

MIN_VOICES = 1
MAX_VOICES = 4

PREVIEW_TEXT = "Hello, this is a preview of my voice for your podcast."


def get_voice_list() -> list[dict[str, Any]]:
    """Get list of available voices with fallback."""
    try:
        from storage.voice import get_available_voices

        voices = get_available_voices()
        if voices:
            return voices
    except Exception as e:
        print(f"Warning: Could not load voices: {e}")

    return [
        {"voice_id": "serena", "name": "Serena", "type": "preset"},
        {"voice_id": "ryan", "name": "Ryan", "type": "preset"},
        {"voice_id": "vivian", "name": "Vivian", "type": "preset"},
        {"voice_id": "aiden", "name": "Aiden", "type": "preset"},
    ]


def generate_preview(voice_id: str, voice_type: str) -> str | None:
    """Generate a 3-second preview audio for a voice."""
    try:
        from qwen_tts_ui import get_model
        import soundfile as sf

        if voice_type == "preset":
            model = get_model("1.7B-CustomVoice")
            wavs, sr = model.generate_custom_voice(
                text=PREVIEW_TEXT,
                speaker=voice_id,
                language="auto",
                instruct=None,
                temperature=0.9,
                top_k=50,
                top_p=1.0,
                repetition_penalty=1.05,
                max_new_tokens=256,
                subtalker_temperature=0.9,
                subtalker_top_k=50,
                subtalker_top_p=1.0,
            )
        else:
            import pickle
            from pathlib import Path

            SAVED_VOICES_DIR = Path("saved_voices")
            voice_dir = SAVED_VOICES_DIR / voice_id
            prompt_path = voice_dir / "prompt.pkl"
            meta_path = voice_dir / "metadata.json"

            if not prompt_path.exists():
                return None

            with open(meta_path) as f:
                meta = json.load(f)
            model_name = meta.get("model", "1.7B-Base")

            with open(prompt_path, "rb") as f:
                voice_clone_prompt = pickle.load(f)

            model = get_model(model_name)
            wavs, sr = model.generate_voice_clone(
                text=PREVIEW_TEXT,
                language="auto",
                voice_clone_prompt=voice_clone_prompt,
                temperature=0.9,
                top_k=50,
                top_p=1.0,
                repetition_penalty=1.05,
                max_new_tokens=256,
                subtalker_temperature=0.9,
                subtalker_top_k=50,
                subtalker_top_p=1.0,
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, wavs[0], sr)
            return f.name

    except Exception as e:
        print(f"Preview generation failed: {e}")
        return None


def render_voice_cards(voices: list[dict], selections: dict) -> str:
    """Render HTML for voice cards grid."""

    cards_html = []

    for voice in voices:
        voice_id = voice["voice_id"]
        name = voice.get("name", voice_id)
        voice_type = voice.get("type", "preset")
        is_selected = voice_id in selections
        role = selections.get(voice_id, {}).get("role", "")

        type_badge = "PRESET" if voice_type == "preset" else "SAVED"
        type_class = "badge-preset" if voice_type == "preset" else "badge-saved"
        selected_class = "selected" if is_selected else ""
        role_indicator = (
            f'<div class="role-indicator" style="background: {ROLE_COLORS.get(role, "transparent")}"></div>'
            if role
            else ""
        )

        card_html = f'''
        <div class="voice-card {selected_class}" data-voice-id="{voice_id}">
            {role_indicator}
            <div class="card-content">
                <div class="card-header">
                    <span class="voice-name">{name}</span>
                    <span class="type-badge {type_class}">{type_badge}</span>
                </div>
                <div class="card-status">
                    {"Selected as " + role if is_selected and role else "Click to select"}
                </div>
            </div>
        </div>
        '''
        cards_html.append(card_html)

    return f'<div class="voice-cards-grid">{"".join(cards_html)}</div>'


def get_selection_summary(selections: dict) -> str:
    """Generate selection summary HTML."""
    count = len(selections)

    if count == 0:
        return '<div class="selection-summary empty">No voices selected. Select 1-4 voices to continue.</div>'

    if count < MIN_VOICES:
        return f'<div class="selection-summary warning">Selected {count} voice{"s" if count != 1 else ""}. Need at least {MIN_VOICES}.</div>'

    if count > MAX_VOICES:
        return f'<div class="selection-summary error">Too many voices! Maximum is {MAX_VOICES}.</div>'

    voice_list = ", ".join(
        [
            f'<span class="voice-tag" style="border-left: 3px solid {ROLE_COLORS.get(v["role"], "#666")}">{v["name"]} ({v["role"]})</span>'
            for v in selections.values()
        ]
    )
    return f'<div class="selection-summary valid">{voice_list}</div>'


def toggle_voice_selection(
    voice_id: str, voice_name: str, voice_type: str, role: str, current_selections: dict
) -> tuple[dict, str, str]:
    """Toggle voice selection on/off and update state."""
    selections = current_selections.copy() if current_selections else {}

    if voice_id in selections:
        del selections[voice_id]
    else:
        if len(selections) >= MAX_VOICES:
            return (
                selections,
                get_selection_summary(selections),
                f"Maximum {MAX_VOICES} voices allowed!",
            )

        if not role:
            role = "Guest"

        selections[voice_id] = {
            "voice_id": voice_id,
            "name": voice_name,
            "role": role,
            "type": voice_type,
        }

    summary = get_selection_summary(selections)
    status = f"{'Selected' if voice_id in selections else 'Deselected'} {voice_name}"

    return selections, summary, status


def update_voice_role(
    voice_id: str, new_role: str, current_selections: dict
) -> tuple[dict, str]:
    """Update the role for a selected voice."""
    selections = current_selections.copy() if current_selections else {}

    if voice_id in selections:
        selections[voice_id]["role"] = new_role

    return selections, get_selection_summary(selections)


def validate_selections(selections: dict) -> tuple[bool, str, list[dict]]:
    """Validate current selections and return formatted output."""
    if not selections:
        return False, "No voices selected. Please select 1-4 voices.", []

    count = len(selections)

    if count < MIN_VOICES:
        return (
            False,
            f"Need at least {MIN_VOICES} voices. Currently selected: {count}",
            [],
        )

    if count > MAX_VOICES:
        return (
            False,
            f"Maximum {MAX_VOICES} voices allowed. Currently selected: {count}",
            [],
        )

    for v in selections.values():
        if not v.get("role"):
            return False, f"Please assign a role to {v['name']}", []

    output = [
        {"voice_id": v["voice_id"], "role": v["role"], "type": v["type"]}
        for v in selections.values()
    ]

    return True, f"Valid selection: {count} voices ready!", output


custom_css = """
:root {
    --bg-deep: #f8f9fa;
    --bg-card: #ffffff;
    --bg-card-hover: #f1f3f5;
    --bg-selected: #e9ecef;
    --text-primary: #212529;
    --text-secondary: #6c757d;
    --text-muted: #adb5bd;
    --border-default: #dee2e6;
    --border-selected: #495057;
    --accent-orange: #e85d04;
    --accent-blue: #0077b6;
    --accent-green: #38b000;
    --accent-purple: #9d4edd;
    --glow-selected: 0 0 12px rgba(73, 80, 87, 0.2);
    --radius: 12px;
    --radius-sm: 6px;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
    background: var(--bg-deep) !important;
    min-height: 100vh;
}

.main-header {
    text-align: center;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border-default);
}

.main-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.02em;
}

.sub-title {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin: 0.5rem 0 0;
    font-weight: 400;
}

/* Voice Cards Grid */
.voice-cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 1rem;
    padding: 1rem 0;
}

.voice-card {
    position: relative;
    background: var(--bg-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius);
    padding: 1.25rem;
    cursor: pointer;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    overflow: hidden;
}

.voice-card:hover {
    background: var(--bg-card-hover);
    border-color: #ced4da;
    transform: translateY(-2px);
}

.voice-card.selected {
    background: var(--bg-selected);
    border-color: var(--border-selected);
    box-shadow: var(--glow-selected);
}

.role-indicator {
    position: absolute;
    left: 0;
    top: 0;
    width: 4px;
    height: 100%;
    border-radius: var(--radius) 0 0 var(--radius);
}

.card-content {
    position: relative;
    z-index: 1;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.75rem;
}

.voice-name {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}

.type-badge {
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.25rem 0.5rem;
    border-radius: var(--radius-sm);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-preset {
    background: rgba(73, 80, 87, 0.1);
    color: #495057;
    border: 1px solid rgba(73, 80, 87, 0.2);
}

.badge-saved {
    background: rgba(56, 176, 0, 0.1);
    color: #2d8a00;
    border: 1px solid rgba(56, 176, 0, 0.25);
}

.card-status {
    font-size: 0.8rem;
    color: var(--text-muted);
}

.voice-card.selected .card-status {
    color: var(--text-secondary);
}

/* Selection Summary */
.selection-summary {
    padding: 1rem 1.25rem;
    border-radius: var(--radius);
    margin: 1rem 0;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: center;
}

.selection-summary.empty {
    background: rgba(0, 0, 0, 0.02);
    border: 1px dashed var(--border-default);
    color: var(--text-muted);
    justify-content: center;
}

.selection-summary.warning {
    background: rgba(232, 93, 4, 0.1);
    border: 1px solid rgba(232, 93, 4, 0.3);
    color: var(--accent-orange);
}

.selection-summary.error {
    background: rgba(220, 53, 69, 0.1);
    border: 1px solid rgba(220, 53, 69, 0.3);
    color: #ff6b6b;
}

.selection-summary.valid {
    background: rgba(56, 176, 0, 0.08);
    border: 1px solid rgba(56, 176, 0, 0.25);
}

.voice-tag {
    background: var(--bg-card);
    padding: 0.4rem 0.75rem 0.4rem 1rem;
    border-radius: var(--radius-sm);
    font-size: 0.85rem;
    color: var(--text-primary);
}

/* Controls Panel */
.controls-panel {
    background: var(--bg-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius);
    padding: 1.25rem;
    margin-top: 1rem;
}

.panel-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border-default);
}

/* Form Controls - Light Theme */
.dark-form input, .dark-form select, .dark-form textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-default) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
}

.dark-form label {
    color: var(--text-secondary) !important;
}

/* Primary Button */
.primary-btn {
    background: linear-gradient(135deg, #343a40 0%, #495057 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: var(--radius-sm) !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}

.primary-btn:hover {
    background: linear-gradient(135deg, #495057 0%, #6c757d 100%) !important;
    box-shadow: 0 4px 15px rgba(73, 80, 87, 0.25) !important;
}

/* Preview Button */
.preview-btn {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-default) !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1rem !important;
    border-radius: var(--radius-sm) !important;
}

.preview-btn:hover {
    background: var(--bg-card-hover) !important;
    border-color: #ced4da !important;
    color: var(--text-primary) !important;
}

/* Role Dropdown Styling */
.role-dropdown {
    min-width: 120px;
}

/* Audio Player */
.gradio-audio {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-sm) !important;
}

/* Validation Output */
.validation-output {
    background: var(--bg-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius);
    padding: 1rem;
    margin-top: 0.5rem;
}

.validation-success {
    color: var(--accent-green);
}

.validation-error {
    color: #ff6b6b;
}

/* Responsive */
@media (max-width: 768px) {
    .voice-cards-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    .main-title {
        font-size: 1.5rem;
    }
}

@media (max-width: 480px) {
    .voice-cards-grid {
        grid-template-columns: 1fr;
    }
}
"""


def create_voice_cards_ui() -> gr.Blocks:
    """Create the voice cards selection UI."""

    voices = get_voice_list()

    with gr.Blocks(
        css=custom_css, title="Voice Selection", theme=gr.themes.Base()
    ) as demo:
        selections_state = gr.State({})
        voices_state = gr.State(voices)

        gr.HTML("""
        <div class="main-header">
            <h1 class="main-title">Voice Selection</h1>
            <p class="sub-title">Choose 1-4 voices for your podcast</p>
        </div>
        """)

        summary_html = gr.HTML(
            value=get_selection_summary({}),
            elem_classes=["selection-summary-container"],
        )

        with gr.Row():
            with gr.Column(scale=3):
                gr.HTML('<div class="panel-title">Available Voices</div>')

                voice_rows = []
                for i, voice in enumerate(voices):
                    with gr.Row(elem_classes=["voice-row"]):
                        cb = gr.Checkbox(
                            label=f"{voice['name']} ({voice['type'].upper()})",
                            value=False,
                            scale=2,
                        )
                        role = gr.Dropdown(
                            choices=ROLES,
                            value="Guest",
                            label="Role",
                            scale=1,
                            interactive=True,
                        )
                        preview_btn = gr.Button(
                            "Preview", size="sm", elem_classes=["preview-btn"], scale=1
                        )
                        voice_rows.append((voice, cb, role, preview_btn))

            with gr.Column(scale=1):
                gr.HTML('<div class="panel-title">Preview</div>')
                preview_audio = gr.Audio(
                    label="Voice Preview", type="filepath", interactive=False
                )
                preview_status = gr.Textbox(label="Status", interactive=False, lines=1)

        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="panel-title">Finalize Selection</div>')

                with gr.Row():
                    validate_btn = gr.Button(
                        "Validate & Continue",
                        variant="primary",
                        elem_classes=["primary-btn"],
                    )
                    refresh_btn = gr.Button(
                        "Refresh Voices", elem_classes=["preview-btn"]
                    )

                validation_status = gr.HTML(value="")
                output_json = gr.JSON(label="Selection Output", visible=True)

        def make_toggle_handler(voice_info):
            def handler(is_checked, role_val, current_sels):
                sels = current_sels.copy() if current_sels else {}
                vid = voice_info["voice_id"]

                if is_checked:
                    if len(sels) < MAX_VOICES:
                        sels[vid] = {
                            "voice_id": vid,
                            "name": voice_info["name"],
                            "role": role_val,
                            "type": voice_info["type"],
                        }
                else:
                    sels.pop(vid, None)

                return sels, get_selection_summary(sels)

            return handler

        def make_role_handler(voice_info):
            def handler(new_role, current_sels):
                sels = current_sels.copy() if current_sels else {}
                vid = voice_info["voice_id"]

                if vid in sels:
                    sels[vid]["role"] = new_role

                return sels, get_selection_summary(sels)

            return handler

        def make_preview_handler(voice_info):
            def handler():
                audio_path = generate_preview(
                    voice_info["voice_id"], voice_info["type"]
                )
                if audio_path:
                    return audio_path, f"Preview: {voice_info['name']}"
                return (
                    None,
                    "Preview generation failed. Check if TTS model is available.",
                )

            return handler

        for voice_info, checkbox, role_dd, prev_btn in voice_rows:
            checkbox.change(
                fn=make_toggle_handler(voice_info),
                inputs=[checkbox, role_dd, selections_state],
                outputs=[selections_state, summary_html],
            )

            role_dd.change(
                fn=make_role_handler(voice_info),
                inputs=[role_dd, selections_state],
                outputs=[selections_state, summary_html],
            )

            prev_btn.click(
                fn=make_preview_handler(voice_info),
                inputs=[],
                outputs=[preview_audio, preview_status],
            )

        def handle_validate(selections):
            is_valid, message, output = validate_selections(selections)

            if is_valid:
                status_html = (
                    f'<div class="validation-output validation-success">{message}</div>'
                )
            else:
                status_html = (
                    f'<div class="validation-output validation-error">{message}</div>'
                )

            return status_html, output if is_valid else {}

        validate_btn.click(
            fn=handle_validate,
            inputs=[selections_state],
            outputs=[validation_status, output_json],
        )

        def handle_refresh():
            new_voices = get_voice_list()
            return f"Loaded {len(new_voices)} voices"

        refresh_btn.click(fn=handle_refresh, inputs=[], outputs=[preview_status])

    return demo


def get_voice_selection_components():
    """Create reusable voice selection components for embedding in other UIs."""
    voices = get_voice_list()
    selections_state = gr.State({})

    summary = gr.HTML(value=get_selection_summary({}))

    voice_checkboxes = []
    role_dropdowns = []

    for voice in voices[:8]:
        cb = gr.Checkbox(label=f"{voice['name']} ({voice['type']})", value=False)
        role = gr.Dropdown(choices=ROLES, value="Guest", label="Role", interactive=True)
        voice_checkboxes.append((voice, cb))
        role_dropdowns.append(role)

    return selections_state, summary, voice_checkboxes, role_dropdowns


if __name__ == "__main__":
    demo = create_voice_cards_ui()
    demo.launch(server_name="127.0.0.1", server_port=7862, share=False, show_error=True)

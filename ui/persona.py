# -*- coding: utf-8 -*-
"""Gradio UI components for the Personas management tab.

Provides a visual interface for creating, editing, and managing voice personas.
Each persona defines character traits, speaking style, and expertise for a voice.
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from storage.persona_models import (
    ALLOWED_PERSONALITIES,
    ALLOWED_SPEAKING_STYLES,
    Persona,
)
from storage.persona import (
    delete_persona,
    list_personas,
    load_persona,
    save_persona,
)
from storage.voice import get_available_voices


PERSONA_CSS = """
.persona-cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1rem;
    padding: 1rem 0;
}

.persona-card {
    position: relative;
    background: #14141f;
    border: 1px solid #2a2a40;
    border-radius: 12px;
    padding: 1.25rem;
    cursor: pointer;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    overflow: hidden;
}

.persona-card:hover {
    background: #1a1a2e;
    border-color: #3a3a55;
    transform: translateY(-2px);
}

.persona-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.75rem;
}

.persona-name {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f0f0f5;
    letter-spacing: -0.01em;
}

.persona-voice-badge {
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.25rem 0.5rem;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: rgba(80, 80, 255, 0.15);
    color: #8080ff;
    border: 1px solid rgba(80, 80, 255, 0.3);
}

.persona-traits {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
}

.persona-trait {
    font-size: 0.75rem;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.05);
    color: #8888a0;
}

.persona-bio {
    font-size: 0.8rem;
    color: #555570;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

.persona-gallery-empty {
    padding: 2rem;
    text-align: center;
    color: #555570;
    border: 1px dashed #2a2a40;
    border-radius: 12px;
}

.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #f0f0f5;
    margin-bottom: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2a2a40;
}
"""


def _get_voice_choices() -> list[tuple[str, str]]:
    """Get voice choices for dropdown.
    
    Returns:
        List of (display_name, value) tuples for Gradio dropdown.
    """
    voices = get_available_voices()
    choices = []
    for voice in voices:
        voice_id = voice.get("voice_id", "")
        name = voice.get("name", voice_id)
        voice_type = voice.get("type", "preset")
        display = f"{name} ({voice_type})"
        value = f"{voice_id}|{voice_type}"
        choices.append((display, value))
    return choices


def _parse_voice_value(value: str) -> tuple[str, str]:
    """Parse voice dropdown value into (voice_id, voice_type).
    
    Args:
        value: Combined value in format "voice_id|voice_type"
        
    Returns:
        Tuple of (voice_id, voice_type)
    """
    if not value or "|" not in value:
        return "", ""
    parts = value.split("|", 1)
    return parts[0], parts[1]


def _render_persona_cards(personas: list[tuple[str, str, Persona]]) -> str:
    """Render HTML for persona cards gallery.
    
    Args:
        personas: List of (voice_id, voice_type, persona) tuples.
        
    Returns:
        HTML string for the personas gallery.
    """
    if not personas:
        return '<div class="persona-gallery-empty">No personas saved yet. Create one above!</div>'
    
    cards_html = []
    for voice_id, voice_type, persona in personas:
        traits_html = f'''
            <span class="persona-trait">{persona.personality}</span>
            <span class="persona-trait">{persona.speaking_style}</span>
        '''
        if persona.expertise:
            for exp in persona.expertise[:2]:
                traits_html += f'<span class="persona-trait">{exp}</span>'
        
        bio_preview = persona.bio[:100] + "..." if len(persona.bio) > 100 else persona.bio
        
        card_html = f'''
        <div class="persona-card" onclick="document.getElementById('persona-select-{voice_id}-{voice_type}').click()">
            <div class="persona-card-header">
                <span class="persona-name">{persona.character_name}</span>
                <span class="persona-voice-badge">{voice_type.upper()}</span>
            </div>
            <div class="persona-traits">{traits_html}</div>
            <div class="persona-bio">{bio_preview or "No bio"}</div>
            <div style="font-size: 0.7rem; color: #555570; margin-top: 0.5rem;">
                Voice: {voice_id}
            </div>
        </div>
        '''
        cards_html.append(card_html)
    
    return f'<div class="persona-cards-grid">{"".join(cards_html)}</div>'


def _generate_voice_preview(voice_id: str, voice_type: str) -> str | None:
    """Generate a preview audio for a voice.
    
    Args:
        voice_id: The voice identifier.
        voice_type: Type of voice ("preset" or "saved").
        
    Returns:
        Path to the generated audio file, or None if generation fails.
    """
    try:
        from voice_cards_ui import generate_preview
        return generate_preview(voice_id, voice_type)
    except Exception as e:
        print(f"Voice preview generation failed: {e}")
        return None


def create_personas_tab() -> None:
    """Create the Personas management tab.
    
    Renders persona management UI components in the current Gradio context.
    This function should be called within a TabItem context.
    """
    # Temporary: Comment out CSS injection to debug nested UI bug
    # gr.HTML(f"<style>{PERSONA_CSS}</style>")
    
    gr.Markdown("## Persona Management")
    gr.Markdown("*Define character personas for your podcast voices*")

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="section-header">Voice Selection</div>')
        
            voice_dropdown = gr.Dropdown(
                label="Select Voice",
                choices=_get_voice_choices(),
                value=None,
                interactive=True,
                info="Choose a voice to create or edit its persona"
            )
        
            refresh_voices_btn = gr.Button("Refresh Voices", size="sm")
    
        with gr.Column(scale=2):
            gr.HTML('<div class="section-header">Character Definition</div>')
        
            character_name = gr.Textbox(
                label="Character Name",
                placeholder="e.g., Dr. Sarah Chen, The Wise Narrator",
                info="Display name for this character"
            )
        
            with gr.Row():
                personality = gr.Dropdown(
                    label="Personality",
                    choices=sorted(ALLOWED_PERSONALITIES),
                    value=None,
                    interactive=True,
                    info="Core personality trait"
                )
                speaking_style = gr.Dropdown(
                    label="Speaking Style",
                    choices=sorted(ALLOWED_SPEAKING_STYLES),
                    value=None,
                    interactive=True,
                    info="How they communicate"
                )
        
            expertise = gr.Textbox(
                label="Expertise (comma-separated)",
                placeholder="e.g., AI Ethics, Philosophy, Technology",
                info="Areas of knowledge or expertise"
            )
        
            background = gr.Textbox(
                label="Background",
                placeholder="Brief background information about the character...",
                lines=2,
                info="Character's history, role, or context"
            )
        
            bio = gr.Textbox(
                label="Bio / Personality Notes",
                placeholder="Detailed character description, personality quirks, mannerisms...",
                lines=3,
                info="Extended character description for transcript generation"
            )
        
            with gr.Row():
                save_btn = gr.Button(
                    "Save Persona",
                    variant="primary",
                    size="lg"
                )
                delete_btn = gr.Button(
                    "Delete Persona",
                    variant="stop",
                    size="lg"
                )
                preview_btn = gr.Button(
                    "Preview Voice",
                    size="lg"
                )
        
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                show_label=True
            )
        
            preview_audio = gr.Audio(
                label="Voice Preview",
                type="filepath",
                visible=True
            )

    gr.HTML('<div class="section-header" style="margin-top: 2rem;">Saved Personas Gallery</div>')

    personas_gallery = gr.HTML(
        value=_render_persona_cards(list_personas())
    )

    refresh_gallery_btn = gr.Button("Refresh Gallery", size="sm")

    selected_voice_state = gr.State(value=None)
    persona_delete_confirm = gr.State(value=False)

    def on_voice_select(voice_value: str) -> tuple[str, str, str | None, str | None, str, str, str, str, bool]:
        """Handle voice selection - load existing persona if available.
    
        Args:
            voice_value: Combined voice value "voice_id|voice_type"
        
        Returns:
            Tuple of (character_name, expertise, personality, speaking_style, 
                     background, bio, status, voice_value, confirm_reset)
        """
        if not voice_value:
            return "", "", None, None, "", "", "Select a voice to begin", voice_value, False
    
        voice_id, voice_type = _parse_voice_value(voice_value)
    
        if not voice_id:
            return "", "", None, None, "", "", "Invalid voice selection", voice_value, False
    
        existing = load_persona(voice_id, voice_type)
    
        if existing:
            expertise_str = ", ".join(existing.expertise) if existing.expertise else ""
            return (
                existing.character_name,
                expertise_str,
                existing.personality,
                existing.speaking_style,
                existing.background,
                existing.bio,
                f"Loaded persona for {voice_id}",
                voice_value,
                False
            )
        else:
            return (
                "",
                "",
                None,
                None,
                "",
                "",
                f"No persona found for {voice_id}. Create one!",
                voice_value,
                False
            )

    def on_save_persona(
        voice_value: str,
        char_name: str,
        pers: str,
        style: str,
        exp: str,
        bg: str,
        bio_text: str
    ) -> tuple[str, str]:
        """Save persona with validation.
    
        Args:
            voice_value: Combined voice value "voice_id|voice_type"
            char_name: Character name
            pers: Personality trait
            style: Speaking style
            exp: Comma-separated expertise list
            bg: Background text
            bio_text: Bio/personality notes
        
        Returns:
            Tuple of (status_message, updated_gallery_html)
        """
        if not voice_value:
            gr.Warning("Please select a voice first")
            return "Error: No voice selected", _render_persona_cards(list_personas())
    
        voice_id, voice_type = _parse_voice_value(voice_value)
    
        if not voice_id:
            gr.Warning("Invalid voice selection")
            return "Error: Invalid voice", _render_persona_cards(list_personas())
    
        if not char_name or not char_name.strip():
            gr.Warning("Character name is required")
            return "Error: Character name required", _render_persona_cards(list_personas())
    
        if not pers:
            gr.Warning("Personality is required")
            return "Error: Personality required", _render_persona_cards(list_personas())
    
        if not style:
            gr.Warning("Speaking style is required")
            return "Error: Speaking style required", _render_persona_cards(list_personas())
    
        expertise_list = [e.strip() for e in exp.split(",") if e.strip()] if exp else []
    
        try:
            persona = Persona(
                voice_id=voice_id,
                voice_type=voice_type,
                character_name=char_name.strip(),
                personality=pers,
                speaking_style=style,
                expertise=expertise_list,
                background=bg.strip() if bg else "",
                bio=bio_text.strip() if bio_text else ""
            )
        
            save_persona(persona)
            gr.Info(f"Persona saved for {char_name}")
            return f"Saved persona: {char_name}", _render_persona_cards(list_personas())
        
        except ValueError as e:
            gr.Warning(f"Validation error: {e}")
            return f"Error: {e}", _render_persona_cards(list_personas())
        except Exception as e:
            gr.Warning(f"Failed to save: {e}")
            return f"Error saving persona: {e}", _render_persona_cards(list_personas())

    def on_delete_persona(
        voice_value: str,
        confirm_state: bool
    ) -> tuple[str, str, str, str | None, str | None, str, str, str, bool]:
        """Delete persona with two-step confirmation.
    
        Args:
            voice_value: Combined voice value "voice_id|voice_type"
            confirm_state: Whether first confirmation click has occurred
        
        Returns:
            Tuple of (status, gallery_html, char_name, personality, style, 
                     expertise, background, bio, confirm_state)
        """
        gallery_html = _render_persona_cards(list_personas())
    
        if not voice_value:
            gr.Warning("Please select a voice first")
            return (
                "Error: No voice selected",
                gallery_html,
                "", None, None, "", "", "",
                False
            )
    
        voice_id, voice_type = _parse_voice_value(voice_value)
    
        if not voice_id:
            gr.Warning("Invalid voice selection")
            return (
                "Error: Invalid voice",
                gallery_html,
                "", None, None, "", "", "",
                False
            )
    
        # First click: Request confirmation
        if not confirm_state:
            gr.Warning(f"⚠️ Click Delete again to confirm deletion of persona for '{voice_id}'")
            return (
                f"Click Delete again to confirm deletion for {voice_id}",
                gallery_html,
                gr.update(), gr.update(), gr.update(), 
                gr.update(), gr.update(), gr.update(),
                True
            )
    
        # Second click: Actually delete
        try:
            deleted = delete_persona(voice_id, voice_type)
        
            if deleted:
                gr.Info(f"✅ Persona deleted for {voice_id}")
                return (
                    f"Deleted persona for {voice_id}",
                    _render_persona_cards(list_personas()),
                    "", None, None, "", "", "",
                    False
                )
            else:
                gr.Warning(f"❌ No persona found for {voice_id}")
                return (
                    f"No persona found for {voice_id}",
                    gallery_html,
                    "", None, None, "", "", "",
                    False
                )
            
        except Exception as e:
            gr.Warning(f"Failed to delete: {e}")
            return (
                f"Error deleting persona: {e}",
                gallery_html,
                "", None, None, "", "", "",
                False
            )

    def on_preview_voice(voice_value: str) -> tuple[str | None, str]:
        """Generate voice preview audio.
    
        Args:
            voice_value: Combined voice value "voice_id|voice_type"
        
        Returns:
            Tuple of (audio_path, status_message)
        """
        if not voice_value:
            gr.Warning("Please select a voice first")
            return None, "Select a voice to preview"
    
        voice_id, voice_type = _parse_voice_value(voice_value)
    
        if not voice_id:
            gr.Warning("Invalid voice selection")
            return None, "Invalid voice selection"
    
        audio_path = _generate_voice_preview(voice_id, voice_type)
    
        if audio_path:
            gr.Info("Voice preview generated!")
            return audio_path, f"Preview generated for {voice_id}"
        else:
            gr.Warning("Failed to generate preview")
            return None, "Failed to generate voice preview"

    def on_refresh_voices() -> dict[str, Any]:
        """Refresh voice dropdown choices.
    
        Returns:
            Gradio update dict for dropdown.
        """
        choices = _get_voice_choices()
        return gr.update(choices=choices, value=None)

    def on_refresh_gallery() -> str:
        """Refresh personas gallery.
    
        Returns:
            Updated gallery HTML.
        """
        return _render_persona_cards(list_personas())

    voice_dropdown.change(
        fn=on_voice_select,
        inputs=[voice_dropdown],
        outputs=[
            character_name,
            expertise,
            personality,
            speaking_style,
            background,
            bio,
            status_text,
            selected_voice_state,
            persona_delete_confirm
        ]
    )

    save_btn.click(
        fn=on_save_persona,
        inputs=[
            voice_dropdown,
            character_name,
            personality,
            speaking_style,
            expertise,
            background,
            bio
        ],
        outputs=[status_text, personas_gallery]
    )

    delete_btn.click(
        fn=on_delete_persona,
        inputs=[voice_dropdown, persona_delete_confirm],
        outputs=[
            status_text,
            personas_gallery,
            character_name,
            personality,
            speaking_style,
            expertise,
            background,
            bio,
            persona_delete_confirm
        ]
    )

    preview_btn.click(
        fn=on_preview_voice,
        inputs=[voice_dropdown],
        outputs=[preview_audio, status_text]
    )

    refresh_voices_btn.click(
        fn=on_refresh_voices,
        outputs=[voice_dropdown]
    )

    refresh_gallery_btn.click(
        fn=on_refresh_gallery,
        outputs=[personas_gallery]
        )


if False:  # Disabled - causes nested UI bug when imported
    with gr.Blocks() as demo:
        create_personas_tab()
    demo.launch()

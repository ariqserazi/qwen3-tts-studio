#!/usr/bin/env python3
"""Draft Preview UI for podcast generation.

A two-column layout for reviewing and editing podcast outlines and dialogues:
- Left: Expandable outline tree with segments
- Right: Dialogue editor for selected segment

Run standalone: python3 draft_preview_ui.py
"""

import os

os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import copy
from typing import Any

import gradio as gr

from podcast.models import Dialogue, Outline, Segment, Transcript


# =============================================================================
# MOCK DATA
# =============================================================================


def create_mock_outline() -> Outline:
    """Create mock outline data for testing."""
    return Outline(
        segments=[
            Segment(
                title="Introduction",
                description="Welcome listeners and introduce today's topic on AI in healthcare.",
                size="short",
            ),
            Segment(
                title="The Current State of AI in Healthcare",
                description="Overview of how AI is currently being used in diagnostics, treatment planning, and patient care.",
                size="medium",
            ),
            Segment(
                title="Expert Interview: Dr. Sarah Chen",
                description="Deep dive with our guest expert on breakthrough AI applications in radiology.",
                size="long",
            ),
            Segment(
                title="Ethical Considerations",
                description="Discussion on privacy, bias, and accountability in medical AI systems.",
                size="medium",
            ),
            Segment(
                title="Wrap-up & Next Episode Preview",
                description="Summarize key takeaways and tease next week's episode on AI in education.",
                size="short",
            ),
        ]
    )


def create_mock_transcript() -> dict[int, Transcript]:
    """Create mock transcript data for each segment (indexed by segment number)."""
    return {
        0: Transcript(
            dialogues=[
                Dialogue(
                    speaker="Alex",
                    text="Welcome to Tech Horizons, your weekly deep dive into emerging technologies!",
                ),
                Dialogue(
                    speaker="Alex",
                    text="I'm your host Alex, and today we're exploring one of the most transformative applications of AI: healthcare.",
                ),
                Dialogue(
                    speaker="Riley",
                    text="And I'm Riley. This is genuinely one of the topics I'm most excited about.",
                ),
            ]
        ),
        1: Transcript(
            dialogues=[
                Dialogue(
                    speaker="Alex",
                    text="So Riley, let's start with the big picture. Where is AI making the biggest impact in healthcare today?",
                ),
                Dialogue(
                    speaker="Riley",
                    text="Great question. The three main areas are diagnostics, drug discovery, and personalized treatment plans.",
                ),
                Dialogue(
                    speaker="Riley",
                    text="In diagnostics alone, AI systems are now matching or exceeding human radiologists in detecting certain cancers.",
                ),
                Dialogue(
                    speaker="Alex",
                    text="That's remarkable. What about the accuracy concerns we hear about?",
                ),
                Dialogue(
                    speaker="Riley",
                    text="Valid point. While accuracy is high, the real challenge is ensuring these systems work across diverse patient populations.",
                ),
            ]
        ),
        2: Transcript(
            dialogues=[
                Dialogue(
                    speaker="Alex",
                    text="We're thrilled to have Dr. Sarah Chen joining us today. Dr. Chen, welcome to the show!",
                ),
                Dialogue(
                    speaker="Sarah",
                    text="Thank you for having me. It's wonderful to be here.",
                ),
                Dialogue(
                    speaker="Alex",
                    text="You've been pioneering AI applications in radiology. Can you tell us about your latest research?",
                ),
                Dialogue(
                    speaker="Sarah",
                    text="Absolutely. We've developed a system that can detect early-stage lung cancer with 94% accuracy from CT scans.",
                ),
                Dialogue(
                    speaker="Riley",
                    text="That's incredible. How does this compare to traditional detection methods?",
                ),
                Dialogue(
                    speaker="Sarah",
                    text="Traditional methods catch about 70% of early-stage cases. So we're seeing a significant improvement.",
                ),
                Dialogue(
                    speaker="Alex",
                    text="What inspired you to focus on this particular application?",
                ),
                Dialogue(
                    speaker="Sarah",
                    text="Personal experience, actually. I lost my grandmother to lung cancer that was caught too late.",
                ),
            ]
        ),
        3: Transcript(
            dialogues=[
                Dialogue(
                    speaker="Alex",
                    text="Let's shift to the ethical side. Riley, what are the main concerns?",
                ),
                Dialogue(
                    speaker="Riley",
                    text="There are three big ones: patient privacy, algorithmic bias, and the question of accountability.",
                ),
                Dialogue(
                    speaker="Riley",
                    text="If an AI makes a wrong diagnosis, who's responsible? The developer? The hospital? The doctor who relied on it?",
                ),
                Dialogue(
                    speaker="Alex",
                    text="Those are thorny questions. Are there any regulatory frameworks emerging?",
                ),
                Dialogue(
                    speaker="Riley",
                    text="The FDA has started approving AI medical devices, but the regulatory landscape is still catching up.",
                ),
            ]
        ),
        4: Transcript(
            dialogues=[
                Dialogue(
                    speaker="Alex",
                    text="What a fascinating discussion! Let's recap the key takeaways.",
                ),
                Dialogue(
                    speaker="Riley",
                    text="AI in healthcare is advancing rapidly, with real benefits in diagnostics and treatment.",
                ),
                Dialogue(
                    speaker="Riley",
                    text="But we need to address ethical concerns around privacy, bias, and accountability.",
                ),
                Dialogue(
                    speaker="Alex",
                    text="Next week, we'll be exploring AI in education. You won't want to miss it!",
                ),
                Dialogue(
                    speaker="Alex",
                    text="Thanks for listening to Tech Horizons. Until next time!",
                ),
            ]
        ),
    }


# =============================================================================
# CSS STYLING
# =============================================================================

DRAFT_PREVIEW_CSS = """
:root {
    --slate-50: #f8fafc;
    --slate-100: #f1f5f9;
    --slate-200: #e2e8f0;
    --slate-300: #cbd5e1;
    --slate-400: #94a3b8;
    --slate-500: #64748b;
    --slate-600: #475569;
    --slate-700: #334155;
    --slate-800: #1e293b;
    --slate-900: #0f172a;
    --amber-100: #fef3c7;
    --amber-200: #fde68a;
    --amber-500: #f59e0b;
    --emerald-100: #d1fae5;
    --emerald-500: #10b981;
    --radius: 6px;
}

.gradio-container {
    max-width: 1600px !important;
    margin: 0 auto;
    background: var(--slate-50) !important;
}

/* ===== HEADER ===== */
.draft-header {
    text-align: center;
    padding: 1.5rem 0;
    margin-bottom: 1.5rem;
    border-bottom: 2px solid var(--slate-200);
    background: linear-gradient(to bottom, var(--slate-100), var(--slate-50));
}

.draft-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--slate-800);
    margin: 0;
    letter-spacing: -0.02em;
}

.draft-subtitle {
    color: var(--slate-500);
    font-size: 0.95rem;
    margin: 0.5rem 0 0;
}

/* ===== PANEL HEADERS ===== */
.panel-header {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--slate-700);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
    padding: 0.75rem 1rem;
    background: var(--slate-100);
    border-radius: var(--radius);
    border-left: 4px solid var(--slate-600);
}

/* ===== SEGMENT ACCORDION ===== */
.segment-accordion {
    margin-bottom: 0.75rem !important;
}

.segment-accordion .label-wrap {
    background: white !important;
    border: 1px solid var(--slate-200) !important;
    border-radius: var(--radius) !important;
    padding: 0.875rem 1rem !important;
    transition: all 0.15s ease !important;
}

.segment-accordion .label-wrap:hover {
    background: var(--slate-50) !important;
    border-color: var(--slate-400) !important;
}

.segment-accordion.edited .label-wrap {
    border-left: 4px solid var(--amber-500) !important;
    background: var(--amber-100) !important;
}

/* ===== SEGMENT CONTENT ===== */
.segment-content {
    padding: 1rem;
    background: var(--slate-50);
    border: 1px solid var(--slate-200);
    border-top: none;
    border-radius: 0 0 var(--radius) var(--radius);
}

.segment-size-badge {
    display: inline-block;
    padding: 0.25rem 0.625rem;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    border-radius: 99px;
    margin-left: 0.5rem;
}

.size-short {
    background: var(--emerald-100);
    color: var(--emerald-500);
}

.size-medium {
    background: var(--slate-200);
    color: var(--slate-600);
}

.size-long {
    background: var(--amber-100);
    color: var(--amber-500);
}

/* ===== DIALOGUE PANEL ===== */
.dialogue-panel {
    background: white;
    border: 1px solid var(--slate-200);
    border-radius: var(--radius);
    padding: 1.25rem;
    min-height: 500px;
}

.dialogue-item {
    padding: 1rem;
    margin-bottom: 0.75rem;
    background: var(--slate-50);
    border: 1px solid var(--slate-200);
    border-radius: var(--radius);
    transition: all 0.15s ease;
}

.dialogue-item:hover {
    border-color: var(--slate-400);
}

.dialogue-item.edited {
    border-left: 4px solid var(--amber-500);
    background: var(--amber-100);
}

.dialogue-speaker {
    font-weight: 600;
    color: var(--slate-800);
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.speaker-icon {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--slate-600);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
}

.dialogue-text {
    color: var(--slate-700);
    font-size: 0.95rem;
    line-height: 1.6;
    padding-left: 2.25rem;
}

/* ===== EDIT BUTTONS ===== */
.edit-btn {
    padding: 0.375rem 0.75rem !important;
    font-size: 0.8rem !important;
    background: white !important;
    border: 1px solid var(--slate-300) !important;
    color: var(--slate-600) !important;
    border-radius: var(--radius) !important;
}

.edit-btn:hover {
    background: var(--slate-100) !important;
    border-color: var(--slate-400) !important;
}

.save-btn {
    background: var(--slate-800) !important;
    color: white !important;
    border: none !important;
}

.save-btn:hover {
    background: var(--slate-900) !important;
}

/* ===== EMPTY STATE ===== */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: var(--slate-400);
}

.empty-state-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}

.empty-state-text {
    font-size: 1rem;
    color: var(--slate-500);
}

/* ===== EDIT INDICATOR ===== */
.edit-indicator {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    font-size: 0.7rem;
    font-weight: 600;
    background: var(--amber-200);
    color: var(--amber-500);
    border-radius: var(--radius);
    margin-left: 0.5rem;
}

/* ===== STATUS BAR ===== */
.status-bar {
    padding: 0.75rem 1rem;
    background: var(--slate-100);
    border: 1px solid var(--slate-200);
    border-radius: var(--radius);
    font-size: 0.85rem;
    color: var(--slate-600);
    margin-top: 1rem;
}

/* ===== SCROLLABLE CONTAINER ===== */
.outline-scroll {
    max-height: 600px;
    overflow-y: auto;
    padding-right: 0.5rem;
}

.dialogue-scroll {
    max-height: 550px;
    overflow-y: auto;
}
"""


# =============================================================================
# UI HELPER FUNCTIONS
# =============================================================================


def get_size_badge_html(size: str) -> str:
    """Get HTML for size badge."""
    return f'<span class="segment-size-badge size-{size}">{size}</span>'


def get_speaker_initial(speaker: str) -> str:
    """Get first letter of speaker name for icon."""
    return speaker[0].upper() if speaker else "?"


def format_segment_header(segment: Segment, idx: int, is_edited: bool) -> str:
    """Format segment accordion header."""
    edited_badge = '<span class="edit-indicator">EDITED</span>' if is_edited else ""
    return f"{idx + 1}. {segment.title} {get_size_badge_html(segment.size)}{edited_badge}"


def render_dialogues_html(
    dialogues: list[Dialogue], edited_indices: set[int]
) -> str:
    """Render dialogues as HTML."""
    if not dialogues:
        return """
        <div class="empty-state">
            <div class="empty-state-icon">💬</div>
            <div class="empty-state-text">No dialogues for this segment</div>
        </div>
        """

    html_parts = []
    for idx, dialogue in enumerate(dialogues):
        edited_class = "edited" if idx in edited_indices else ""
        initial = get_speaker_initial(dialogue.speaker)
        html_parts.append(f"""
        <div class="dialogue-item {edited_class}" data-idx="{idx}">
            <div class="dialogue-speaker">
                <span class="speaker-icon">{initial}</span>
                {dialogue.speaker}
                {"<span class='edit-indicator'>EDITED</span>" if idx in edited_indices else ""}
            </div>
            <div class="dialogue-text">{dialogue.text}</div>
        </div>
        """)

    return "".join(html_parts)


# =============================================================================
# STATE MANAGEMENT
# =============================================================================


def initialize_state() -> dict[str, Any]:
    """Initialize application state."""
    return {
        "outline": create_mock_outline(),
        "transcripts": create_mock_transcript(),
        "selected_segment": 0,
        "edited_segments": set(),  # Set of segment indices that have been edited
        "edited_dialogues": {},  # Dict[segment_idx, Set[dialogue_idx]]
    }


def get_segment_dialogues(
    state: dict[str, Any], segment_idx: int
) -> list[Dialogue]:
    """Get dialogues for a specific segment."""
    transcripts = state.get("transcripts", {})
    if segment_idx in transcripts:
        return transcripts[segment_idx].dialogues
    return []


# =============================================================================
# EVENT HANDLERS
# =============================================================================


def on_segment_select(
    segment_idx: int, state: dict[str, Any]
) -> tuple[str, dict[str, Any], gr.update, gr.update, gr.update]:
    """Handle segment selection."""
    state = copy.deepcopy(state)
    state["selected_segment"] = segment_idx

    dialogues = get_segment_dialogues(state, segment_idx)
    edited_dialogues = state.get("edited_dialogues", {}).get(segment_idx, set())
    dialogues_html = render_dialogues_html(dialogues, edited_dialogues)

    # Get segment info for display
    outline = state.get("outline")
    segment = outline.segments[segment_idx] if outline and segment_idx < len(outline.segments) else None

    segment_title = segment.title if segment else ""
    segment_desc = segment.description if segment else ""

    # Show edit panel with current values
    return (
        dialogues_html,
        state,
        gr.update(value=segment_title, visible=True),
        gr.update(value=segment_desc, visible=True),
        gr.update(visible=True),
    )


def on_save_segment_edit(
    new_title: str,
    new_desc: str,
    state: dict[str, Any],
) -> tuple[str, dict[str, Any], str]:
    """Save segment title/description edits."""
    state = copy.deepcopy(state)
    segment_idx = state.get("selected_segment", 0)
    outline = state.get("outline")

    if outline and segment_idx < len(outline.segments):
        segment = outline.segments[segment_idx]
        if new_title.strip() != segment.title or new_desc.strip() != segment.description:
            # Update segment
            outline.segments[segment_idx] = Segment(
                title=new_title.strip(),
                description=new_desc.strip(),
                size=segment.size,
            )
            state["outline"] = outline

            # Mark as edited
            edited = state.get("edited_segments", set())
            edited.add(segment_idx)
            state["edited_segments"] = edited

    # Rebuild outline HTML
    outline_html = build_outline_html(state)
    status = f"Saved changes to segment {segment_idx + 1}: {new_title[:30]}..."

    return outline_html, state, status


def on_edit_dialogue(
    dialogue_idx: int,
    state: dict[str, Any],
) -> tuple[gr.update, gr.update, gr.update, gr.update]:
    """Open dialogue edit modal."""
    segment_idx = state.get("selected_segment", 0)
    dialogues = get_segment_dialogues(state, segment_idx)

    if 0 <= dialogue_idx < len(dialogues):
        dialogue = dialogues[dialogue_idx]
        return (
            gr.update(value=dialogue.speaker, visible=True),
            gr.update(value=dialogue.text, visible=True),
            gr.update(visible=True),
            gr.update(value=dialogue_idx),
        )

    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(),
    )


def on_save_dialogue_edit(
    new_speaker: str,
    new_text: str,
    editing_idx: int,
    state: dict[str, Any],
) -> tuple[str, dict[str, Any], str, gr.update, gr.update, gr.update]:
    """Save dialogue edits."""
    state = copy.deepcopy(state)
    segment_idx = state.get("selected_segment", 0)
    transcripts = state.get("transcripts", {})

    if segment_idx in transcripts:
        dialogues = list(transcripts[segment_idx].dialogues)
        if 0 <= editing_idx < len(dialogues):
            old_dialogue = dialogues[editing_idx]
            if new_speaker.strip() != old_dialogue.speaker or new_text.strip() != old_dialogue.text:
                # Update dialogue
                dialogues[editing_idx] = Dialogue(
                    speaker=new_speaker.strip(),
                    text=new_text.strip(),
                )
                transcripts[segment_idx] = Transcript(dialogues=dialogues)
                state["transcripts"] = transcripts

                # Mark as edited
                edited_dialogues = state.get("edited_dialogues", {})
                if segment_idx not in edited_dialogues:
                    edited_dialogues[segment_idx] = set()
                edited_dialogues[segment_idx].add(editing_idx)
                state["edited_dialogues"] = edited_dialogues

    # Refresh dialogues display
    dialogues = get_segment_dialogues(state, segment_idx)
    edited_indices = state.get("edited_dialogues", {}).get(segment_idx, set())
    dialogues_html = render_dialogues_html(dialogues, edited_indices)

    status = f"Saved dialogue {editing_idx + 1}: {new_speaker} - {new_text[:30]}..."

    # Hide edit fields
    return (
        dialogues_html,
        state,
        status,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def build_outline_html(state: dict[str, Any]) -> str:
    """Build complete outline HTML with segments."""
    outline = state.get("outline")
    edited_segments = state.get("edited_segments", set())

    if not outline or not outline.segments:
        return """
        <div class="empty-state">
            <div class="empty-state-icon">📋</div>
            <div class="empty-state-text">No outline segments available</div>
        </div>
        """

    html_parts = ['<div class="outline-scroll">']
    for idx, segment in enumerate(outline.segments):
        is_edited = idx in edited_segments
        edited_class = "edited" if is_edited else ""
        edited_badge = '<span class="edit-indicator">EDITED</span>' if is_edited else ""

        html_parts.append(f"""
        <div class="segment-accordion {edited_class}">
            <div class="label-wrap" data-segment="{idx}">
                <strong>{idx + 1}. {segment.title}</strong>
                {get_size_badge_html(segment.size)}
                {edited_badge}
            </div>
            <div class="segment-content">
                <p>{segment.description}</p>
            </div>
        </div>
        """)

    html_parts.append("</div>")
    return "".join(html_parts)


def get_edit_count(state: dict[str, Any]) -> str:
    """Get count of edited items for status display."""
    edited_segments = len(state.get("edited_segments", set()))
    edited_dialogues = sum(
        len(indices) for indices in state.get("edited_dialogues", {}).values()
    )
    return f"Edited: {edited_segments} segments, {edited_dialogues} dialogue lines"


# =============================================================================
# MAIN UI
# =============================================================================


def create_draft_preview_ui() -> gr.Blocks:
    """Create the draft preview Gradio interface."""
    initial_state = initialize_state()
    initial_outline_html = build_outline_html(initial_state)
    initial_dialogues_html = render_dialogues_html(
        get_segment_dialogues(initial_state, 0), set()
    )

    with gr.Blocks(css=DRAFT_PREVIEW_CSS, title="Draft Preview") as demo:
        # State
        app_state = gr.State(value=initial_state)
        editing_dialogue_idx = gr.State(value=-1)

        # Header
        gr.HTML("""
        <div class="draft-header">
            <h1 class="draft-title">Draft Preview</h1>
            <p class="draft-subtitle">Review and edit your podcast outline and dialogues</p>
        </div>
        """)

        with gr.Row():
            # LEFT COLUMN: Outline Tree
            with gr.Column(scale=2):
                gr.HTML('<div class="panel-header">Outline Segments</div>')

                outline_html = gr.HTML(value=initial_outline_html)

                # Segment selection buttons
                gr.HTML('<div style="margin-top: 1rem; font-size: 0.85rem; color: var(--slate-500);">Click a segment to edit:</div>')
                with gr.Row():
                    seg_btns = []
                    for i in range(5):
                        btn = gr.Button(
                            f"Segment {i + 1}",
                            size="sm",
                            elem_classes=["edit-btn"],
                        )
                        seg_btns.append(btn)

                # Segment edit section
                gr.HTML('<div class="panel-header" style="margin-top: 1.5rem;">Edit Segment</div>')
                segment_title_input = gr.Textbox(
                    label="Title",
                    visible=True,
                    value=initial_state["outline"].segments[0].title,
                )
                segment_desc_input = gr.Textbox(
                    label="Description",
                    lines=3,
                    visible=True,
                    value=initial_state["outline"].segments[0].description,
                )
                save_segment_btn = gr.Button(
                    "Save Segment Changes",
                    elem_classes=["edit-btn", "save-btn"],
                    visible=True,
                )

            # RIGHT COLUMN: Dialogue Editor
            with gr.Column(scale=3):
                gr.HTML('<div class="panel-header">Dialogue Editor</div>')

                with gr.Column(elem_classes=["dialogue-panel"]):
                    dialogues_html = gr.HTML(value=initial_dialogues_html)

                    # Dialogue selection
                    gr.HTML('<div style="margin-top: 1rem; font-size: 0.85rem; color: var(--slate-500);">Select dialogue to edit:</div>')
                    dialogue_idx_slider = gr.Slider(
                        minimum=0,
                        maximum=7,
                        step=1,
                        value=0,
                        label="Dialogue Line",
                    )
                    edit_dialogue_btn = gr.Button(
                        "Edit Selected Dialogue",
                        elem_classes=["edit-btn"],
                    )

                    # Dialogue edit fields
                    gr.HTML('<div style="margin-top: 1rem;"></div>')
                    dialogue_speaker_input = gr.Textbox(
                        label="Speaker",
                        visible=False,
                    )
                    dialogue_text_input = gr.Textbox(
                        label="Dialogue Text",
                        lines=3,
                        visible=False,
                    )
                    save_dialogue_btn = gr.Button(
                        "Save Dialogue",
                        elem_classes=["edit-btn", "save-btn"],
                        visible=False,
                    )

        # Status bar
        status_text = gr.Textbox(
            value=get_edit_count(initial_state),
            label="Status",
            interactive=False,
            elem_classes=["status-bar"],
        )

        # Event handlers for segment buttons
        for i, btn in enumerate(seg_btns):
            btn.click(
                fn=lambda idx=i: on_segment_select(idx, initial_state),
                inputs=[app_state],
                outputs=[
                    dialogues_html,
                    app_state,
                    segment_title_input,
                    segment_desc_input,
                    save_segment_btn,
                ],
            ).then(
                fn=lambda s: on_segment_select(s.get("selected_segment", 0), s),
                inputs=[app_state],
                outputs=[
                    dialogues_html,
                    app_state,
                    segment_title_input,
                    segment_desc_input,
                    save_segment_btn,
                ],
            )

        # Segment selection with proper state update
        def select_segment(idx: int, state: dict) -> tuple:
            return on_segment_select(idx, state)

        for i, btn in enumerate(seg_btns):
            btn.click(
                fn=select_segment,
                inputs=[gr.Number(value=i, visible=False), app_state],
                outputs=[
                    dialogues_html,
                    app_state,
                    segment_title_input,
                    segment_desc_input,
                    save_segment_btn,
                ],
            )

        # Save segment changes
        save_segment_btn.click(
            fn=on_save_segment_edit,
            inputs=[segment_title_input, segment_desc_input, app_state],
            outputs=[outline_html, app_state, status_text],
        )

        # Edit dialogue button
        edit_dialogue_btn.click(
            fn=on_edit_dialogue,
            inputs=[dialogue_idx_slider, app_state],
            outputs=[
                dialogue_speaker_input,
                dialogue_text_input,
                save_dialogue_btn,
                editing_dialogue_idx,
            ],
        )

        # Save dialogue
        save_dialogue_btn.click(
            fn=on_save_dialogue_edit,
            inputs=[
                dialogue_speaker_input,
                dialogue_text_input,
                editing_dialogue_idx,
                app_state,
            ],
            outputs=[
                dialogues_html,
                app_state,
                status_text,
                dialogue_speaker_input,
                dialogue_text_input,
                save_dialogue_btn,
            ],
        )

    return demo


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    demo = create_draft_preview_ui()
    print("Starting Draft Preview UI...")
    print("Open http://localhost:7861 in your browser")
    demo.launch(server_port=7861, share=False)

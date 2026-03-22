#!/usr/bin/env python3
"""
Podcast Content Input UI Component

Gradio-based UI for capturing podcast topic, key points, and style briefing.
Can be used standalone or integrated into larger applications.
"""

import gradio as gr
from typing import Dict, Optional, Tuple

MAX_TOPIC_CHARS = 10000
TOPIC_WARNING_THRESHOLD = 8000


def update_topic_char_count(text: str) -> str:
    """Update character counter for topic field with warning states."""
    count = len(text)
    if count > MAX_TOPIC_CHARS:
        return f'<span class="char-count char-error">{count:,} / {MAX_TOPIC_CHARS:,} characters (too long)</span>'
    elif count > TOPIC_WARNING_THRESHOLD:
        return f'<span class="char-count char-warning">{count:,} / {MAX_TOPIC_CHARS:,} characters</span>'
    else:
        return f'<span class="char-count">{count:,} / {MAX_TOPIC_CHARS:,} characters</span>'


def validate_content(topic: str, key_points: str, briefing: str) -> Tuple[bool, str]:
    """
    Validate podcast content inputs.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not topic or not topic.strip():
        return False, "Topic is required. Please enter a podcast topic."
    
    if len(topic) > MAX_TOPIC_CHARS:
        return False, f"Topic exceeds {MAX_TOPIC_CHARS:,} character limit ({len(topic):,} chars)."
    
    return True, ""


def get_content_dict(topic: str, key_points: str, briefing: str) -> Dict[str, str]:
    """Package content inputs into a dictionary."""
    return {
        "topic": topic.strip() if topic else "",
        "key_points": key_points.strip() if key_points else "",
        "briefing": briefing.strip() if briefing else ""
    }


def submit_content(topic: str, key_points: str, briefing: str) -> Tuple[str, Dict[str, str]]:
    """
    Validate and submit podcast content.
    
    Returns:
        Tuple of (status_message, content_dict)
    """
    is_valid, error_msg = validate_content(topic, key_points, briefing)
    
    if not is_valid:
        return f"<span style='color: #dc3545;'>Error: {error_msg}</span>", {}
    
    content = get_content_dict(topic, key_points, briefing)
    return "<span style='color: #28a745;'>Content validated successfully!</span>", content


custom_css = """
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

.gradio-container {
    max-width: 900px !important;
    margin: 0 auto;
    background: var(--gray-50) !important;
}

.main-header {
    text-align: center;
    padding: 1.25rem 0;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--gray-200);
}

.main-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--gray-800);
    margin: 0;
    letter-spacing: -0.01em;
}

.sub-title {
    color: var(--gray-600);
    font-size: 0.875rem;
    margin: 0.25rem 0 0;
}

.section-header {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--gray-700);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--gray-200);
}

.char-count {
    font-size: 0.75rem;
    color: var(--gray-600);
    padding: 0.25rem 0.5rem;
    background: var(--gray-100);
    border-radius: var(--radius);
    display: inline-block;
    margin-top: 0.25rem;
}

.char-count.char-warning {
    color: #856404;
    background: #fff3cd;
    font-weight: 500;
}

.char-count.char-error {
    color: #721c24;
    background: #f8d7da;
    font-weight: 600;
}

.submit-btn {
    min-height: 44px !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    border-radius: var(--radius) !important;
    background: var(--gray-800) !important;
    border: none !important;
    color: var(--white) !important;
}

.submit-btn:hover {
    background: var(--gray-900) !important;
}

.content-panel {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius);
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.optional-label {
    color: var(--gray-500);
    font-size: 0.75rem;
    font-style: italic;
}
"""


def create_content_input_ui() -> gr.Blocks:
    """
    Create the podcast content input UI component.
    
    Returns:
        gr.Blocks: The Gradio Blocks interface
    """
    with gr.Blocks(css=custom_css, title="Podcast Content Input") as demo:
        gr.HTML("""
        <div class="main-header">
            <h1 class="main-title">Podcast Content Input</h1>
            <p class="sub-title">Define your podcast topic and content</p>
        </div>
        """)
        
        content_state = gr.State({})
        
        with gr.Column(elem_classes=["content-panel"]):
            gr.HTML('<div class="section-header">Topic</div>')
            
            topic_input = gr.Textbox(
                label="Podcast Topic",
                placeholder="Enter your podcast topic or main subject...\n\nExample: The future of artificial intelligence and its impact on creative industries, exploring how AI tools are transforming music production, visual arts, and content creation.",
                lines=3,
                max_lines=3,
                info="Required. What is your podcast about?"
            )
            topic_char_count = gr.HTML(value=update_topic_char_count(""))
        
        with gr.Column(elem_classes=["content-panel"]):
            gr.HTML('<div class="section-header">Supporting Content <span class="optional-label">(Optional)</span></div>')
            
            key_points_input = gr.Textbox(
                label="Key Points (Optional)",
                placeholder="List the main points you want to cover...\n\nExample:\n- History of AI in creative fields\n- Current tools and technologies\n- Case studies of successful AI-human collaboration\n- Ethical considerations and future outlook",
                lines=5,
                info="Optional. Bullet points or key topics to discuss"
            )
            
            style_briefing_input = gr.Textbox(
                label="Style & Tone (Optional)",
                placeholder="Describe the desired style and tone...\n\nExample: Conversational and engaging, with a balance of technical depth and accessibility for general audiences.",
                lines=2,
                info="Optional. How should the podcast sound?"
            )
        
        with gr.Row():
            submit_btn = gr.Button(
                "Validate Content",
                variant="primary",
                elem_classes=["submit-btn"],
                size="lg"
            )
        
        status_output = gr.HTML(value="")
        
        content_output = gr.JSON(
            label="Content Data",
            visible=True
        )
        
        topic_input.change(
            fn=update_topic_char_count,
            inputs=[topic_input],
            outputs=[topic_char_count]
        )
        
        submit_btn.click(
            fn=submit_content,
            inputs=[topic_input, key_points_input, style_briefing_input],
            outputs=[status_output, content_output]
        )
    
    return demo


def get_content_components():
    """
    Create content input components for integration into other UIs.
    
    Returns:
        Tuple of (topic, key_points, briefing, char_counter) components
    """
    topic_input = gr.Textbox(
        label="Podcast Topic",
        placeholder="Enter your podcast topic or main subject...",
        lines=3,
        max_lines=3,
        info="Required. What is your podcast about?"
    )
    
    topic_char_count = gr.HTML(value=update_topic_char_count(""))
    
    key_points_input = gr.Textbox(
        label="Key Points (Optional)",
        placeholder="List the main points you want to cover...",
        lines=5,
        info="Optional. Bullet points or key topics to discuss"
    )
    
    style_briefing_input = gr.Textbox(
        label="Style & Tone (Optional)",
        placeholder="Describe the desired style and tone...",
        lines=2,
        info="Optional. How should the podcast sound?"
    )
    
    return topic_input, key_points_input, style_briefing_input, topic_char_count


if __name__ == "__main__":
    demo = create_content_input_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True
    )

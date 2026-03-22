"""Prompt templates for podcast outline and transcript generation."""

import json
from typing import Optional


def get_outline_prompt(
    topic: str,
    key_points: list[str],
    briefing: str,
    num_segments: int = 5,
    speakers: Optional[list[str]] = None,
) -> str:
    """
    Generate a prompt for creating a podcast outline.

    Args:
        topic: Main topic of the podcast episode.
        key_points: List of key points to cover.
        briefing: Background information or context.
        num_segments: Number of segments to create (default: 5).
        speakers: List of speaker names participating (optional).

    Returns:
        A formatted prompt string for the LLM.
    """
    speakers_list = speakers or ["Host", "Expert"]
    speakers_str = ", ".join(speakers_list)

    outline_schema = {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Segment title"},
                        "description": {
                            "type": "string",
                            "description": "Segment summary or notes",
                        },
                        "size": {
                            "type": "string",
                            "enum": ["short", "medium", "long"],
                            "description": "Segment length",
                        },
                    },
                    "required": ["title", "description", "size"],
                },
                "minItems": 1,
                "maxItems": 10,
            }
        },
        "required": ["segments"],
    }

    examples = """
EXAMPLE 1: Tech Interview Outline
Topic: "AI in Healthcare"
Key Points: ["Diagnosis accuracy", "Patient privacy", "Cost reduction"]
Speakers: ["Host", "Dr. Smith (Expert)"]

Output:
{
  "segments": [
    {
      "title": "Introduction & Guest Welcome",
      "description": "Host introduces Dr. Smith and the topic of AI in healthcare. Brief overview of what listeners will learn.",
      "size": "short"
    },
    {
      "title": "Current State of AI in Diagnosis",
      "description": "Dr. Smith explains how AI is currently being used for medical diagnosis, with real-world examples.",
      "size": "medium"
    },
    {
      "title": "Privacy & Ethical Concerns",
      "description": "Discussion of patient data privacy, regulatory compliance, and ethical considerations.",
      "size": "medium"
    },
    {
      "title": "Cost Impact & Accessibility",
      "description": "How AI can reduce healthcare costs and improve accessibility in underserved areas.",
      "size": "medium"
    },
    {
      "title": "Future Outlook & Q&A",
      "description": "Dr. Smith shares predictions for the next 5 years. Host asks audience questions.",
      "size": "short"
    }
  ]
}

EXAMPLE 2: Panel Discussion Outline
Topic: "Remote Work Trends"
Key Points: ["Productivity", "Work-life balance", "Team collaboration"]
Speakers: ["Host", "Manager", "Employee", "HR Lead"]

Output:
{
  "segments": [
    {
      "title": "Opening & Panel Introduction",
      "description": "Host welcomes panelists and sets context for remote work discussion.",
      "size": "short"
    },
    {
      "title": "Productivity Metrics & Results",
      "description": "Manager shares data on productivity changes. Employee perspective on focus and distractions.",
      "size": "medium"
    },
    {
      "title": "Work-Life Balance Challenges",
      "description": "Panelists discuss boundary-setting, burnout prevention, and mental health.",
      "size": "medium"
    },
    {
      "title": "Team Collaboration & Culture",
      "description": "HR Lead discusses maintaining company culture. Panelists share collaboration tools and practices.",
      "size": "medium"
    },
    {
      "title": "Closing Remarks & Takeaways",
      "description": "Each panelist shares one key takeaway. Host summarizes main points.",
      "size": "short"
    }
  ]
}
"""

    prompt = f"""You are an expert podcast producer creating a structured outline for an engaging podcast episode.

TOPIC: {topic}

KEY POINTS TO COVER:
{chr(10).join(f"- {point}" for point in key_points)}

BACKGROUND & BRIEFING:
{briefing}

SPEAKERS PARTICIPATING:
{speakers_str}

REQUIREMENTS:
1. Create exactly {num_segments} segments for this podcast episode.
2. Each segment should have a clear title, description, and size (short/medium/long).
3. Segments should flow logically and build on each other.
4. Include all key points across the segments.
5. Vary segment sizes for pacing (don't make all segments the same length).
6. Ensure the outline is engaging and maintains listener interest.
7. Consider the speaker roles when planning dialogue flow.

SPEAKER ROLE GUIDELINES:
- Host: Asks questions, guides conversation, keeps time, welcomes guests
- Expert: Provides deep knowledge, explains concepts, answers questions
- Guest: Shares personal experience, offers perspective, responds to host
- Narrator: Provides context, transitions, background information

OUTPUT FORMAT:
Return ONLY valid JSON matching this schema:
{json.dumps(outline_schema, indent=2)}

EXAMPLES OF GOOD OUTLINES:
{examples}

Now create the outline for this episode:"""

    return prompt


def get_transcript_prompt(
    outline: str,
    segment: str,
    briefing: str,
    speakers: list[str],
    is_final: bool = False,
    turns: int = 8,
    language: str = "English",
) -> str:
    dialogue_schema = {
        "type": "object",
        "properties": {
            "dialogues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {
                            "type": "string",
                            "description": "Name of the speaker",
                        },
                        "text": {
                            "type": "string",
                            "description": "Spoken dialogue (natural, conversational)",
                        },
                    },
                    "required": ["speaker", "text"],
                },
                "minItems": 2,
                "maxItems": 50,
            }
        },
        "required": ["dialogues"],
    }

    speaker_roles = {
        "Host": "Asks questions, guides conversation, keeps things on track, shows enthusiasm",
        "Expert": "Provides detailed explanations, shares knowledge, answers questions thoroughly",
        "Guest": "Shares personal experience, offers perspective, responds naturally to questions",
        "Narrator": "Provides context and transitions between topics",
    }

    examples = """
EXAMPLE 1: Expert Interview Segment
Segment: "Current State of AI in Diagnosis"
Speakers: ["Host", "Dr. Smith (Expert)"]

Output:
{
  "dialogues": [
    {
      "speaker": "Host",
      "text": "Dr. Smith, let's dive into how AI is actually being used in medical diagnosis today. Can you give us some concrete examples?"
    },
    {
      "speaker": "Dr. Smith",
      "text": "Absolutely. One of the most successful applications is in radiology. AI systems can now detect certain cancers in X-rays and CT scans with accuracy rates that match or exceed human radiologists."
    },
    {
      "speaker": "Host",
      "text": "That's impressive. Are there other areas where you're seeing significant impact?"
    },
    {
      "speaker": "Dr. Smith",
      "text": "Yes, pathology is another big one. AI can analyze tissue samples to identify diseases like cancer much faster than traditional methods. We're also seeing promising results in cardiology and ophthalmology."
    },
    {
      "speaker": "Host",
      "text": "How long does it typically take to implement these systems in a hospital?"
    },
    {
      "speaker": "Dr. Smith",
      "text": "That varies, but usually between 6 to 18 months. It depends on the complexity of the system and how well it integrates with existing infrastructure."
    }
  ]
}

EXAMPLE 2: Panel Discussion Segment
Segment: "Work-Life Balance Challenges"
Speakers: ["Host", "Manager", "Employee"]

Output:
{
  "dialogues": [
    {
      "speaker": "Host",
      "text": "One thing we hear a lot about with remote work is the challenge of maintaining work-life balance. Manager, from your perspective, how are you seeing this play out?"
    },
    {
      "speaker": "Manager",
      "text": "It's definitely a concern. Without the physical separation of an office, people tend to work longer hours. We've had to be intentional about encouraging breaks and respecting off-hours."
    },
    {
      "speaker": "Employee",
      "text": "I can relate to that. At first, I found myself working until 8 or 9 PM because my home office was just steps away. I had to set hard boundaries."
    },
    {
      "speaker": "Host",
      "text": "What kind of boundaries did you set?"
    },
    {
      "speaker": "Employee",
      "text": "I created a ritual where I literally close my laptop and put it away at 5 PM. I also changed my Slack status to 'offline' so people know not to expect immediate responses."
    },
    {
      "speaker": "Manager",
      "text": "That's smart. We've also implemented 'no meeting Fridays' in the afternoon to give people time to catch up and decompress."
    }
  ]
}
"""

    segment_tone = "concluding remarks and key takeaways" if is_final else "engaging discussion"

    prompt = f"""You are an expert podcast scriptwriter creating natural, engaging dialogue for a podcast segment.

IMPORTANT: Generate all dialogue in {language}. The entire transcript must be written in {language}.

PODCAST OUTLINE (for context):
{outline}

CURRENT SEGMENT TO WRITE:
{segment}

BACKGROUND & BRIEFING:
{briefing}

SPEAKERS:
{", ".join(speakers)}

SPEAKER ROLE GUIDELINES:
{chr(10).join(f"- {name}: {role}" for name, role in speaker_roles.items() if name in speakers)}

REQUIREMENTS:
1. Generate approximately {turns} dialogue turns (back-and-forth exchanges).
2. Write ALL dialogue in {language} - this is mandatory.
3. Make dialogue natural and conversational, not scripted or robotic.
4. Include follow-up questions from the Host to keep conversation flowing.
5. Vary sentence length and structure for natural pacing.
6. Use contractions and natural speech patterns appropriate for {language}.
7. Avoid jargon unless the Expert is explaining it.
8. Include natural transitions and acknowledgments appropriate for {language}.
9. {"Focus on concluding remarks, key takeaways, and wrapping up the discussion." if is_final else "Keep the discussion engaging and exploratory."}
10. Ensure all speakers get roughly equal speaking time.
11. Make the dialogue sound like a real conversation, not a Q&A interview.

OUTPUT FORMAT:
Return ONLY valid JSON matching this schema:
{json.dumps(dialogue_schema, indent=2)}

EXAMPLES OF GOOD DIALOGUE:
{examples}

Now create the transcript for this segment:"""

    return prompt


if __name__ == "__main__":
    # Test outline prompt rendering
    outline_prompt = get_outline_prompt(
        topic="The Future of Remote Work",
        key_points=[
            "Productivity trends",
            "Employee satisfaction",
            "Company culture challenges",
            "Technology infrastructure",
        ],
        briefing="Post-pandemic analysis of remote work adoption and its long-term impact on businesses.",
        num_segments=5,
        speakers=["Host", "HR Manager", "Tech Lead", "Remote Employee"],
    )

    print("=" * 80)
    print("OUTLINE PROMPT TEST")
    print("=" * 80)
    print(outline_prompt[:500] + "...\n")
    assert "The Future of Remote Work" in outline_prompt
    assert "5" in outline_prompt
    assert "segments" in outline_prompt
    assert "short" in outline_prompt
    print("✓ Outline prompt renders correctly\n")

    # Test transcript prompt rendering
    transcript_prompt = get_transcript_prompt(
        outline="1. Introduction\n2. Main Discussion\n3. Closing",
        segment="Main Discussion: Productivity Trends",
        briefing="Focus on measurable productivity metrics and employee feedback.",
        speakers=["Host", "HR Manager"],
        is_final=False,
        turns=6,
    )

    print("=" * 80)
    print("TRANSCRIPT PROMPT TEST")
    print("=" * 80)
    print(transcript_prompt[:500] + "...\n")
    assert "Main Discussion" in transcript_prompt
    assert "dialogues" in transcript_prompt
    assert "Host" in transcript_prompt
    assert "HR Manager" in transcript_prompt
    print("✓ Transcript prompt renders correctly\n")

    # Test final segment variant
    final_transcript_prompt = get_transcript_prompt(
        outline="1. Introduction\n2. Main Discussion\n3. Closing",
        segment="Closing: Key Takeaways",
        briefing="Summarize main points and provide actionable insights.",
        speakers=["Host", "HR Manager"],
        is_final=True,
        turns=4,
    )

    print("=" * 80)
    print("FINAL SEGMENT TRANSCRIPT PROMPT TEST")
    print("=" * 80)
    print(final_transcript_prompt[:500] + "...\n")
    assert "concluding remarks" in final_transcript_prompt
    assert "Key Takeaways" in final_transcript_prompt
    print("✓ Final segment prompt renders correctly\n")

    print("=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)

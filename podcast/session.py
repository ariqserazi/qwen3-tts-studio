"""Podcast session state management for step-by-step workflow."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class StepStatus(str, Enum):
    EMPTY = "empty"
    GENERATING = "generating"
    READY = "ready"
    EDITED = "edited"
    STALE = "stale"
    ERROR = "error"


@dataclass
class VoiceSelection:
    voice_id: str
    name: str
    role: str
    voice_type: str
    
    def to_dict(self) -> dict[str, str]:
        return {
            "voice_id": self.voice_id,
            "name": self.name,
            "role": self.role,
            "type": self.voice_type,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, str]) -> VoiceSelection:
        return cls(
            voice_id=data.get("voice_id", ""),
            name=data.get("name", ""),
            role=data.get("role", ""),
            voice_type=data.get("type", "preset"),
        )


@dataclass
class PodcastSessionState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    artifacts_dir: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    topic: str = ""
    key_points: str = ""
    briefing: str = ""
    num_segments: int = 2
    language: str = "English"
    quality_preset: str = "standard"
    
    voices: dict[str, dict[str, str]] = field(default_factory=dict)
    
    outline_text: str = ""
    outline_struct: list[dict[str, str]] = field(default_factory=list)
    
    transcript_text: str = ""
    transcript_struct: list[dict[str, str]] = field(default_factory=list)
    
    audio_clips: list[str] = field(default_factory=list)
    combined_audio_path: str = ""
    
    outline_status: str = StepStatus.EMPTY.value
    transcript_status: str = StepStatus.EMPTY.value
    audio_status: str = StepStatus.EMPTY.value
    
    outline_fingerprint: str = ""
    transcript_fingerprint: str = ""
    voices_fingerprint: str = ""
    
    last_error: str = ""
    
    def touch(self) -> None:
        self.updated_at = datetime.now().isoformat()
    
    def compute_outline_fingerprint(self) -> str:
        return hashlib.md5(self.outline_text.encode()).hexdigest()[:12]
    
    def compute_transcript_fingerprint(self) -> str:
        return hashlib.md5(self.transcript_text.encode()).hexdigest()[:12]
    
    def compute_voices_fingerprint(self) -> str:
        voices_str = json.dumps(self.voices, sort_keys=True)
        return hashlib.md5(voices_str.encode()).hexdigest()[:12]
    
    def mark_outline_ready(self) -> None:
        self.outline_status = StepStatus.READY.value
        self.outline_fingerprint = self.compute_outline_fingerprint()
        self.invalidate_downstream_from_outline()
        self.touch()
    
    def mark_outline_edited(self) -> None:
        self.outline_status = StepStatus.EDITED.value
        new_fp = self.compute_outline_fingerprint()
        if new_fp != self.outline_fingerprint:
            self.outline_fingerprint = new_fp
            self.invalidate_downstream_from_outline()
        self.touch()
    
    def mark_transcript_ready(self) -> None:
        self.transcript_status = StepStatus.READY.value
        self.transcript_fingerprint = self.compute_transcript_fingerprint()
        self.invalidate_downstream_from_transcript()
        self.touch()
    
    def mark_transcript_edited(self) -> None:
        self.transcript_status = StepStatus.EDITED.value
        new_fp = self.compute_transcript_fingerprint()
        if new_fp != self.transcript_fingerprint:
            self.transcript_fingerprint = new_fp
            self.invalidate_downstream_from_transcript()
        self.touch()
    
    def mark_audio_ready(self) -> None:
        self.audio_status = StepStatus.READY.value
        self.touch()
    
    def mark_voices_changed(self) -> None:
        new_fp = self.compute_voices_fingerprint()
        if new_fp != self.voices_fingerprint:
            self.voices_fingerprint = new_fp
            if self.audio_status in (StepStatus.READY.value, StepStatus.EDITED.value):
                self.audio_status = StepStatus.STALE.value
        self.touch()
    
    def invalidate_downstream_from_outline(self) -> None:
        if self.transcript_status in (StepStatus.READY.value, StepStatus.EDITED.value):
            self.transcript_status = StepStatus.STALE.value
        if self.audio_status in (StepStatus.READY.value, StepStatus.EDITED.value):
            self.audio_status = StepStatus.STALE.value
    
    def invalidate_downstream_from_transcript(self) -> None:
        if self.audio_status in (StepStatus.READY.value, StepStatus.EDITED.value):
            self.audio_status = StepStatus.STALE.value
    
    def can_generate_transcript(self) -> bool:
        return self.outline_status in (StepStatus.READY.value, StepStatus.EDITED.value)
    
    def can_generate_audio(self) -> bool:
        return self.transcript_status in (StepStatus.READY.value, StepStatus.EDITED.value)
    
    def is_outline_stale(self) -> bool:
        return False
    
    def is_transcript_stale(self) -> bool:
        return self.transcript_status == StepStatus.STALE.value
    
    def is_audio_stale(self) -> bool:
        return self.audio_status == StepStatus.STALE.value
    
    def reset(self) -> None:
        self.outline_text = ""
        self.outline_struct = []
        self.transcript_text = ""
        self.transcript_struct = []
        self.audio_clips = []
        self.combined_audio_path = ""
        self.outline_status = StepStatus.EMPTY.value
        self.transcript_status = StepStatus.EMPTY.value
        self.audio_status = StepStatus.EMPTY.value
        self.outline_fingerprint = ""
        self.transcript_fingerprint = ""
        self.last_error = ""
        self.touch()
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PodcastSessionState:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def save(self, path: Path | None = None) -> Path:
        if path is None:
            if not self.artifacts_dir:
                raise ValueError("No artifacts_dir set and no path provided")
            path = Path(self.artifacts_dir) / "session_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path
    
    @classmethod
    def load(cls, path: Path) -> PodcastSessionState:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def parse_outline_text(text: str) -> list[dict[str, str]]:
    segments = []
    pattern = r'(\d+)\.\s*(.+?)(?:\n|$)([\s\S]*?)(?=\n\d+\.|$)'
    matches = re.findall(pattern, text.strip())
    
    for match in matches:
        num, title, description = match
        segments.append({
            "title": title.strip(),
            "description": description.strip(),
            "size": "medium",
        })
    
    if not segments and text.strip():
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        for i, line in enumerate(lines):
            clean_line = re.sub(r'^\d+\.\s*', '', line)
            segments.append({
                "title": clean_line,
                "description": "",
                "size": "medium",
            })
    
    return segments


def format_outline_text(segments: list[dict[str, str]]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        title = seg.get("title", f"Segment {i}")
        desc = seg.get("description", "")
        lines.append(f"{i}. {title}")
        if desc:
            lines.append(desc)
        lines.append("")
    return "\n".join(lines).strip()


def parse_transcript_text(text: str) -> list[dict[str, str]]:
    dialogues = []
    pattern = r'^([A-Z][A-Za-z_\-\s]*?):\s*(.+?)(?=\n[A-Z][A-Za-z_\-\s]*?:|$)'
    matches = re.findall(pattern, text.strip(), re.MULTILINE | re.DOTALL)
    
    for speaker, content in matches:
        dialogues.append({
            "speaker": speaker.strip().lower(),
            "text": content.strip(),
        })
    
    if not dialogues and text.strip():
        dialogues.append({
            "speaker": "narrator",
            "text": text.strip(),
        })
    
    return dialogues


def format_transcript_text(dialogues: list[dict[str, str]]) -> str:
    lines = []
    for dlg in dialogues:
        speaker = dlg.get("speaker", "Speaker").upper()
        text = dlg.get("text", "")
        lines.append(f"{speaker}: {text}")
        lines.append("")
    return "\n".join(lines).strip()


def get_step_status_display(status: str) -> tuple[str, str]:
    status_map = {
        StepStatus.EMPTY.value: ("○", "Not started"),
        StepStatus.GENERATING.value: ("◐", "Generating..."),
        StepStatus.READY.value: ("●", "Ready"),
        StepStatus.EDITED.value: ("✎", "Edited"),
        StepStatus.STALE.value: ("⚠", "Needs regeneration"),
        StepStatus.ERROR.value: ("✗", "Error"),
    }
    return status_map.get(status, ("?", "Unknown"))


def create_step_indicator_html(
    outline_status: str,
    transcript_status: str,
    audio_status: str,
) -> str:
    def step_html(name: str, status: str, step_num: int) -> str:
        icon, label = get_step_status_display(status)
        
        color_map = {
            StepStatus.EMPTY.value: "#888",
            StepStatus.GENERATING.value: "#2196F3",
            StepStatus.READY.value: "#4CAF50",
            StepStatus.EDITED.value: "#9C27B0",
            StepStatus.STALE.value: "#FF9800",
            StepStatus.ERROR.value: "#f44336",
        }
        color = color_map.get(status, "#888")
        
        return f'''
        <div style="display: flex; flex-direction: column; align-items: center; min-width: 80px;">
            <div style="
                width: 32px; height: 32px; 
                border-radius: 50%; 
                background: {color}; 
                color: white; 
                display: flex; 
                align-items: center; 
                justify-content: center;
                font-size: 16px;
                font-weight: bold;
            ">{icon if status != StepStatus.EMPTY.value else step_num}</div>
            <div style="margin-top: 4px; font-size: 12px; font-weight: 600;">{name}</div>
            <div style="font-size: 10px; color: {color};">{label}</div>
        </div>
        '''
    
    connector = '''
    <div style="
        flex: 1; 
        height: 2px; 
        background: linear-gradient(to right, #ddd, #ddd);
        margin: 0 8px;
        margin-top: -20px;
    "></div>
    '''
    
    return f'''
    <div style="display: flex; align-items: flex-start; justify-content: center; padding: 16px 0;">
        {step_html("Outline", outline_status, 1)}
        {connector}
        {step_html("Transcript", transcript_status, 2)}
        {connector}
        {step_html("Audio", audio_status, 3)}
    </div>
    '''

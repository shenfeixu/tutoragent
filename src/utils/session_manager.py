"""Session management for conversation history."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "sessions"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_session_files() -> List[Path]:
    """Get all session files sorted by modification time (newest first)."""
    files = list(DATA_DIR.glob("*.json"))
    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def session_to_dict(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert session data to serializable dict."""
    result = {
        "session_id": session_data.get("session_id"),
        "title": session_data.get("title", "新对话"),
        "created_at": session_data.get("created_at"),
        "updated_at": session_data.get("updated_at"),
        "messages": [],
        "accumulated_info": session_data.get("accumulated_info", {}),
    }
    for msg in session_data.get("messages", []):
        msg_dict = {
            "role": msg.get("role"),
            "content": msg.get("content"),
            "timestamp": msg.get("timestamp"),
        }
        if msg.get("state"):
            state = msg["state"]
            if hasattr(state, "model_dump"):
                msg_dict["state"] = state.model_dump()
            elif hasattr(state, "dict"):
                msg_dict["state"] = state.dict()
            else:
                msg_dict["state"] = state
        result["messages"].append(msg_dict)
    return result


def save_session(session_data: Dict[str, Any]) -> str:
    """Save session to file."""
    session_id = session_data.get("session_id")
    if not session_id:
        session_id = generate_session_id()
        session_data["session_id"] = session_id
    
    if not session_data.get("created_at"):
        session_data["created_at"] = datetime.now().isoformat()
    session_data["updated_at"] = datetime.now().isoformat()
    
    if session_data.get("messages"):
        first_msg = session_data["messages"][0]
        content = first_msg.get("content", "")
        if content:
            session_data["title"] = content[:30] + ("..." if len(content) > 30 else "")
    
    file_path = DATA_DIR / f"{session_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(session_to_dict(session_data), f, ensure_ascii=False, indent=2)
    
    return session_id


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Load session from file."""
    file_path = DATA_DIR / f"{session_id}.json"
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_session(session_id: str) -> bool:
    """Delete session file."""
    file_path = DATA_DIR / f"{session_id}.json"
    if file_path.exists():
        file_path.unlink()
        return True
    return False


def list_sessions() -> List[Dict[str, Any]]:
    """List all sessions with metadata."""
    sessions = []
    for file_path in get_session_files():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                sessions.append({
                    "session_id": data.get("session_id"),
                    "title": data.get("title", "新对话"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "message_count": len(data.get("messages", [])),
                })
        except Exception:
            continue
    return sessions

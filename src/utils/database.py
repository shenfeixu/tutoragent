"""Database management for user authentication and sessions."""
from __future__ import annotations

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "tutoragent.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'student',
        display_name TEXT,
        email TEXT,
        created_at TEXT NOT NULL,
        last_login TEXT
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT UNIQUE NOT NULL,
        user_id INTEGER NOT NULL,
        title TEXT DEFAULT '新对话',
        messages TEXT DEFAULT '[]',
        accumulated_info TEXT DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS teacher_students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        teacher_id INTEGER NOT NULL,
        student_id INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (teacher_id) REFERENCES users (id),
        FOREIGN KEY (student_id) REFERENCES users (id),
        UNIQUE(teacher_id, student_id)
    )
    """)
    
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(
    username: str,
    password: str,
    role: str = "student",
    display_name: str = None,
    email: str = None,
) -> Optional[int]:
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """
            INSERT INTO users (username, password_hash, role, display_name, email, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (username, hash_password(password), role, display_name or username, email, datetime.now().isoformat()),
        )
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        conn.close()
        return None


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM users WHERE username = ?",
        (username,),
    )
    row = cursor.fetchone()
    conn.close()
    
    if row and row["password_hash"] == hash_password(password):
        return {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "display_name": row["display_name"],
            "email": row["email"],
        }
    return None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "display_name": row["display_name"],
            "email": row["email"],
        }
    return None


def update_last_login(user_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET last_login = ? WHERE id = ?",
        (datetime.now().isoformat(), user_id),
    )
    conn.commit()
    conn.close()


def save_user_session(
    user_id: int,
    session_id: str,
    title: str,
    messages: List[Dict],
    accumulated_info: Dict,
) -> str:
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id FROM sessions WHERE session_id = ? AND user_id = ?",
        (session_id, user_id),
    )
    existing = cursor.fetchone()
    
    now = datetime.now().isoformat()
    messages_json = json.dumps(messages, ensure_ascii=False)
    info_json = json.dumps(accumulated_info, ensure_ascii=False)
    
    if existing:
        cursor.execute(
            """
            UPDATE sessions 
            SET title = ?, messages = ?, accumulated_info = ?, updated_at = ?
            WHERE session_id = ? AND user_id = ?
            """,
            (title, messages_json, info_json, now, session_id, user_id),
        )
    else:
        cursor.execute(
            """
            INSERT INTO sessions (session_id, user_id, title, messages, accumulated_info, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, user_id, title, messages_json, info_json, now, now),
        )
    
    conn.commit()
    conn.close()
    return session_id


def load_user_session(user_id: int, session_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM sessions WHERE session_id = ? AND user_id = ?",
        (session_id, user_id),
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "session_id": row["session_id"],
            "title": row["title"],
            "messages": json.loads(row["messages"]),
            "accumulated_info": json.loads(row["accumulated_info"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
    return None


def list_user_sessions(user_id: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT session_id, title, created_at, updated_at,
               json_array_length(messages) as message_count
        FROM sessions 
        WHERE user_id = ? 
        ORDER BY updated_at DESC
        """,
        (user_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "session_id": row["session_id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message_count": row["message_count"] or 0,
        }
        for row in rows
    ]


def delete_user_session(user_id: int, session_id: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "DELETE FROM sessions WHERE session_id = ? AND user_id = ?",
        (session_id, user_id),
    )
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def add_student_to_teacher(teacher_id: int, student_id: int) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """
            INSERT INTO teacher_students (teacher_id, student_id, created_at)
            VALUES (?, ?, ?)
            """,
            (teacher_id, student_id, datetime.now().isoformat()),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_teacher_students(teacher_id: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT u.id, u.username, u.display_name, u.email, u.created_at
        FROM users u
        JOIN teacher_students ts ON u.id = ts.student_id
        WHERE ts.teacher_id = ?
        ORDER BY u.display_name
        """,
        (teacher_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": row["id"],
            "username": row["username"],
            "display_name": row["display_name"],
            "email": row["email"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def get_student_sessions_for_teacher(teacher_id: int, student_id: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT ts.student_id FROM teacher_students ts
        WHERE ts.teacher_id = ? AND ts.student_id = ?
        """,
        (teacher_id, student_id),
    )
    if not cursor.fetchone():
        conn.close()
        return []
    
    cursor.execute(
        """
        SELECT session_id, title, created_at, updated_at,
               json_array_length(messages) as message_count
        FROM sessions 
        WHERE user_id = ? 
        ORDER BY updated_at DESC
        """,
        (student_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "session_id": row["session_id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message_count": row["message_count"] or 0,
        }
        for row in rows
    ]


def get_all_students() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT id, username, display_name, email, created_at
        FROM users 
        WHERE role = 'student'
        ORDER BY display_name
        """
    )
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": row["id"],
            "username": row["username"],
            "display_name": row["display_name"],
            "email": row["email"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def get_all_sessions_with_evidence() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT s.session_id, s.title, s.messages, s.accumulated_info, 
               s.created_at, s.updated_at, u.id as user_id, u.username, u.display_name
        FROM sessions s
        JOIN users u ON s.user_id = u.id
        ORDER BY s.updated_at DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        messages = json.loads(row["messages"]) if row["messages"] else []
        accumulated = json.loads(row["accumulated_info"]) if row["accumulated_info"] else {}
        
        all_fallacies = []
        all_evidence = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("state"):
                state = msg["state"]
                all_fallacies.extend(state.get("detected_fallacies", []))
                all_evidence.extend(state.get("evidence", []))
        
        results.append({
            "session_id": row["session_id"],
            "title": row["title"],
            "user_id": row["user_id"],
            "username": row["username"],
            "display_name": row["display_name"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message_count": len(messages),
            "fallacies": all_fallacies,
            "evidence": all_evidence,
            "accumulated_info": accumulated,
        })
    
    return results


def get_class_fallacy_stats() -> Dict[str, Any]:
    sessions = get_all_sessions_with_evidence()
    
    fallacy_counts = {}
    total_sessions = len(sessions)
    
    for session in sessions:
        unique_fallacies = set(session["fallacies"])
        for f in unique_fallacies:
            fallacy_counts[f] = fallacy_counts.get(f, 0) + 1
    
    sorted_fallacies = sorted(fallacy_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "total_sessions": total_sessions,
        "fallacy_counts": dict(sorted_fallacies),
        "top_5": sorted_fallacies[:5],
    }


def get_student_scores() -> List[Dict[str, Any]]:
    sessions = get_all_sessions_with_evidence()
    
    student_data = {}
    for session in sessions:
        user_id = session["user_id"]
        if user_id not in student_data:
            student_data[user_id] = {
                "user_id": user_id,
                "username": session["username"],
                "display_name": session["display_name"],
                "sessions": [],
                "total_fallacies": [],
                "accumulated_info": {},
            }
        
        student_data[user_id]["sessions"].append(session)
        student_data[user_id]["total_fallacies"].extend(session["fallacies"])
        
        info = session["accumulated_info"]
        for key, value in info.items():
            if value and key not in student_data[user_id]["accumulated_info"]:
                student_data[user_id]["accumulated_info"][key] = value
    
    results = []
    for user_id, data in student_data.items():
        fallacy_set = set(data["total_fallacies"])
        
        pain_point_score = max(0, 100 - len([f for f in fallacy_set if f in ["H3", "H5"]]) * 15)
        planning_score = max(0, 100 - len([f for f in fallacy_set if f in ["H2", "H9", "H10", "H11"]]) * 12)
        modeling_score = max(0, 100 - len([f for f in fallacy_set if f in ["H4", "H7", "H8"]]) * 15)
        leverage_score = max(0, 100 - len([f for f in fallacy_set if f in ["H1", "H6", "H12"]]) * 12)
        presentation_score = max(0, 100 - len([f for f in fallacy_set if f in ["H13", "H14", "H15"]]) * 12)
        
        total_score = (pain_point_score + planning_score + modeling_score + leverage_score + presentation_score) / 5
        
        results.append({
            "user_id": user_id,
            "username": data["username"],
            "display_name": data["display_name"],
            "session_count": len(data["sessions"]),
            "pain_point_score": pain_point_score,
            "planning_score": planning_score,
            "modeling_score": modeling_score,
            "leverage_score": leverage_score,
            "presentation_score": presentation_score,
            "total_score": total_score,
            "risk_level": "高" if total_score < 60 else "中" if total_score < 80 else "低",
            "fallacies": list(fallacy_set),
            "accumulated_info": data["accumulated_info"],
            "sessions": data["sessions"],
        })
    
    return sorted(results, key=lambda x: x["total_score"])


init_database()

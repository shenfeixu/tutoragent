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
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS intervention_rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        teacher_id INTEGER NOT NULL,
        student_id INTEGER, -- NULL means class-wide
        content TEXT NOT NULL,
        is_active INTEGER DEFAULT 1,
        created_at TEXT NOT NULL,
        FOREIGN KEY (teacher_id) REFERENCES users (id),
        FOREIGN KEY (student_id) REFERENCES users (id)
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


def get_user_memory(user_id: int) -> Optional[str]:
    """Retrieve the latest valid student_memory from the user's past sessions."""
    conn = get_connection()
    cursor = conn.cursor()
    # Search all sessions of user, ordered by latest update
    cursor.execute(
        "SELECT accumulated_info FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT 10",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    for row in rows:
        if row["accumulated_info"]:
            info = json.loads(row["accumulated_info"])
            if "student_memory" in info and info["student_memory"]:
                return info["student_memory"]
    return None


def get_system_stats() -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT count(*) as count FROM users WHERE role = 'student'")
    student_count = cursor.fetchone()["count"]
    
    cursor.execute("SELECT count(*) as count FROM users WHERE role = 'teacher'")
    teacher_count = cursor.fetchone()["count"]
    
    cursor.execute("SELECT count(*) as count FROM sessions")
    session_count = cursor.fetchone()["count"]
    
    cursor.execute("SELECT sum(length(messages)) as m_len FROM sessions")
    msg_len_row = cursor.fetchone()
    msg_len = msg_len_row["m_len"] if (msg_len_row and msg_len_row["m_len"]) else 0
    msg_count = msg_len // 300
    
    conn.close()
    return {
        "student_count": student_count,
        "teacher_count": teacher_count,
        "session_count": session_count,
        "estimated_messages": msg_count,
    }


def get_all_users() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role, display_name, email, created_at, last_login FROM users ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [{
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "display_name": row["display_name"],
        "email": row["email"],
        "created_at": row["created_at"],
        "last_login": row["last_login"]
    } for row in rows]


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

def get_global_fallacy_stats() -> Dict[str, Any]:
    """Alias for class stats, but logically for the entire system."""
    return get_class_fallacy_stats()


def get_global_health_metrics() -> Dict[str, Any]:
    """Calculate average rubric scores across all students."""
    # Default to 互联网+ weights for global benchmark
    default_weights = {
        "pain_point": 0.2,
        "planning": 0.2,
        "modeling": 0.2,
        "leverage": 0.2,
        "presentation": 0.2,
    }
    
    student_scores = get_student_scores(default_weights)
    if not student_scores:
        return {
            "avg_total": 0.0,
            "avg_dims": {k: 0.0 for k in default_weights.keys()}
        }
        
    count = len(student_scores)
    avg_total = sum(s["total_score"] for s in student_scores) / count
    avg_dims = {}
    for dim in default_weights.keys():
        key = f"{dim}_score"
        avg_dims[dim] = sum(s[key] for s in student_scores) / count
        
    return {
        "avg_total": avg_total,
        "avg_dims": avg_dims
    }


def get_student_scores(competition_weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
    sessions = get_all_sessions_with_evidence()
    
    # Default equal weights if not provided
    if not competition_weights:
        competition_weights = {
            "pain_point": 0.2,
            "planning": 0.2,
            "modeling": 0.2,
            "leverage": 0.2,
            "presentation": 0.2,
        }
    
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
        
        total_score = (
            pain_point_score * competition_weights.get("pain_point", 0.2) +
            planning_score * competition_weights.get("planning", 0.2) +
            modeling_score * competition_weights.get("modeling", 0.2) +
            leverage_score * competition_weights.get("leverage", 0.2) +
            presentation_score * competition_weights.get("presentation", 0.2)
        )
        
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


class Neo4jManager:
    """Neo4j 知识图谱管理类"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None
    
    def _get_driver(self):
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
                auth = (self.user, self.password) if self.password else None
                self._driver = GraphDatabase.driver(self.uri, auth=auth)
            except ImportError:
                raise RuntimeError("neo4j package not installed")
        return self._driver
    
    def clear_graph(self):
        """清空图谱中的所有节点和关系"""
        driver = self._get_driver()
        with driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def create_project_nodes(self, projects: List[Dict]):
        """批量创建项目节点"""
        driver = self._get_driver()
        query = """
        UNWIND $projects AS p
        CREATE (project:Project {
            id: p.id,
            name: p.name,
            description: p.description
        })
        """
        with driver.session(database=self.database) as session:
            session.run(query, projects=projects)
    
    def create_tech_nodes(self, techs: List[Dict]):
        """批量创建技术节点"""
        driver = self._get_driver()
        query = """
        UNWIND $techs AS t
        CREATE (tech:Tech {
            project_id: t.project_id,
            name: t.name,
            maturity: t.maturity,
            barrier: t.barrier
        })
        """
        with driver.session(database=self.database) as session:
            session.run(query, techs=techs)
    
    def create_market_nodes(self, markets: List[Dict]):
        """批量创建市场节点"""
        driver = self._get_driver()
        query = """
        UNWIND $markets AS m
        CREATE (market:Market {
            project_id: m.project_id,
            name: m.name,
            tam: m.tam,
            sam: m.sam,
            som: m.som
        })
        """
        with driver.session(database=self.database) as session:
            session.run(query, markets=markets)
    
    def create_risk_nodes(self, risks: List[Dict]):
        """批量创建风险节点"""
        driver = self._get_driver()
        query = """
        UNWIND $risks AS r
        CREATE (risk:Risk {
            project_id: r.project_id,
            name: r.name,
            severity: r.severity
        })
        """
        with driver.session(database=self.database) as session:
            session.run(query, risks=risks)
    
    def create_value_loop_edges(self, edges: List[Dict]):
        """批量创建价值闭环超边节点"""
        driver = self._get_driver()
        query = """
        UNWIND $edges AS e
        CREATE (vle:ValueLoopEdge {
            id: e.id,
            name: e.name,
            description: e.description,
            project_id: e.project_id,
            ltv: e.ltv,
            cac: e.cac,
            revenue_model: e.revenue_model
        })
        """
        with driver.session(database=self.database) as session:
            session.run(query, edges=edges)
    
    def create_risk_pattern_edges(self, edges: List[Dict]):
        """批量创建风险模式超边节点"""
        driver = self._get_driver()
        query = """
        UNWIND $edges AS e
        CREATE (rpe:RiskPatternEdge {
            id: e.id,
            name: e.name,
            description: e.description,
            project_id: e.project_id,
            mitigation: e.mitigation
        })
        """
        with driver.session(database=self.database) as session:
            session.run(query, edges=edges)
    
    def create_relationships(self, project_ids: List[str]):
        """批量创建关系"""
        driver = self._get_driver()
        
        with driver.session(database=self.database) as session:
            session.run("""
                UNWIND $ids AS pid
                MATCH (p:Project {id: pid})
                MATCH (t:Tech {project_id: pid})
                MERGE (p)-[:USE]->(t)
            """, ids=project_ids)
            
            session.run("""
                UNWIND $ids AS pid
                MATCH (p:Project {id: pid})
                MATCH (m:Market {project_id: pid})
                MERGE (p)-[:TARGET]->(m)
            """, ids=project_ids)
            
            session.run("""
                UNWIND $ids AS pid
                MATCH (p:Project {id: pid})
                MATCH (r:Risk {project_id: pid})
                MERGE (p)-[:TRIGGER_RISK]->(r)
            """, ids=project_ids)
            
            session.run("""
                UNWIND $ids AS pid
                MATCH (p:Project {id: pid})
                MATCH (vle:ValueLoopEdge {project_id: pid})
                MERGE (p)-[:HAS_VALUE_LOOP]->(vle)
            """, ids=project_ids)
            
            session.run("""
                UNWIND $ids AS pid
                MATCH (p:Project {id: pid})
                MATCH (rpe:RiskPatternEdge {project_id: pid})
                MERGE (p)-[:HAS_RISK_PATTERN]->(rpe)
            """, ids=project_ids)
            
            session.run("""
                UNWIND $ids AS pid
                MATCH (vle:ValueLoopEdge {project_id: pid})
                MATCH (t:Tech {project_id: pid})
                MERGE (vle)-[:INVOLVES_TECH]->(t)
            """, ids=project_ids)
            
            session.run("""
                UNWIND $ids AS pid
                MATCH (vle:ValueLoopEdge {project_id: pid})
                MATCH (m:Market {project_id: pid})
                MERGE (vle)-[:INVOLVES_MARKET]->(m)
            """, ids=project_ids)
            
            session.run("""
                UNWIND $ids AS pid
                MATCH (rpe:RiskPatternEdge {project_id: pid})
                MATCH (t:Tech {project_id: pid})
                MERGE (rpe)-[:INVOLVES_TECH]->(t)
            """, ids=project_ids)
            
            session.run("""
                UNWIND $ids AS pid
                MATCH (rpe:RiskPatternEdge {project_id: pid})
                MATCH (r:Risk {project_id: pid})
                MERGE (rpe)-[:INVOLVES_RISK]->(r)
            """, ids=project_ids)
    
    def load_kg_from_json(self, json_path: str) -> Dict[str, Any]:
        """从 JSON 文件加载知识图谱数据"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        projects_data = data.get("projects", [])
        
        projects = []
        techs = []
        markets = []
        risks = []
        value_loop_edges = []
        risk_pattern_edges = []
        project_ids = []
        
        for p in projects_data:
            project_ids.append(p["id"])
            
            projects.append({
                "id": p["id"],
                "name": p["name"],
                "description": p.get("description", "")
            })
            
            tech = p.get("tech", {})
            techs.append({
                "project_id": p["id"],
                "name": tech.get("name", ""),
                "maturity": tech.get("maturity", "概念验证"),
                "barrier": tech.get("barrier", "中")
            })
            
            market = p.get("market", {})
            markets.append({
                "project_id": p["id"],
                "name": market.get("name", ""),
                "tam": market.get("tam", 0),
                "sam": market.get("sam", 0),
                "som": market.get("som", 0)
            })
            
            risk = p.get("risk", {})
            risks.append({
                "project_id": p["id"],
                "name": risk.get("name", ""),
                "severity": risk.get("severity", "中")
            })
            
            vle = p.get("value_loop_edge", {})
            metrics = vle.get("metrics", {})
            value_loop_edges.append({
                "id": vle.get("id", f"vl_{p['id']}"),
                "name": vle.get("name", ""),
                "description": vle.get("description", ""),
                "project_id": p["id"],
                "ltv": metrics.get("ltv", 0),
                "cac": metrics.get("cac", 0),
                "revenue_model": metrics.get("revenue_model", "")
            })
            
            rpe = p.get("risk_pattern_edge", {})
            risk_pattern_edges.append({
                "id": rpe.get("id", f"rp_{p['id']}"),
                "name": rpe.get("name", ""),
                "description": rpe.get("description", ""),
                "project_id": p["id"],
                "mitigation": rpe.get("mitigation", "")
            })
        
        self.clear_graph()
        
        self.create_project_nodes(projects)
        self.create_tech_nodes(techs)
        self.create_market_nodes(markets)
        self.create_risk_nodes(risks)
        self.create_value_loop_edges(value_loop_edges)
        self.create_risk_pattern_edges(risk_pattern_edges)
        self.create_relationships(project_ids)
        
        return {
            "success": True,
            "projects_count": len(projects),
            "techs_count": len(techs),
            "markets_count": len(markets),
            "risks_count": len(risks),
            "value_loop_edges_count": len(value_loop_edges),
            "risk_pattern_edges_count": len(risk_pattern_edges),
        }
    
    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None


init_database()



def add_intervention_rule(teacher_id: int, content: str, student_id: int = None) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO intervention_rules (teacher_id, student_id, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (teacher_id, student_id, content, datetime.now().isoformat()),
    )
    rule_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return rule_id


def get_all_intervention_rules(teacher_id: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT r.*, u.display_name as student_name
        FROM intervention_rules r
        LEFT JOIN users u ON r.student_id = u.id
        WHERE r.teacher_id = ?
        ORDER BY r.created_at DESC
        """,
        (teacher_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_intervention_rule(rule_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM intervention_rules WHERE id = ?", (rule_id,))
    conn.commit()
    conn.close()


def get_active_intervention_rules(teacher_id: int, student_id: int = None) -> List[str]:
    conn = get_connection()
    cursor = conn.cursor()
    # Fetch class-wide rules (student_id IS NULL) OR specific student rules
    query = """
    SELECT content FROM intervention_rules 
    WHERE teacher_id = ? AND is_active = 1 
    AND (student_id IS NULL OR student_id = ?)
    """
    cursor.execute(query, (teacher_id, student_id))
    rows = cursor.fetchall()
    conn.close()
    return [row["content"] for row in rows]

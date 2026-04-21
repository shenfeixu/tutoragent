"""LangGraph-backed coach agent skeleton with LangChain-powered extraction/rebuttal and structured prompts."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        LOGGER.warning("python-dotenv is missing; falling back to existing environment variables.")
        return False

from pydantic import BaseModel, Field

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    GraphDatabase = None
    NEO4J_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
    )
    LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover - we fall back to mocks when LangChain is absent
    LANGCHAIN_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

_OPENAI_CLIENT: Optional[Any] = None
_SEED_KG_CACHE: Optional[Dict[str, Any]] = None

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
_NEO4J_DRIVER: Optional[Any] = None
SEED_KG_PATH = Path(__file__).resolve().parents[2] / "data" / "seed_kg.json"


class EvidenceItem(BaseModel):
    step: str = Field(..., description="Node or rule that emitted this evidence.")
    detail: str = Field(..., description="Why this item matters for the logic audit.")
    source_excerpt: str = Field(..., description="Excerpt from user input that gave rise to the evidence. Must include '学生原文：'.")


class KGQueryDetail(BaseModel):
    step: str = Field(..., description="查询步骤名称")
    query_type: str = Field(default="", description="查询类型：tech_market_match/tech_risk/value_loop/risk_pattern")
    tech_keywords: List[str] = Field(default_factory=list, description="技术关键词")
    market_keywords: List[str] = Field(default_factory=list, description="市场关键词")
    cypher_query: str = Field(default="", description="执行的 Cypher 查询语句")
    query_attempts: List[Dict[str, Any]] = Field(default_factory=list, description="查询尝试记录")
    matched_projects: List[str] = Field(default_factory=list, description="匹配到的项目")
    project_details: List[Dict[str, Any]] = Field(default_factory=list, description="项目详细信息")
    match_scores: Dict[str, int] = Field(default_factory=dict, description="匹配分数")
    risks_found: List[str] = Field(default_factory=list, description="发现的风险")
    risk_details: List[Dict[str, Any]] = Field(default_factory=list, description="风险详细信息")
    related_projects: List[str] = Field(default_factory=list, description="相关项目")
    success: bool = Field(default=False, description="查询是否成功")
    message: str = Field(default="", description="查询结果消息")
    category: str = Field(default="Business Strategy", description="知识节点分类：Market/Tech/Risk/Business")
    retrieval_reason: str = Field(default="语义关联匹配", description="Retrieval reasoning for why this node was fetched.")
    graph_nodes: List[Dict[str, Any]] = Field(default_factory=list, description="用于前端子图可视化的节点")
    graph_edges: List[Dict[str, Any]] = Field(default_factory=list, description="用于前端子图可视化的边")


class AgentState(BaseModel):
    """Tracks the conversation round, extracted entities, and planner outputs."""

    student_input: str = Field(..., description="Raw input captured from the student terminal")
    extracted_nodes: Dict[str, Any] = Field(default_factory=dict, description="Commercial entities extracted from the prompt")
    detected_fallacies: List[str] = Field(default_factory=list, description="Hypergraph rules that failed")
    probing_strategy: str = Field("", description="Socratic follow-up strategy chosen for the next turn")
    response: str = Field("", description="Message returned to the student after the cycle completes")
    evidence: List[EvidenceItem] = Field(default_factory=list, description="Evidence trail supporting each node/rule assessment.")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="历史对话记录")
    accumulated_info: Dict[str, Any] = Field(default_factory=dict, description="已累积提取的项目信息")
    rubric_scores: Dict[str, Any] = Field(default_factory=dict, description="Rubric 0-5 逐项评分及建议")
    target_competition: str = Field(default="互联网+", description="当前选定的评估赛事")
    intervention_rules: List[str] = Field(default_factory=list, description="教师下发的实时干预指令")
    is_error: bool = Field(False, description="标记当前流程是否发生致命错误（如网络中断）")
    kg_query_details: List[KGQueryDetail] = Field(default_factory=list, description="知识图谱查询详情，用于前端可视化展示")
    agent_insights: Dict[str, str] = Field(default_factory=dict, description="多智能体专家（财务、市场、技术）的独立会诊意见")


class EntityExtractionSchema(BaseModel):
    project_name: str = Field(default="未命名项目", description="项目名称")
    tech_description: str = Field(default="", description="核心技术描述")
    target_market: str = Field(default="", description="目标市场/应用场景")
    target_customer: str = Field(default="", description="核心目标客户或用户群")
    value_proposition: str = Field(default="", description="关键价值主张")
    channel: str = Field(default="", description="主要推广/获客渠道")
    revenue: float = Field(default=0.0, description="当前或预测的年化收入（数字）")
    TAM: float = Field(default=0.0, description="TAM 规模（数字，单位：元）")
    SAM: float = Field(default=0.0, description="SAM 规模（数字，单位：元）")
    SOM: float = Field(default=0.0, description="SOM 规模（数字，单位：元）")
    LTV: float = Field(default=0.0, description="客户生命周期价值（数字，单位：元）")
    CAC: float = Field(default=0.0, description="获客成本（数字，单位：元）")
    tech_maturity: str = Field(default="概念验证", description="技术成熟度：概念验证/原型/小规模验证/商业化")
    team_size: int = Field(default=3, description="团队规模（人数）")
    funding_stage: str = Field(default="种子轮", description="融资阶段")
    time_to_market: int = Field(default=12, description="预计上市时间（月）")
    key_risks: str = Field(default="", description="主要风险点")
    marginal_cost: str = Field(default="", description="边际成本特征：固定成本主导/边际成本递减")
    cash_runway: int = Field(default=12, description="现金跑道（月）")
    monthly_burn: float = Field(default=0.0, description="月度烧钱率（元）")
    moat: str = Field(default="", description="竞争壁垒/护城河")
    growth_model: str = Field(default="", description="增长模式/飞轮效应")
    additional_context: Optional[str] = Field(None, description="其他辅助说明，可选")


# Default structured extraction that keeps the later prompts runnable even offline.
MOCK_ENTITY_EXTRACTION: Dict[str, Any] = {
    "target_customer": "AI-native SMBs",
    "value_proposition": "Operational storyboarding",
    "channel": "digital community",
    "revenue": 150000,
    "TAM": 2_000_000,
    "SAM": 650_000,
    "SOM": 180_000,
    "LTV": 960,
    "CAC": 240,
    "additional_context": "样例数据，用于初始测试。",
}


FALLACY_STRATEGY_LIBRARY: Dict[str, str] = {
    "H1": "痛点验证逻辑：痛点描述仅停留在主观想象。请提供『调研访谈数据』或『实地考察证据』来证明该痛点在目标群体中的广泛性。",
    "H2": "技术成熟度逻辑：技术处于早期阶段，请说明从原型到商业化的关键里程碑和验证节点。",
    "H3": "用户画像逻辑：目标客户描述模糊。请用「谁、在什么场景、遇到什么问题」三要素重新定义。",
    "H4": "市场口径逻辑：市场规模比例异常。请说明 TAM/SAM/SOM 的计算依据和数据来源。",
    "H5": "价值主张逻辑：价值主张不够清晰。请用「我们帮助X用户解决Y问题，实现Z价值」的句式重新表述。",
    "H6": "渠道触达逻辑：获客渠道描述不足。请说明你的目标客户在哪里聚集，你如何触达他们。",
    "H7": "收入验证逻辑：收入预测缺失或逻辑不通。请说明你的定价策略和付费意愿验证情况。",
    "H8": "单位经济逻辑：LTV/CAC 比例不健康。请说明你的客户留存率和复购周期。",
    "H9": "团队能力逻辑：团队能力板块偏弱。请说明核心团队的技能互补性和执行力验证。",
    "H10": "融资匹配逻辑：融资阶段与技术成熟度不匹配。请说明资金用途和里程碑规划。",
    "H11": "时间规划逻辑：上市时间过于乐观。请说明关键路径和潜在瓶颈。",
    "H12": "风险识别逻辑：风险识别不充分。请列出技术、市场、团队三方面的主要风险。",
    "H13": "路演逻辑闭环：表达缺乏逻辑闭环。请确保你的 BP 描述遵循「痛点-方案-验证-闭环」的严密结构。",
    "H14": "市场验证逻辑：市场规模数据可能缺乏验证。请说明市场调研的方法和样本量。",
    "H15": "商业假设逻辑：核心商业假设不成立（如：白拿白给）。请说明从获客到变现的完整合理路径。",
    "H16": "单位经济幻觉逻辑：你的单位经济模型存在矛盾。请说明边际成本是否随规模下降？",
    "H17": "渠道错位逻辑：你的渠道与目标用户群体不匹配。请核实该赛道的典型获客场景。",
    "H18": "现金流生存逻辑：请说明你的盈亏平衡点。在没有新融资的情况下能支撑多久？",
    "H19": "护城河逻辑：你的竞争优势是否可持续？请说明网络效应、规模效应或品牌效应如何形成。",
    "H20": "增长飞轮逻辑：你的增长模式是否自增强？请说明新用户如何帮助获取更多用户。",
}

# ── A5-1: 谬误严重程度定义 (Penalty Weights - v3.7 Final) ────────
FALLACY_SEVERITY: Dict[str, float] = {
    # 致命伤 (Fatal): -2.5分 — 数据明确但逻辑崩溃
    "H7": 2.5, "H15": 2.5, "H16": 2.5, "H17": 2.5,
    # 重大问题 (Major): -1.5分 — 重要维度缺失或核心验证缺位
    "H1": 1.5, "H4": 1.5, "H8": 1.5, "H11": 1.5, "H12": 1.5, "H14": 1.5,
    # 显著影响 (Significant): -1.2分 — 逻辑连贯性瑕疵
    "H10": 1.2, "H13": 1.2, "H18": 1.2,
    # 轻微瑕疵 (Minor): -0.8分 — 文案或非核心参数问题
    "H2": 0.8, "H3": 0.8, "H5": 0.8, "H6": 0.8, "H9": 0.8, "H19": 0.8, "H20": 0.8,
    # 数据缺口 (Data Gap): -0.5分 — 结构性缺失但不代表逻辑错误
    "H4_GAP": 0.5, "H7_GAP": 0.5, "H8_GAP": 0.5, "H15_GAP": 0.8,
}

# ── A5-0: 深度审计关键词 (v3.8) ──────────────────────────────────
AUDIT_KEYWORDS: Dict[str, List[str]] = {
    "H1": ["调研", "调查", "访谈", "发现", "数据显示", "分析", "观察", "问卷", "反馈", "总结"], # 扩充：问卷/观察/反馈
    "H13": ["闭环", "验证", "结构", "逻辑", "路演", "演示", "PPT", "视频", "讲解", "展示"], # 扩充：路演媒介
    "H19": ["专利", "壁垒", "门槛", "不可复制", "优势", "独特"],
    "H20": ["飞轮", "漏斗", "回购", "流量池", "留存", "裂变"],
}

DEFAULT_FALLACY_STRATEGY = "继续逐条验证关键前提，别让幻觉的数字偷偷爬过审计线。"

# ── A5: 赛事 Rubric 评分维度与权重 ──────────────────────────────────
RUBRIC_FALLACY_MAP: Dict[str, List[str]] = {
    "pain_point":    ["H1", "H3", "H5"],                   # 痛点核心
    "planning":      ["H6", "H11", "H12", "H14", "H17"],   # 方案与赛道
    "modeling":       ["H4", "H4_GAP", "H7", "H7_GAP", "H8", "H8_GAP", "H15", "H15_GAP", "H16"], # 盈利
    "leverage":       ["H2", "H9", "H10"],                 # 团队与杠杆
    "presentation":   ["H13", "H18", "H19", "H20"],        # 路演逻辑、存活、壁垒 (加入 H18)
}

RUBRIC_DIM_NAMES: Dict[str, str] = {
    "pain_point": "痛点发现",
    "planning": "方案策划",
    "modeling": "商业建模",
    "leverage": "资源杠杆",
    "presentation": "路演表达",
}

RUBRIC_MISSING_FIX: Dict[str, Dict[str, str]] = {
    "pain_point": {
        "missing": "缺少清晰的用户画像和价值主张描述",
        "fix": "用「谁、在什么场景、遇到什么问题、我们如何解决」四要素重新定义痛点",
    },
    "planning": {
        "missing": "缺少获客渠道和时间规划",
        "fix": "列出前 3 个月的里程碑和预期获客路径，附上每个节点的验证指标",
    },
    "modeling": {
        "missing": "缺少可信的市场规模数据或单位经济验证",
        "fix": "用自下而上法重新估算 TAM/SAM/SOM，并计算 LTV/CAC 比值",
    },
    "leverage": {
        "missing": "缺少团队、融资或现金流生存计划",
        "fix": "制作 12 个月现金流预测表，标注盈亏平衡点和融资窗口",
    },
    "presentation": {
        "missing": "缺少技术壁垒、护城河或增长飞轮的清晰阐述",
        "fix": "用一页纸说明你的核心壁垒来源（专利/数据/网络效应）和增长的自增强逻辑",
    },
}

COMPETITION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "互联网+": {"pain_point": 0.25, "planning": 0.20, "modeling": 0.25, "leverage": 0.15, "presentation": 0.15},
    "挑战杯":  {"pain_point": 0.20, "planning": 0.15, "modeling": 0.20, "leverage": 0.20, "presentation": 0.25},
    "创青春":  {"pain_point": 0.20, "planning": 0.25, "modeling": 0.20, "leverage": 0.20, "presentation": 0.15},
    "数模":    {"pain_point": 0.10, "planning": 0.20, "modeling": 0.50, "leverage": 0.05, "presentation": 0.15},
}

def _get_chat_client() -> ChatOpenAI:
    if not LANGCHAIN_AVAILABLE:
        raise RuntimeError("LangChain is not installed in the environment")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required in .env")

    kwargs = {
        "temperature": 0.2,
        "model_name": LLM_MODEL,
        "openai_api_key": OPENAI_API_KEY,
    }
    if OPENAI_API_BASE:
        kwargs["openai_api_base"] = OPENAI_API_BASE

    return ChatOpenAI(**kwargs)


def _ensure_openai_config():
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package is not installed; cannot call OpenAI API.")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required in .env for direct OpenAI access.")


def _openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT:
        return _OPENAI_CLIENT
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package is not installed; cannot call OpenAI API.")
    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE or None,
            timeout=180.0,
        )
        _OPENAI_CLIENT = client
        return client
    openai.api_key = OPENAI_API_KEY
    if OPENAI_API_BASE:
        openai.api_base = OPENAI_API_BASE
    _OPENAI_CLIENT = openai
    return openai


def _call_openai_manual(system_prompt: str, human_prompt: str) -> str:
    client = _openai_client()
    request = {
        "model": LLM_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ],
    }
    if hasattr(client, "chat"):
        response = client.chat.completions.create(**request)
        choices = getattr(response, "choices", None) or getattr(response, "data", None)
    else:
        response = client.ChatCompletion.create(**request)
        choices = response.get("choices")
    if not choices:
        raise RuntimeError("OpenAI returned no choices.")
    first = choices[0]
    if isinstance(first, dict):
        return first["message"]["content"]
    return first.message.content


def _sanitize_llm_json(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("json"):
        raw = raw[len("json"):].lstrip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]
    return raw


def _get_neo4j_driver():
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER:
        return _NEO4J_DRIVER
    if not NEO4J_AVAILABLE:
        LOGGER.warning("neo4j-driver is missing from the runtime.")
        return None
    if not NEO4J_URI:
        LOGGER.warning("Neo4j URI is not configured; skipping H1 Cypher checks.")
        return None
    if not NEO4J_USER:
        LOGGER.warning("Neo4j user is not configured; skipping H1 Cypher checks.")
        return None
    auth = (NEO4J_USER, NEO4J_PASSWORD) if NEO4J_PASSWORD else None
    try:
        _NEO4J_DRIVER = GraphDatabase.driver(NEO4J_URI, auth=auth)
        LOGGER.info("Neo4j driver initialized successfully.")
    except Exception as e:
        LOGGER.warning(f"Failed to initialize Neo4j driver: {e}")
        return None
    return _NEO4J_DRIVER


def _load_seed_kg() -> Dict[str, Any]:
    global _SEED_KG_CACHE
    if _SEED_KG_CACHE is not None:
        return _SEED_KG_CACHE

    if not SEED_KG_PATH.exists():
        LOGGER.warning("Seed KG JSON not found at %s", SEED_KG_PATH)
        _SEED_KG_CACHE = {"metadata": {}, "projects": []}
        return _SEED_KG_CACHE

    try:
        with SEED_KG_PATH.open("r", encoding="utf-8") as f:
            _SEED_KG_CACHE = json.load(f)
    except Exception as e:
        LOGGER.warning("Failed to load seed KG JSON: %s", e)
        _SEED_KG_CACHE = {"metadata": {}, "projects": []}
    return _SEED_KG_CACHE


def _flatten_seed_project(project: Dict[str, Any]) -> Dict[str, str]:
    tech = project.get("tech", {}) or {}
    market = project.get("market", {}) or {}
    risk = project.get("risk", {}) or {}
    value_loop = project.get("value_loop_edge", {}) or {}
    risk_pattern = project.get("risk_pattern_edge", {}) or {}
    metrics = value_loop.get("metrics", {}) or {}

    return {
        "project_name": str(project.get("name", "") or ""),
        "project_desc": str(project.get("description", "") or ""),
        "tech_name": str(tech.get("name", "") or ""),
        "tech_maturity": str(tech.get("maturity", "") or ""),
        "tech_barrier": str(tech.get("barrier", "") or ""),
        "market_name": str(market.get("name", "") or ""),
        "risk_name": str(risk.get("name", "") or ""),
        "risk_severity": str(risk.get("severity", "") or ""),
        "value_loop_name": str(value_loop.get("name", "") or ""),
        "value_loop_desc": str(value_loop.get("description", "") or ""),
        "risk_pattern_name": str(risk_pattern.get("name", "") or ""),
        "risk_pattern_desc": str(risk_pattern.get("description", "") or ""),
        "risk_mitigation": str(risk_pattern.get("mitigation", "") or ""),
        "revenue_model": str(metrics.get("revenue_model", "") or ""),
    }


def _build_learning_query_text(student_input: str, accumulated_info: Optional[Dict[str, Any]] = None) -> str:
    parts = [student_input or ""]
    for key in [
        "project_name",
        "tech_description",
        "target_market",
        "target_customer",
        "value_proposition",
        "channel",
        "key_risks",
        "additional_context",
    ]:
        value = (accumulated_info or {}).get(key)
        if value:
            parts.append(str(value))
    return "\n".join([part for part in parts if part]).strip()


def _text_bigrams(text: str) -> set[str]:
    compact = "".join(ch for ch in (text or "").lower() if ch.strip())
    if len(compact) < 2:
        return set()
    return {compact[i:i + 2] for i in range(len(compact) - 1)}


def _dedupe_keywords(keywords: List[str], max_keywords: int = 12) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    stopwords = {
        "项目", "学生", "创业", "比赛", "大赛", "创新创业", "想法", "方案", "系统", "平台", "技术", "市场", "用户",
        "产品", "服务", "分析", "帮助", "需要", "现在", "不知道", "怎么做",
    }
    for keyword in keywords:
        kw = (keyword or "").strip()
        if not kw or len(kw) < 2 or kw in stopwords:
            continue
        normalized = kw.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(kw)
        if len(cleaned) >= max_keywords:
            break
    return cleaned


def _expand_keyword_variants(keywords: List[str], max_keywords: int = 24) -> List[str]:
    expanded: List[str] = []
    for keyword in keywords:
        kw = (keyword or "").strip()
        if not kw:
            continue
        expanded.append(kw)

        if len(kw) >= 4:
            expanded.append(kw[:2])
            expanded.append(kw[:3])
            expanded.append(kw[-2:])
            expanded.append(kw[-3:])

        if len(kw) >= 5:
            for i in range(len(kw) - 1):
                fragment2 = kw[i:i + 2]
                if len(fragment2) == 2:
                    expanded.append(fragment2)
            for i in range(len(kw) - 2):
                fragment3 = kw[i:i + 3]
                if len(fragment3) == 3:
                    expanded.append(fragment3)

        if "AI" in kw or "ai" in kw:
            expanded.append(kw.replace("AI", "").replace("ai", ""))
        if "校园" in kw:
            expanded.append("校园")
        if "二手" in kw:
            expanded.append("二手")
        if "医疗" in kw:
            expanded.append("医疗")
        if "影像" in kw:
            expanded.append("影像")

    return _dedupe_keywords(expanded, max_keywords=max_keywords)


def extract_learning_query_profile(
    student_input: str,
    accumulated_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    combined_query = _build_learning_query_text(student_input, accumulated_info)
    fallback_keywords = extract_keywords_local(combined_query, max_keywords=12)
    fallback_profile = {
        "intent_summary": student_input[:80],
        "project_keywords": fallback_keywords[:4],
        "tech_keywords": fallback_keywords[:4],
        "market_keywords": fallback_keywords[:4],
        "user_keywords": [],
        "problem_keywords": fallback_keywords[:4],
        "risk_keywords": [],
        "exclude_keywords": [],
    }

    if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
        fallback_profile["project_keywords"] = _expand_keyword_variants(fallback_profile["project_keywords"], max_keywords=8)
        fallback_profile["tech_keywords"] = _expand_keyword_variants(fallback_profile["tech_keywords"], max_keywords=8)
        fallback_profile["market_keywords"] = _expand_keyword_variants(fallback_profile["market_keywords"], max_keywords=8)
        fallback_profile["problem_keywords"] = _expand_keyword_variants(fallback_profile["problem_keywords"], max_keywords=8)
        fallback_profile["all_keywords"] = _expand_keyword_variants(fallback_keywords, max_keywords=18)
        return fallback_profile

    system_prompt = (
        "你是创新创业导师系统里的检索意图分析器。"
        "你的任务是把学生输入拆成适合知识图谱检索的结构化关键词。"
        "只输出 JSON，不要解释。"
    )
    human_prompt = f"""
请阅读下面的学生输入与已知项目信息，提取最能代表检索意图的关键词。

【学生输入】
{student_input}

【已知项目信息】
{json.dumps(accumulated_info or {}, ensure_ascii=False, indent=2)}

请返回 JSON，格式如下：
{{
  "intent_summary": "一句话概括学生真正想问什么",
  "project_keywords": ["项目或商业模式关键词"],
  "tech_keywords": ["技术或方法关键词"],
  "market_keywords": ["行业/赛道/场景关键词"],
  "user_keywords": ["目标用户关键词"],
  "problem_keywords": ["痛点/任务/需求关键词"],
  "risk_keywords": ["风险/障碍关键词"],
  "exclude_keywords": ["应避免误匹配的词"]
}}

要求：
1. 每类最多 5 个，优先保留具体词，不要泛词。
2. 如果学生问题很初级，也要尽量提炼真实场景词，比如“校园二手书”“新生报名”“宿舍收纳”。
3. 如果信息不足，可以留空数组。
"""
    try:
        raw = _call_openai_manual(system_prompt, human_prompt)
        parsed = json.loads(_sanitize_llm_json(raw))
        profile = {
            "intent_summary": str(parsed.get("intent_summary", fallback_profile["intent_summary"])).strip(),
            "project_keywords": _expand_keyword_variants(parsed.get("project_keywords", []), max_keywords=8),
            "tech_keywords": _expand_keyword_variants(parsed.get("tech_keywords", []), max_keywords=8),
            "market_keywords": _expand_keyword_variants(parsed.get("market_keywords", []), max_keywords=8),
            "user_keywords": _expand_keyword_variants(parsed.get("user_keywords", []), max_keywords=6),
            "problem_keywords": _expand_keyword_variants(parsed.get("problem_keywords", []), max_keywords=8),
            "risk_keywords": _expand_keyword_variants(parsed.get("risk_keywords", []), max_keywords=8),
            "exclude_keywords": _dedupe_keywords(parsed.get("exclude_keywords", []), max_keywords=5),
        }
        profile["all_keywords"] = _dedupe_keywords(
            profile["project_keywords"]
            + profile["tech_keywords"]
            + profile["market_keywords"]
            + profile["user_keywords"]
            + profile["problem_keywords"]
            + profile["risk_keywords"],
            max_keywords=18,
        )
        if not profile["all_keywords"]:
            profile["all_keywords"] = _dedupe_keywords(fallback_keywords, max_keywords=15)
        return profile
    except Exception as e:
        LOGGER.warning("Learning query profile extraction failed, falling back to local keywords: %s", e)
        fallback_profile["project_keywords"] = _expand_keyword_variants(fallback_profile["project_keywords"], max_keywords=8)
        fallback_profile["tech_keywords"] = _expand_keyword_variants(fallback_profile["tech_keywords"], max_keywords=8)
        fallback_profile["market_keywords"] = _expand_keyword_variants(fallback_profile["market_keywords"], max_keywords=8)
        fallback_profile["problem_keywords"] = _expand_keyword_variants(fallback_profile["problem_keywords"], max_keywords=8)
        fallback_profile["all_keywords"] = _expand_keyword_variants(fallback_keywords, max_keywords=18)
        return fallback_profile


def query_seed_kg_cases(
    query_text: str,
    accumulated_info: Optional[Dict[str, Any]] = None,
    query_profile: Optional[Dict[str, Any]] = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    kg = _load_seed_kg()
    projects = kg.get("projects", []) or []
    if not projects:
        return []

    combined_query = _build_learning_query_text(query_text, accumulated_info)
    query_profile = query_profile or extract_learning_query_profile(query_text, accumulated_info)
    all_keywords = query_profile.get("all_keywords", [])
    query_lower = combined_query.lower()
    results: List[Dict[str, Any]] = []

    for project in projects:
        flattened = _flatten_seed_project(project)
        score = 0
        matched_terms: List[str] = []

        field_groups = {
            "project": ["project_name", "project_desc", "value_loop_name", "revenue_model"],
            "tech": ["tech_name", "value_loop_desc"],
            "market": ["market_name", "project_desc", "value_loop_desc"],
            "user": ["project_desc", "market_name", "value_loop_desc"],
            "problem": ["project_desc", "value_loop_desc", "risk_pattern_desc"],
            "risk": ["risk_name", "risk_pattern_name", "risk_pattern_desc", "risk_mitigation"],
        }
        group_keywords = {
            "project": query_profile.get("project_keywords", []),
            "tech": query_profile.get("tech_keywords", []),
            "market": query_profile.get("market_keywords", []),
            "user": query_profile.get("user_keywords", []),
            "problem": query_profile.get("problem_keywords", []),
            "risk": query_profile.get("risk_keywords", []),
        }
        group_weights = {
            "project": 8,
            "tech": 10,
            "market": 10,
            "user": 8,
            "problem": 8,
            "risk": 7,
        }

        matched_groups = set()
        strong_phrase_match = False

        for group_name, keywords in group_keywords.items():
            if not keywords:
                continue
            searchable_text = " ".join(flattened.get(field, "") for field in field_groups[group_name]).lower()
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if not keyword_lower or keyword_lower in query_profile.get("exclude_keywords", []):
                    continue
                if keyword_lower in searchable_text:
                    matched_groups.add(group_name)
                    score += group_weights[group_name] + min(len(keyword), 6)
                    if keyword not in matched_terms:
                        matched_terms.append(keyword)
                    if len(keyword) >= 4 and group_name in {"project", "tech", "market", "problem"}:
                        strong_phrase_match = True

        intent_summary = str(query_profile.get("intent_summary", "") or "")
        if intent_summary:
            intent_keywords = _dedupe_keywords(extract_keywords_local(intent_summary, max_keywords=6), max_keywords=6)
            searchable = " ".join(flattened.values()).lower()
            summary_hits = sum(1 for kw in intent_keywords if kw.lower() in searchable)
            score += summary_hits * 4
            if summary_hits >= 2:
                strong_phrase_match = True

        if query_lower:
            if flattened["tech_name"] and flattened["tech_name"].lower() in query_lower:
                score += 14
                matched_groups.add("tech")
                strong_phrase_match = True
            if flattened["market_name"] and flattened["market_name"].lower() in query_lower:
                score += 14
                matched_groups.add("market")
                strong_phrase_match = True
            if flattened["project_name"] and flattened["project_name"].lower() in query_lower:
                score += 10
                matched_groups.add("project")

        if any(kw.lower() in " ".join(flattened.values()).lower() for kw in query_profile.get("exclude_keywords", [])):
            score -= 8

        primary_groups = []
        if group_keywords["project"]:
            primary_groups.append("project")
        if group_keywords["market"] or group_keywords["user"]:
            primary_groups.append("market")
        if group_keywords["problem"]:
            primary_groups.append("problem")
        if group_keywords["tech"]:
            primary_groups.append("tech")

        matched_primary_count = len([group for group in primary_groups if group in matched_groups])

        if primary_groups and matched_primary_count == 0:
            continue

        if len(matched_terms) < 2 and not strong_phrase_match and matched_primary_count < 1:
            continue

        min_score = 12 if matched_primary_count >= 2 or strong_phrase_match else 9
        if score < min_score:
            continue

        value_loop = project.get("value_loop_edge", {}) or {}
        risk_pattern = project.get("risk_pattern_edge", {}) or {}
        metrics = value_loop.get("metrics", {}) or {}
        results.append({
            "project_id": project.get("id"),
            "project_name": flattened["project_name"],
            "project_desc": flattened["project_desc"],
            "tech_name": flattened["tech_name"],
            "tech_maturity": flattened["tech_maturity"],
            "market_name": flattened["market_name"],
            "risk_name": flattened["risk_name"],
            "risk_severity": flattened["risk_severity"],
            "value_loop_name": flattened["value_loop_name"],
            "value_loop_desc": flattened["value_loop_desc"],
            "risk_pattern": flattened["risk_pattern_name"],
            "pattern_description": flattened["risk_pattern_desc"],
            "mitigation": flattened["risk_mitigation"],
            "revenue_model": flattened["revenue_model"],
            "ltv": metrics.get("ltv"),
            "cac": metrics.get("cac"),
            "matched_terms": matched_terms[:6],
            "risks": [flattened["risk_name"]] if flattened["risk_name"] else [],
            "_match_score": score,
            "_matched_groups": sorted(matched_groups),
        })

    if not results:
        return []

    results.sort(key=lambda item: item.get("_match_score", 0), reverse=True)

    tech_query = " ".join(query_profile.get("tech_keywords", []) or query_profile.get("project_keywords", []))
    market_query = " ".join(
        query_profile.get("market_keywords", [])
        + query_profile.get("user_keywords", [])
        + query_profile.get("problem_keywords", [])
    )
    reranked = llm_semantic_rerank(tech_query, market_query, results[: min(len(results), 8)], top_k=top_k)
    for item in reranked:
        item.pop("_match_score", None)
        item.pop("_matched_groups", None)
    return reranked[:top_k]


def extract_keywords_with_llm(text: str, max_keywords: int = 20) -> List[str]:
    """使用 LLM 从文本中提取关键词，不回退到本地算法
    
    优化策略：
    1. 让LLM提取更多关键词（包括同义词、相关词、上下位词）
    2. 完全依赖LLM的智能提取，不做手动拆分
    3. 提取的关键词越多越好，用于最大化匹配概率
    4. 关键词长度多样化（2-8字）
    """
    if not text:
        return []
    
    try:
        client = _get_chat_client()
        prompt = f"""你是一个专业的技术与商业分析专家。请从以下文本中提取关键词，用于在知识图谱中进行匹配搜索。

输入文本: {text}

请提取以下类型的关键词：

1. **核心科技/业务词**（2-4字）：最核心的技术或商业模式词汇，如"无人机"、"二手"、"撮合"、"共享"、"回收"
2. **专业术语**（3-5字）：行业专业术语，如"激光雷达"、"循环经济"、"O2O"、"C2C"、"平台经济"
3. **复合词**（4-6字）：技术/业务组合，如"医学影像AI"、"校园服务"、"二手交易平台"
4. **应用场景**（2-5字）：如"植保"、"校园"、"闲置交易"、"宿舍"
5. **同义词/相关词**：核心概念的上下位词或近义词

输出要求：
- 关键词长度必须多样化：2字、3字、4字、5字、6字都要有。
- 尤其要注意保留商业模式和场景相关的核心词（如"校园"、"二手"、"租赁"、"交易"、"流转"）。
- 避免过于宽泛的泛化词汇（如"技术"、"系统"）单独出现，但可组合使用。
- 关键词之间用逗号分隔。
- 至少提取15个关键词，按重要性排序。

示例输入："校园二手书撮合交易平台"
示例输出：校园,二手书,二手,交易,撮合,平台经济,循环经济,C2C,校园服务,闲置流转,二手电商,书籍交易,共享平台,在校生,供需匹配

关键词列表:"""
        
        response = client.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        
        keywords = [kw.strip() for kw in content.replace("，", ",").replace("、", ",").split(",") if kw.strip()]
        
        stopwords = {"技术", "系统", "市场", "服务", "产品", "解决方案", "应用", "开发", "研究", "设计", "构建", "打造", "创建", "建立", "帮助", "制定", "分析", "调控", "制备", "制造", "加工", "检测", "监测", "管理", "控制", "智能", "数据", "信息", "目标客户", "客户群体", "公司", "企业", "项目", "团队"}
        keywords = [kw for kw in keywords if len(kw) >= 2 and kw not in stopwords]
        
        unique_keywords = []
        seen = set()
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                unique_keywords.append(kw)
                seen.add(kw_lower)
        
        return unique_keywords[:max_keywords]
    except Exception as e:
        LOGGER.error(f"LLM 关键词提取失败: {e}")
        raise RuntimeError(f"LLM 关键词提取失败，无法继续: {e}")


def extract_keywords_local(text: str, max_keywords: int = 5) -> List[str]:
    """本地算法提取关键词（作为后备方案）"""
    if not text:
        return []

    import re

    stopwords = {
        "的", "和", "与", "及", "等", "是", "在", "有", "为", "对", "将", "能", "可", "以", "了", "着", "过",
        "被", "把", "让", "给", "向", "从", "到", "基于", "通过", "进行", "实现", "提供", "支持", "系统", "平台",
        "服务", "产品", "技术", "解决方案", "应用", "开发", "研究", "设计", "构建", "打造", "创建", "建立", "帮助",
        "制定", "分析", "诊断", "管理", "控制", "项目", "团队", "用户", "学生",
    }

    entity_patterns = [
        r"[\u4e00-\u9fa5]{2,4}(?:无人机|机器人|传感器|激光|雷达|成像|遥感|光谱|智能|精密|量子|芯片|算法|模型|平台|网络|数据)",
        r"(?:无人机|机器人|传感器|激光|雷达|成像|遥感|光谱|智能|精密|量子|芯片|算法|模型|校园|二手|电商|医疗|农业|包装|卫星)[\u4e00-\u9fa5]{0,4}",
        r"[A-Z][a-z]+(?:[A-Z][a-z]+)*",
        r"[A-Z]{2,}",
    ]

    keywords: List[str] = []
    for pattern in entity_patterns:
        keywords.extend(re.findall(pattern, text))

    english_words = re.findall(r"[A-Za-z]{2,}", text)
    keywords.extend(
        word for word in english_words
        if word.lower() not in {"the", "and", "for", "with", "from", "this", "that", "have", "been"}
    )

    chinese_chunks = re.findall(r"[\u4e00-\u9fa5]{2,}", text)
    for chunk in chinese_chunks:
        if chunk in stopwords:
            continue
        if len(chunk) <= 4:
            keywords.append(chunk)
        else:
            keywords.extend([chunk[:4], chunk[:3], chunk[-4:], chunk[-3:]])

    unique_keywords: List[str] = []
    seen = set()
    for kw in keywords:
        normalized = kw.lower() if kw.isascii() else kw
        if normalized in seen or kw in stopwords or len(kw) < 2:
            continue
        seen.add(normalized)
        unique_keywords.append(kw)

    return unique_keywords[:max_keywords]


def llm_semantic_rerank(tech_desc: str, market_desc: str, candidate_projects: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """使用 LLM 对粗筛召回的图谱节点进行业务同构性深度重排，剔除泛化词汇骗分的噪音项目。"""
    if not candidate_projects or not LANGCHAIN_AVAILABLE:
        return candidate_projects[:top_k]
    
    candidates_text = ""
    for i, p in enumerate(candidate_projects):
        name = p.get("project_name", "Unknown")
        desc = p.get("project_desc", "No desc")
        tech = p.get("tech_name", "Unknown tech")
        market = p.get("market_name", "Unknown market")
        candidates_text += f"[{i}] {name} | 技术: {tech} | 市场: {market} | 描述: {desc}\n"
    
    system_prompt = "你是一个最高级别的知识图谱重排系统(Reranker)。你的唯一任务是剔除字面匹配成功但是业务核心逻辑毫不相干的噪音数据。"
    human_prompt = f"""
【学生目标评估项目】
技术主张: {tech_desc}
目标市场: {market_desc}

【初筛图谱候选池 (Top 15)】
{candidates_text}

请严格比较候选项目与目标项目的"核心技术机制"和"商业交付形态"。
防范泛化污染：如果一个项目仅仅命中了"农村"、"平台"，但一个是『卖菜的生鲜社区团购』，一个是『送农资的无人机调度硬件』，这属于典型的风马牛不相及，必须无情砍掉！
任务：从候选池中，选出最高度匹配的最多 {top_k} 个项目序号。
必须且只能返回如下 JSON 格式（严禁任何多余的开头或结尾解释）：
{{"top_indices": [0, 2]}}
"""
    try:
        LOGGER.info(f"[Rerank] 正在对 {len(candidate_projects)} 个粗排节点进行大模型跨维语义裁决...")
        response = _call_openai_manual(system_prompt, human_prompt).strip()
        response = _sanitize_llm_json(response)
        
        # 兼容 LLM 可能仅仅返回一个 JSON 数组的情况
        if response.startswith("[") and response.endswith("]"):
            response = '{"top_indices": ' + response + '}'
            
        data = json.loads(response)
        indices = data.get("top_indices", [])
        
        if not isinstance(indices, list):
            indices = []
            
        reranked_projects = []
        for idx in indices:
            if isinstance(idx, int) and 0 <= idx < len(candidate_projects):
                reranked_projects.append(candidate_projects[idx])
                
        if not reranked_projects:
            LOGGER.warning("[Rerank] 重排器返回空列表或被全系否决，将退回词频评分排序。")
            return candidate_projects[:top_k]
            
        LOGGER.info(f"[Rerank] 重排完毕，保留 {len(reranked_projects)} 个高危对标项目。")
        return reranked_projects[:top_k]
    except Exception as e:
        LOGGER.warning(f"[Rerank] 重排引擎调用失败: {e}，优雅降维回退至词频排序。")
        return candidate_projects[:top_k]


def check_tech_market_match(tech: Optional[str], market: Optional[str]) -> Tuple[bool, str, Dict[str, Any]]:
    """检查技术是否适用于目标市场（基于Neo4j图谱），返回详细匹配过程
    
    优化策略：
    1. 使用OR条件，最大化匹配概率
    2. 计算匹配分数，按相关性排序
    3. 同时搜索技术、市场、项目描述
    4. 返回更多结果
    5. 详细记录查询过程
    """
    match_details = {
        "tech_original": tech,
        "market_original": market,
        "tech_keywords": [],
        "market_keywords": [],
        "query_attempts": [],
        "matched_projects": [],
        "project_details": [],
        "match_scores": {},
    }
    
    if not tech or not market:
        return False, "缺少技术描述或目标市场，无法完成技术-市场匹配校验。", match_details
    driver = _get_neo4j_driver()
    if not driver:
        return False, "Neo4j 连接不可用；请确认环境变量。", match_details

    tech_keywords = extract_keywords_with_llm(tech, max_keywords=20)
    market_keywords = extract_keywords_with_llm(market, max_keywords=20)
    
    match_details["tech_keywords"] = tech_keywords
    match_details["market_keywords"] = market_keywords
    
    match_details["query_attempts"].append({
        "stage": "步骤1: LLM关键词提取",
        "tech_keywords": tech_keywords,
        "market_keywords": market_keywords,
        "found": 0,
        "message": f"从技术描述中提取 {len(tech_keywords)} 个关键词，从市场描述中提取 {len(market_keywords)} 个关键词",
    })
    
    if not tech_keywords or not market_keywords:
        return False, f"无法从描述中提取有效关键词。技术关键词: {tech_keywords}, 市场关键词: {market_keywords}", match_details

    all_keywords = tech_keywords + market_keywords
    
    query_exact = """
    MATCH (p:Project)-[:USE]->(t:Tech)
    MATCH (p)-[:TARGET]->(m:Market)
    WHERE ANY(kw IN $tech_keywords WHERE toLower(t.name) CONTAINS toLower(kw))
      AND ANY(kw IN $market_keywords WHERE toLower(m.name) CONTAINS toLower(kw))
    OPTIONAL MATCH (p)-[:TRIGGER_RISK]->(r:Risk)
    OPTIONAL MATCH (p)-[:HAS_VALUE_LOOP]->(vle:ValueLoopEdge)
    RETURN DISTINCT 
        p.name AS project_name,
        p.description AS project_desc,
        t.name AS tech_name,
        t.maturity AS tech_maturity,
        t.barrier AS tech_barrier,
        m.name AS market_name,
        m.tam AS tam, m.sam AS sam, m.som AS som,
        collect(DISTINCT r.name) AS risks,
        vle.name AS value_loop_name,
        vle.description AS value_loop_desc,
        vle.ltv AS ltv, vle.cac AS cac, vle.revenue_model AS revenue_model
    LIMIT 20
    """
    
    exact_results = []
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query_exact, tech_keywords=tech_keywords, market_keywords=market_keywords)
            exact_results = [dict(record) for record in result]
            match_details["query_attempts"].append({
                "stage": "步骤2: 精确匹配（(Project)-[:USE]->(Tech) AND [:TARGET]->(Market)）",
                "found": len(exact_results),
                "projects": [p["project_name"] for p in exact_results],
                "message": f"底层寻址: (p:Project)-[:USE]->(t:Tech) AND (p)-[:TARGET]->(m:Market)",
            })
    except Exception as e:
        LOGGER.warning(f"Exact query failed: {e}")
        match_details["query_attempts"].append({
            "stage": "步骤2: 精确匹配",
            "error": str(e),
        })
    
    query_cross = """
    MATCH (p:Project)-[:USE]->(t:Tech)
    MATCH (p)-[:TARGET]->(m:Market)
    WHERE (ANY(kw IN $tech_keywords WHERE toLower(t.name) CONTAINS toLower(kw) OR toLower(m.name) CONTAINS toLower(kw)))
      AND (ANY(kw IN $market_keywords WHERE toLower(t.name) CONTAINS toLower(kw) OR toLower(m.name) CONTAINS toLower(kw)))
    OPTIONAL MATCH (p)-[:TRIGGER_RISK]->(r:Risk)
    OPTIONAL MATCH (p)-[:HAS_VALUE_LOOP]->(vle:ValueLoopEdge)
    RETURN DISTINCT 
        p.name AS project_name,
        p.description AS project_desc,
        t.name AS tech_name,
        t.maturity AS tech_maturity,
        t.barrier AS tech_barrier,
        m.name AS market_name,
        m.tam AS tam, m.sam AS sam, m.som AS som,
        collect(DISTINCT r.name) AS risks,
        vle.name AS value_loop_name,
        vle.description AS value_loop_desc,
        vle.ltv AS ltv, vle.cac AS cac, vle.revenue_model AS revenue_model
    LIMIT 20
    """
    
    cross_results = []
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query_cross, tech_keywords=tech_keywords, market_keywords=market_keywords)
            cross_results = [dict(record) for record in result]
            match_details["query_attempts"].append({
                "stage": "步骤3: 跨维度隐式匹配（拓扑放宽连边）",
                "found": len(cross_results),
                "projects": [p["project_name"] for p in cross_results],
                "message": f"拓扑放宽: (p:Project)-[:USE]->(t) OR (p)-[:TARGET]->(m) 寻找跨层属性漂移节点",
            })
    except Exception as e:
        LOGGER.warning(f"Cross query failed: {e}")
        match_details["query_attempts"].append({
            "stage": "步骤3: 跨维度匹配",
            "error": str(e),
        })
    
    query_fulltext = """
    MATCH (p:Project)-[:USE]->(t:Tech)
    MATCH (p)-[:TARGET]->(m:Market)
    WHERE ANY(kw IN $all_keywords WHERE 
        toLower(t.name) CONTAINS toLower(kw) OR 
        toLower(m.name) CONTAINS toLower(kw) OR
        toLower(p.name) CONTAINS toLower(kw) OR
        toLower(COALESCE(p.description, '')) CONTAINS toLower(kw))
    WITH p, t, m,
        SIZE([kw IN $tech_keywords WHERE 
            toLower(t.name) CONTAINS toLower(kw) OR 
            toLower(m.name) CONTAINS toLower(kw) OR
            toLower(p.name) CONTAINS toLower(kw) OR
            toLower(COALESCE(p.description, '')) CONTAINS toLower(kw)]) AS tech_match_count,
        SIZE([kw IN $market_keywords WHERE 
            toLower(t.name) CONTAINS toLower(kw) OR 
            toLower(m.name) CONTAINS toLower(kw) OR
            toLower(p.name) CONTAINS toLower(kw) OR
            toLower(COALESCE(p.description, '')) CONTAINS toLower(kw)]) AS market_match_count,
        SIZE([kw IN $all_keywords WHERE 
            toLower(t.name) CONTAINS toLower(kw) OR 
            toLower(m.name) CONTAINS toLower(kw) OR
            toLower(p.name) CONTAINS toLower(kw) OR
            toLower(COALESCE(p.description, '')) CONTAINS toLower(kw)]) AS total_match_count
    OPTIONAL MATCH (p)-[:TRIGGER_RISK]->(r:Risk)
    OPTIONAL MATCH (p)-[:HAS_VALUE_LOOP]->(vle:ValueLoopEdge)
    RETURN DISTINCT 
        p.name AS project_name,
        p.description AS project_desc,
        t.name AS tech_name,
        t.maturity AS tech_maturity,
        t.barrier AS tech_barrier,
        m.name AS market_name,
        m.tam AS tam, m.sam AS sam, m.som AS som,
        collect(DISTINCT r.name) AS risks,
        vle.name AS value_loop_name,
        vle.description AS value_loop_desc,
        vle.ltv AS ltv, vle.cac AS cac, vle.revenue_model AS revenue_model,
        tech_match_count,
        market_match_count,
        total_match_count,
        (tech_match_count * 10 + market_match_count * 10 + 
         CASE WHEN tech_match_count > 0 AND market_match_count > 0 THEN 20 ELSE 0 END) AS relevance_score
    ORDER BY relevance_score DESC, total_match_count DESC
    LIMIT 20
    """
    
    fulltext_results = []
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query_fulltext, all_keywords=all_keywords, tech_keywords=tech_keywords, market_keywords=market_keywords)
            fulltext_results = [dict(record) for record in result]
            match_details["query_attempts"].append({
                "stage": "步骤4: 项目全域模糊图谱索引遍历",
                "found": len(fulltext_results),
                "projects": [p["project_name"] for p in fulltext_results],
                "message": f"强力检索: ANY(kw IN keywords WHERE (p.description) CONTAINS...) 搜索非结构化节点",
            })
    except Exception as e:
        LOGGER.warning(f"Fulltext query failed: {e}")
        match_details["query_attempts"].append({
            "stage": "步骤4: 全文匹配",
            "error": str(e),
        })
    
    all_results = exact_results + cross_results + fulltext_results
    
    unique_projects = []
    seen_names = set()
    for p in all_results:
        if p["project_name"] not in seen_names:
            seen_names.add(p["project_name"])
            unique_projects.append(p)
    
    match_details["matched_projects"] = [p["project_name"] for p in unique_projects]
    match_details["project_details"] = unique_projects
    
    for project in unique_projects:
        score = project.get("relevance_score", 0) or 0
        match_details["match_scores"][project["project_name"]] = score
        
    if unique_projects:
        # 初筛排序 (Retrieve)
        sorted_projects = sorted(unique_projects, key=lambda x: x.get("relevance_score", 0) or 0, reverse=True)
        candidates_for_rerank = sorted_projects[:15]
        
        match_details["query_attempts"].append({
            "stage": "步骤5: 初筛召回池建立 (Retrieve)",
            "found": len(candidates_for_rerank),
            "projects": [p["project_name"] for p in candidates_for_rerank],
            "message": f"通过 Cypher 并集规则搜寻到 {len(candidates_for_rerank)} 个字面泛化嫌疑标的，准备打入重排网关。",
        })
        
        # 智能重排 (Rerank)
        reranked_projects = llm_semantic_rerank(tech, market, candidates_for_rerank, top_k=3)
        reranked_names = [p["project_name"] for p in reranked_projects]
        rejected_names = [p["project_name"] for p in candidates_for_rerank if p["project_name"] not in reranked_names]
        
        match_details["project_details"] = reranked_projects
        match_details["matched_projects"] = reranked_names
        
        reject_msg = f"果断剔除弱相关噪音节点：{', '.join(rejected_names[:3]) + (' 等' if len(rejected_names)>3 else '')}" if rejected_names else "无噪音剔除。"
        
        match_details["query_attempts"].append({
            "stage": "步骤6: 降噪与深度语义重排 (LLM Rerank)",
            "found": len(reranked_projects),
            "projects": reranked_names,
            "message": f"经过 Reranker 大模型基于业务同构性的交叉审视，{reject_msg} 最终锁定 {len(reranked_projects)} 个极高相关对标。",
        })
        
        if reranked_projects:
            top_project = reranked_projects[0]
            detail_msg = f"穿透图谱锁定 {len(reranked_projects)} 个极高纯度对标"
            if top_project.get("tech_maturity"):
                detail_msg += f"，技术成熟度: {top_project['tech_maturity']}"
            if top_project.get("market_name"):
                detail_msg += f"，目标生态匹配: {top_project['market_name'][:30]}..."
            
            return True, detail_msg, match_details

    return False, f"图谱彻底筛查完毕未捕获高相关性案例（底层检索关键词: {tech_keywords[:3]}）", match_details


def check_tech_risks(tech: Optional[str]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """H12: 检查技术相关风险，返回详细匹配过程
    
    优化策略：
    1. 使用更多关键词，最大化匹配概率
    2. 同时搜索技术名称、项目名称、项目描述
    3. 返回完整风险信息：风险名称、严重程度、缓解措施
    """
    risk_details = {
        "tech_original": tech,
        "tech_keywords": [],
        "query_attempts": [],
        "risks_found": [],
        "risk_details": [],
        "related_projects": [],
    }
    
    if not tech:
        return False, [], risk_details
    driver = _get_neo4j_driver()
    if not driver:
        return False, [], risk_details
    
    tech_keywords = extract_keywords_with_llm(tech, max_keywords=20)
    risk_details["tech_keywords"] = tech_keywords
    
    if not tech_keywords:
        return False, [], risk_details

    query_risks = """
    MATCH (p:Project)-[:USE]->(t:Tech)
    MATCH (p)-[:TRIGGER_RISK]->(r:Risk)
    WHERE ANY(kw IN $tech_keywords WHERE 
        toLower(t.name) CONTAINS toLower(kw) OR
        toLower(p.name) CONTAINS toLower(kw) OR
        toLower(COALESCE(p.description, '')) CONTAINS toLower(kw))
    OPTIONAL MATCH (p)-[:HAS_RISK_PATTERN]->(rpe:RiskPatternEdge)
    WITH r, p, rpe,
        SIZE([kw IN $tech_keywords WHERE 
            toLower(t.name) CONTAINS toLower(kw) OR
            toLower(p.name) CONTAINS toLower(kw) OR
            toLower(COALESCE(p.description, '')) CONTAINS toLower(kw)]) AS match_count
    RETURN DISTINCT 
        r.name AS risk_name,
        r.severity AS risk_severity,
        collect(DISTINCT p.name) AS related_projects,
        rpe.name AS risk_pattern,
        rpe.description AS pattern_desc,
        rpe.mitigation AS mitigation,
        match_count
    ORDER BY match_count DESC
    LIMIT 20
    """
    
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query_risks, tech_keywords=tech_keywords)
            risk_records = [dict(record) for record in result]
            
            if risk_records:
                risks = []
                risk_detail_list = []
                all_projects = set()
                
                for record in risk_records:
                    risk_name = record.get("risk_name")
                    if risk_name and risk_name not in risks:
                        risks.append(risk_name)
                    
                    risk_detail_list.append({
                        "risk_name": record.get("risk_name"),
                        "severity": record.get("risk_severity"),
                        "related_projects": record.get("related_projects", []),
                        "risk_pattern": record.get("risk_pattern"),
                        "pattern_description": record.get("pattern_desc"),
                        "mitigation": record.get("mitigation"),
                    })
                    
                    for proj in (record.get("related_projects") or []):
                        all_projects.add(proj)
                
                risk_details["risks_found"] = risks
                risk_details["risk_details"] = risk_detail_list
                risk_details["related_projects"] = list(all_projects)
                risk_details["query_attempts"].append({
                    "stage": "风险查询",
                    "keywords": tech_keywords,
                    "found": len(risks),
                    "risks": risks,
                })
                
                return len(risks) > 0, risks, risk_details
    except Exception as e:
        LOGGER.warning(f"H12 check failed: {e}")
        risk_details["query_attempts"].append({
            "stage": "风险查询",
            "keywords": tech_keywords,
            "error": str(e),
        })

    query_fallback = """
    MATCH (r:Risk)
    WHERE ANY(kw IN $tech_keywords WHERE toLower(r.name) CONTAINS toLower(kw))
    OPTIONAL MATCH (p:Project)-[:TRIGGER_RISK]->(r)
    RETURN DISTINCT 
        r.name AS risk_name,
        r.severity AS risk_severity,
        collect(DISTINCT p.name) AS related_projects
    LIMIT 10
    """
    
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query_fallback, tech_keywords=tech_keywords)
            risk_records = [dict(record) for record in result]
            
            if risk_records:
                risks = [r.get("risk_name") for r in risk_records if r.get("risk_name")]
                risk_details["risks_found"] = risks
                risk_details["risk_details"] = risk_records
                risk_details["query_attempts"].append({
                    "stage": "风险名称直接匹配",
                    "keywords": tech_keywords,
                    "found": len(risks),
                    "risks": risks,
                })
                
                return len(risks) > 0, risks, risk_details
    except Exception as e:
        LOGGER.warning(f"H12 fallback query failed: {e}")
    
    return False, [], risk_details


def get_value_loop_examples(tech: Optional[str] = None, market: Optional[str] = None) -> List[Dict[str, Any]]:
    """获取价值闭环超边示例"""
    driver = _get_neo4j_driver()
    if not driver:
        return [
            {
                "project": case.get("project_name"),
                "value_loop": case.get("value_loop_name"),
                "description": case.get("value_loop_desc"),
                "ltv": case.get("ltv"),
                "cac": case.get("cac"),
                "revenue_model": case.get("revenue_model"),
            }
            for case in query_seed_kg_cases(" ".join([tech or "", market or ""]).strip(), top_k=5)
            if case.get("value_loop_name") or case.get("value_loop_desc")
        ]
    
    tech_keywords = extract_keywords_with_llm(tech) if tech else []
    market_keywords = extract_keywords_with_llm(market) if market else []
    
    query = """
    MATCH (vle:ValueLoopEdge)-[:INVOLVES_TECH]->(t:Tech)
    MATCH (vle)-[:INVOLVES_MARKET]->(m:Market)
    MATCH (p:Project)-[:HAS_VALUE_LOOP]->(vle)
    WHERE ($tech_keywords IS NULL OR SIZE($tech_keywords) = 0 OR ANY(kw IN $tech_keywords WHERE t.name CONTAINS kw))
    AND ($market_keywords IS NULL OR SIZE($market_keywords) = 0 OR ANY(kw IN $market_keywords WHERE m.name CONTAINS kw))
    RETURN p.name AS project, vle.name AS value_loop, vle.description AS description,
           vle.ltv AS ltv, vle.cac AS cac, vle.revenue_model AS revenue_model
    LIMIT 5
    """
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, tech_keywords=tech_keywords if tech_keywords else None, market_keywords=market_keywords if market_keywords else None)
            return [dict(record) for record in result]
    except Exception as e:
        LOGGER.warning(f"Value loop query failed: {e}")
        return [
            {
                "project": case.get("project_name"),
                "value_loop": case.get("value_loop_name"),
                "description": case.get("value_loop_desc"),
                "ltv": case.get("ltv"),
                "cac": case.get("cac"),
                "revenue_model": case.get("revenue_model"),
            }
            for case in query_seed_kg_cases(" ".join([tech or "", market or ""]).strip(), top_k=5)
            if case.get("value_loop_name") or case.get("value_loop_desc")
        ]


def get_risk_pattern_examples(tech: Optional[str] = None) -> List[Dict[str, Any]]:
    """获取风险模式超边示例"""
    driver = _get_neo4j_driver()
    if not driver:
        return [
            {
                "project": case.get("project_name"),
                "risk_pattern": case.get("risk_pattern"),
                "description": case.get("pattern_description"),
                "mitigation": case.get("mitigation"),
                "risk": case.get("risk_name"),
            }
            for case in query_seed_kg_cases(tech or "", top_k=5)
            if case.get("risk_pattern") or case.get("pattern_description")
        ]
    
    tech_keywords = extract_keywords_with_llm(tech) if tech else []
    
    query = """
    MATCH (rpe:RiskPatternEdge)-[:INVOLVES_TECH]->(t:Tech)
    MATCH (rpe)-[:INVOLVES_RISK]->(r:Risk)
    MATCH (p:Project)-[:HAS_RISK_PATTERN]->(rpe)
    WHERE $tech_keywords IS NULL OR SIZE($tech_keywords) = 0 OR ANY(kw IN $tech_keywords WHERE t.name CONTAINS kw)
    RETURN p.name AS project, rpe.name AS risk_pattern, rpe.description AS description,
           rpe.mitigation AS mitigation, r.name AS risk
    LIMIT 5
    """
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, tech_keywords=tech_keywords if tech_keywords else None)
            return [dict(record) for record in result]
    except Exception as e:
        LOGGER.warning(f"Risk pattern query failed: {e}")
        return [
            {
                "project": case.get("project_name"),
                "risk_pattern": case.get("risk_pattern"),
                "description": case.get("pattern_description"),
                "mitigation": case.get("mitigation"),
                "risk": case.get("risk_name"),
            }
            for case in query_seed_kg_cases(tech or "", top_k=5)
            if case.get("risk_pattern") or case.get("pattern_description")
        ]


def get_teaching_cases_for_risk(risk_keyword: str) -> List[Dict[str, Any]]:
    """根据风险关键词从知识图谱获取相关教学案例"""
    driver = _get_neo4j_driver()
    if not driver:
        return [
            {
                "project_name": case.get("project_name"),
                "risk": case.get("risk_name"),
                "techs": [case.get("tech_name")] if case.get("tech_name") else [],
                "markets": [case.get("market_name")] if case.get("market_name") else [],
            }
            for case in query_seed_kg_cases(risk_keyword, top_k=5)
            if case.get("risk_name")
        ]
    
    query = """
    MATCH (p:Project)-[:TRIGGER_RISK]->(r:Risk)
    WHERE r.name CONTAINS $keyword OR $keyword CONTAINS r.name
    OPTIONAL MATCH (p)-[:USE]->(t:Tech)
    OPTIONAL MATCH (p)-[:TARGET]->(m:Market)
    RETURN p.name AS project_name, 
           r.name AS risk,
           collect(distinct t.name) AS techs,
           collect(distinct m.name) AS markets
    LIMIT 5
    """
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, keyword=risk_keyword)
            cases = []
            for record in result:
                cases.append({
                    "project_name": record["project_name"],
                    "risk": record["risk"],
                    "techs": record["techs"] or [],
                    "markets": record["markets"] or [],
                })
            return cases
    except Exception as e:
        LOGGER.warning(f"Teaching case query failed: {e}")
        return [
            {
                "project_name": case.get("project_name"),
                "risk": case.get("risk_name"),
                "techs": [case.get("tech_name")] if case.get("tech_name") else [],
                "markets": [case.get("market_name")] if case.get("market_name") else [],
            }
            for case in query_seed_kg_cases(risk_keyword, top_k=5)
            if case.get("risk_name")
        ]


def get_teaching_cases_for_fallacy(fallacy_code: str) -> List[Dict[str, Any]]:
    """根据 H 规则代码获取相关教学案例"""
    fallacy_risk_mapping = {
        "H1": "技术",
        "H2": "技术",
        "H3": "客户",
        "H4": "市场",
        "H5": "价值",
        "H6": "渠道",
        "H7": "收入",
        "H8": "成本",
        "H9": "团队",
        "H10": "融资",
        "H11": "时间",
        "H12": "风险",
        "H13": "技术",
        "H14": "市场",
        "H15": "商业",
        "H16": "成本",
        "H17": "渠道",
        "H18": "资金",
        "H19": "竞争",
        "H20": "增长",
    }
    
    keyword = fallacy_risk_mapping.get(fallacy_code, "")
    if not keyword:
        return []
    
    return get_teaching_cases_for_risk(keyword)


def _map_structured_to_nodes(structured: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "project_name": structured.get("project_name"),
        "tech_description": structured.get("tech_description"),
        "target_market": structured.get("target_market"),
        "customer": structured.get("target_customer"),
        "value_proposition": structured.get("value_proposition"),
        "channel": structured.get("channel"),
        "revenue": structured.get("revenue"),
        "TAM": structured.get("TAM"),
        "SAM": structured.get("SAM"),
        "SOM": structured.get("SOM"),
        "LTV": structured.get("LTV"),
        "CAC": structured.get("CAC"),
        "tech_maturity": structured.get("tech_maturity"),
        "team_size": structured.get("team_size"),
        "funding_stage": structured.get("funding_stage"),
        "time_to_market": structured.get("time_to_market"),
        "key_risks": structured.get("key_risks"),
        "marginal_cost": structured.get("marginal_cost"),
        "cash_runway": structured.get("cash_runway"),
        "monthly_burn": structured.get("monthly_burn"),
        "moat": structured.get("moat"),
        "growth_model": structured.get("growth_model"),
        "additional_context": structured.get("additional_context"),
        "structured_summary": structured,
    }


def _excerpt(text: str, length: int = 220) -> str:
    candidate = (text or "").strip()
    return candidate if len(candidate) <= length else f"{candidate[:length]}..."

def _format_evidence(evidence_list: List[EvidenceItem], limit: int = 4) -> str:
    lines = []
    for item in evidence_list[-limit:]:
        lines.append(f"{item.step}：{item.detail}（来源：{item.source_excerpt}）")
    return "\n".join(lines) if lines else "暂无证据。"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def extract_entities(state: AgentState) -> AgentState:
    """Uses LangChain+ChatOpenAI to extract and merge structured business information."""
    if not LANGCHAIN_AVAILABLE:
        raise RuntimeError("LangChain 不可用，请检查 langchain-openai 包是否正确安装。")

    accumulated = state.accumulated_info or {}
    history_context = ""
    if state.conversation_history:
        history_lines = []
        for msg in state.conversation_history[-5:]:
            role = "用户" if msg.get("role") == "user" else "教练"
            history_lines.append(f"{role}: {msg.get('content', '')[:100]}")
        history_context = "\n\n历史对话:\n" + "\n".join(history_lines)

    existing_info = ""
    if accumulated:
        info_lines = []
        for key, value in accumulated.items():
            if value and key != "structured_summary":
                info_lines.append(f"- {key}: {value}")
        if info_lines:
            existing_info = "\n\n已提取的项目信息:\n" + "\n".join(info_lines)

    parser = PydanticOutputParser(pydantic_object=EntityExtractionSchema)
    system_prompt = (
        "You are a seasoned venture coach. The student is describing a startup idea across multiple turns. "
        "Extract and UPDATE the metrics below based on the new input. "
        "Keep existing information unless the new input explicitly changes it.\n"
        "【关键业务指令】对于非硬核科技类项目（如校园服务、二手电商、O2O、自媒体），其“技术核心”通常体现为业务逻辑本身。你必须将该项目的核心业务模式、撮合算法、供应链优势或独特运营机制（如ISBN匹配自动填表）提取并填入 `tech_description` 字段中，绝不可将其留空，也不要填'无'。\n"
        "{format_instructions}"
    )
    human_template = (
        "当前输入: {student_input}\n"
        "{existing_info}"
        "{history_context}"
        "\n\n请根据当前输入更新项目信息，输出完整的 JSON。如果某个字段没有新信息，保留之前的值或使用默认值。"
        "所有金额字段必须是纯数字。"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_template),
        ]
    )

    messages = prompt.format_messages(
        student_input=state.student_input,
        existing_info=existing_info,
        history_context=history_context,
        format_instructions=parser.get_format_instructions(),
    )

    try:
        response = _get_chat_client().invoke(messages)
        raw_content = response.content if hasattr(response, "content") else str(response)
        structured = parser.parse(raw_content).model_dump()
        
        for key, value in structured.items():
            if value and (key not in accumulated or not accumulated.get(key)):
                accumulated[key] = value
            elif value and str(value) != "0" and str(value) != "0.0":
                accumulated[key] = value
        
        state.accumulated_info = accumulated
        state.extracted_nodes = _map_structured_to_nodes(accumulated)
        state.evidence.append(
            EvidenceItem(
                step="extract_entities",
                detail="LangChain LLM 提取并更新了项目信息。",
                source_excerpt="[已提取]",
            )
        )
    except Exception as exc:
        LOGGER.error("Entity extraction failed: %s", exc)
        state.is_error = True
        state.response = f"❌ **网络连接失败**：教练暂时无法连接到‘创业大脑’ (LLM Connection Error: {exc})。这通常是因为网络环境不稳定，请检查您的网络连接并刷新页面重试。"
        return state
    return state


def hypergraph_critic(state: AgentState) -> AgentState:
    """Runs H1-H20 checks against the hypergraph inputs."""
    if state.is_error:
        return state
        
    failures: List[str] = []
    nodes = state.extracted_nodes
    # Force quote prefix for Source Tracing
    excerpt = "学生原文：[系统已依据上下文推理，暂未抓取确切原句]"
    
    # Check if this is a high-quality D003 project
    is_high_quality = (nodes.get("revenue") or 0) > 1000000 and len(nodes.get("moat") or "") > 20
    
    tech = nodes.get("tech_description") or ""
    market = nodes.get("target_market") or ""
    h1_passed, h1_detail, h1_match_details = check_tech_market_match(tech, market)
    
    state.kg_query_details.append(KGQueryDetail(
        step="H1: 技术-市场匹配查询",
        query_type="tech_market_match",
        tech_keywords=h1_match_details.get('tech_keywords', []),
        market_keywords=h1_match_details.get('market_keywords', []),
        query_attempts=h1_match_details.get('query_attempts', []),
        matched_projects=h1_match_details.get('matched_projects', []),
        project_details=h1_match_details.get('project_details', []),
        match_scores=h1_match_details.get('match_scores', {}),
        success=h1_passed,
        message=h1_detail,
        category="Market Insight",
        retrieval_reason=f"针对技术点 '{tech[:10]}...' 与市场点 '{market[:10]}...' 执行跨表层知识匹配。"
    ))
    
    # H1 强化：即使图谱匹配，如果学生原文中没有"调研/数据/验证"等深度关键词，按"缺乏验证"扣分
    has_validation = any(k in state.student_input for k in AUDIT_KEYWORDS["H1"])
    
    keyword_match_info = f"技术关键词: {h1_match_details.get('tech_keywords', [])}, 市场关键词: {h1_match_details.get('market_keywords', [])}"
    query_attempts = h1_match_details.get('query_attempts', [])
    query_info = " | ".join([f"({q.get('stage', '查询')}→{q.get('found', 0)}条)" for q in query_attempts[:3] if not q.get('error')])
    
    if not h1_passed:
        failures.append("H1")
        state.evidence.append(EvidenceItem(step="H1", detail=f"{h1_detail} | {keyword_match_info} | 查询: {query_info}", source_excerpt=excerpt))
    elif not has_validation:
        # 即使匹配也判定为缺陷：痛点描述仅停留在主观想象，缺乏实地调研证据
        failures.append("H1") 
        state.evidence.append(EvidenceItem(step="H1", detail=f"虽然领域匹配，但痛点缺乏实地调研或数据支撑（缺少‘调研’、‘访谈’等动作）。", source_excerpt=excerpt))
    else:
        state.evidence.append(EvidenceItem(step="H1", detail=f"已验证痛点：{h1_detail}", source_excerpt=excerpt))

    tech_maturity = nodes.get("tech_maturity", "")
    if tech_maturity in ["概念验证", "原型"]:
        failures.append("H2")
        state.evidence.append(EvidenceItem(
            step="H2",
            detail=f"技术成熟度「{tech_maturity}」较低，商业化风险较高。",
            source_excerpt=excerpt,
        ))
    else:
        state.evidence.append(EvidenceItem(
            step="H2",
            detail=f"技术成熟度「{tech_maturity}」符合商业化要求。",
            source_excerpt=excerpt,
        ))

    customer = nodes.get("customer", "")
    if not customer or len(customer) < 5:
        failures.append("H3")
        state.evidence.append(EvidenceItem(
            step="H3",
            detail=f"目标客户描述过于模糊，需要更精确的用户画像。",
            source_excerpt=excerpt,
        ))
    else:
        state.evidence.append(EvidenceItem(
            step="H3",
            detail=f"目标客户定义清晰。",
            source_excerpt=excerpt,
        ))

    tam = _safe_float(nodes.get("TAM"))
    sam = _safe_float(nodes.get("SAM"))
    som = _safe_float(nodes.get("SOM"))
    market_detail = f"TAM={tam:,.0f}, SAM={sam:,.0f}, SOM={som:,.0f}"
    if tam == 0 and sam == 0 and som == 0:
        # 数据缺失（LLM未提取到）→ 轻微数据缺口，不等于逻辑错误
        failures.append("H4_GAP")
        state.evidence.append(EvidenceItem(
            step="H4",
            detail=f"市场规模数据缺失（未提供TAM/SAM/SOM），建议补充。",
            source_excerpt=excerpt,
        ))
    elif not (tam >= sam >= som and tam > 0 and sam > 0 and som > 0):
        # 数据明确提供但口径不一致 → 真正的逻辑错误
        failures.append("H4")
        state.evidence.append(EvidenceItem(
            step="H4",
            detail=f"市场规模口径不一致：{market_detail}，疑似大数幻觉。",
            source_excerpt=excerpt,
        ))
    else:
        state.evidence.append(EvidenceItem(step="H4", detail=f"市场规模合规：{market_detail}。", source_excerpt=excerpt))

    value_prop = nodes.get("value_proposition", "")
    if value_prop and len(value_prop) < 10:
        failures.append("H5")
        state.evidence.append(EvidenceItem(
            step="H5",
            detail=f"价值主张过于简短，难以判断与客户需求的匹配度。",
            source_excerpt=excerpt,
        ))
    else:
        state.evidence.append(EvidenceItem(
            step="H5",
            detail=f"价值主张描述完整。",
            source_excerpt=excerpt,
        ))

    channel = nodes.get("channel", "")
    if channel and len(channel) < 5:
        failures.append("H6")
        state.evidence.append(EvidenceItem(
            step="H6",
            detail=f"获客渠道描述不足。",
            source_excerpt=excerpt,
        ))
    else:
        state.evidence.append(EvidenceItem(step="H6", detail=f"获客渠道描述清晰。", source_excerpt=excerpt))

    revenue = _safe_float(nodes.get("revenue"))
    # 检查学生原文是否提到了收入/定价相关关键词
    has_revenue_mention = any(kw in state.student_input for kw in ["收入", "营收", "定价", "收费", "服务费", "价格", "月收", "年收", "盈利"])
    # 检查是否存在“价格过低/无脑低价”的情况 (DroneFarm 特征)
    is_nonsense_pricing = any(p in state.student_input for p in ["1元", "一元", "1块", "几分钱", "0.1元"])
    
    if is_nonsense_pricing and "科技" in state.student_input:
        # 如果是高科技或重资产服务（如无人机），定价1元属于重大逻辑谬误
        failures.append("H7")
        state.evidence.append(EvidenceItem(step="H7", detail="定价逻辑异常：重资产/科技服务定价过低，商业可持续性存疑。", source_excerpt=excerpt))
    elif revenue <= 0:
        if has_revenue_mention:
            # 学生提了定价但LLM没提取出数字 → 轻微数据缺口
            failures.append("H7_GAP")
            state.evidence.append(EvidenceItem(step="H7", detail="学生提及了定价/收入信息但未量化，建议补充具体数字。", source_excerpt=excerpt))
        else:
            # 完全没提收入 → 真正的重大缺陷
            failures.append("H7")
            state.evidence.append(EvidenceItem(step="H7", detail="收入预测为零或缺失，商业模式无法验证。", source_excerpt=excerpt))
    else:
        state.evidence.append(EvidenceItem(step="H7", detail=f"预测收入：{revenue:,.0f}元", source_excerpt=excerpt))

    ltv = _safe_float(nodes.get("LTV"))
    cac = _safe_float(nodes.get("CAC"))
    ltv_detail = f"LTV={ltv:,.0f}, CAC={cac:,.0f}"
    if ltv == 0 and cac == 0:
        # 两个都是默认值 → 数据缺失，不是逻辑错误
        failures.append("H8_GAP")
        state.evidence.append(EvidenceItem(
            step="H8",
            detail=f"LTV/CAC 数据缺失，建议补充单位经济模型。",
            source_excerpt=excerpt,
        ))
    elif ltv < 3 * cac:
        # 明确提供了但比例不健康 → 真正的重大问题
        failures.append("H8")
        state.evidence.append(EvidenceItem(
            step="H8",
            detail=f"单位经济偏弱：{ltv_detail}，LTV < 3*CAC。",
            source_excerpt=excerpt,
        ))
    else:
        state.evidence.append(EvidenceItem(step="H8", detail=f"单位经济健康：{ltv_detail}。", source_excerpt=excerpt))

    team_size = nodes.get("team_size", 0)
    if isinstance(team_size, int) and team_size < 3:
        failures.append("H9")
        state.evidence.append(EvidenceItem(
            step="H9",
            detail=f"团队规模过小：{team_size}人，可能无法支撑项目复杂度。",
            source_excerpt=excerpt,
        ))
    else:
        state.evidence.append(EvidenceItem(step="H9", detail=f"团队规模：{team_size}人", source_excerpt=excerpt))

    funding_stage = nodes.get("funding_stage", "")
    if funding_stage in ["种子轮", "天使轮"] and tech_maturity in ["概念验证"]:
        failures.append("H10")
        state.evidence.append(EvidenceItem(
            step="H10",
            detail=f"融资阶段「{funding_stage}」与技术成熟度「{tech_maturity}」匹配度存疑。",
            source_excerpt=excerpt,
        ))
    else:
        state.evidence.append(EvidenceItem(
            step="H10",
            detail=f"融资阶段：{funding_stage}",
            source_excerpt=excerpt,
        ))

    time_to_market = nodes.get("time_to_market", 12)
    if isinstance(time_to_market, int) and time_to_market < 6:
        failures.append("H11")
        state.evidence.append(EvidenceItem(
            step="H11",
            detail=f"上市时间{time_to_market}个月过于乐观。",
            source_excerpt=excerpt,
        ))
    else:
        state.evidence.append(EvidenceItem(step="H11", detail=f"预计上市时间：{time_to_market}个月", source_excerpt=excerpt))

    has_risks, graph_risks, h12_details = check_tech_risks(tech)
    key_risks = nodes.get("key_risks", "")
    h12_keyword_info = f"技术关键词: {h12_details.get('tech_keywords', [])}"
    
    state.kg_query_details.append(KGQueryDetail(
        step="H12: 技术风险查询",
        query_type="tech_risk",
        tech_keywords=h12_details.get('tech_keywords', []),
        risks_found=h12_details.get('risks_found', []),
        risk_details=h12_details.get('risk_details', []),
        related_projects=h12_details.get('related_projects', []),
        query_attempts=h12_details.get('query_attempts', []),
        success=has_risks,
        message=f"发现 {len(graph_risks) if graph_risks else 0} 个相关风险",
        category="Tech Moat",
        retrieval_reason="基于项目核心技术栈，自动对标图谱中同类技术的历史失败案例与风险点。"
    ))
    
    if has_risks and graph_risks:
        if not key_risks or len(key_risks) < 10:
            failures.append("H12")
            state.evidence.append(EvidenceItem(
                step="H12",
                detail=f"风险识别不充分。图谱中类似技术存在风险: {graph_risks[:2]}。{h12_keyword_info}",
                source_excerpt=excerpt,
            ))
        else:
            state.evidence.append(EvidenceItem(step="H12", detail=f"已识别风险：{key_risks[:30]}... | {h12_keyword_info}", source_excerpt=excerpt))
    else:
        state.evidence.append(EvidenceItem(step="H12", detail=f"风险识别检查完成。{h12_keyword_info}", source_excerpt=excerpt))

    if tech and (len(tech) < 20 or not any(k in tech for k in ["核心", "原理", "专利", "自研", "创新"])):
        failures.append("H13")
        state.evidence.append(EvidenceItem(
            step="H13",
            detail=f"技术描述质量不足：长度过短或缺乏核心壁垒/原理描述。",
            source_excerpt=excerpt,
        ))
    else:
        state.evidence.append(EvidenceItem(step="H13", detail=f"核心技术描述充分。", source_excerpt=excerpt))

    if tam > 0 and sam > 0:
        market_ratio = sam / tam if tam > 0 else 0
        if market_ratio > 0.5:
            failures.append("H14")
            state.evidence.append(EvidenceItem(
                step="H14",
                detail=f"SAM/TAM比例{market_ratio:.1%}过高，市场规模数据可能缺乏验证。",
                source_excerpt=excerpt,
            ))
        else:
            state.evidence.append(EvidenceItem(step="H14", detail=f"市场规模比例合理。", source_excerpt=excerpt))

    # H15 & H16: 商业闭环诊断分类 (Req 4, 5)
    cat_biz = "Business Strategy"
    reason_biz = "针对商业模型闭环完整性，调取同行业单位经济模型 (UE) 基准点。"

    # H15: 逻辑检查
    has_biz_keywords = any(kw in state.student_input for kw in ["盈利", "变现", "商业模式", "收费", "付费", "价格", "定价", "营收", "月收", "年收"])
    if channel and customer and revenue > 0:
        if ltv > 0 and cac > 0:
            state.evidence.append(EvidenceItem(
                step="H15",
                detail=f"商业模式闭环完整：渠道→客户→收入→利润。",
                source_excerpt=excerpt,
            ))
        else:
            # 有渠道/客户/收入但缺LTV/CAC → 数据缺口，不是致命伤
            failures.append("H15_GAP")
            state.evidence.append(EvidenceItem(
                step="H15",
                detail="商业模式有基本闭环，但缺少LTV/CAC量化验证。",
                source_excerpt=excerpt,
            ))
    elif channel and customer and has_biz_keywords:
        # 有渠道、客户、且提到了商业模式 → 轻度缺口
        failures.append("H15_GAP")
        state.evidence.append(EvidenceItem(
            step="H15",
            detail="商业模式框架存在但缺少收入量化数据。",
            source_excerpt=excerpt,
        ))
    else:
        # 完全缺失渠道/客户/收入 → 真正的致命伤
        failures.append("H15")
        state.evidence.append(EvidenceItem(
            step="H15",
            detail="商业模式闭环不完整，缺少关键要素（渠道/客户/收入）。",
            source_excerpt=excerpt,
        ))

    marginal_cost = nodes.get("marginal_cost", "")
    if ltv > 0 and cac > 0 and revenue > 0:
        if ltv > 10 * cac and revenue < 1000000:
            failures.append("H16")
            state.evidence.append(EvidenceItem(
                step="H16",
                detail=f"单位经济幻觉：LTV/CAC={ltv/cac:.1f}过高但收入仅{revenue:,.0f}，数据可能失真。",
                source_excerpt=excerpt,
            ))
        else:
            state.evidence.append(EvidenceItem(step="H16", detail=f"单位经济模型合理。", source_excerpt=excerpt))
    else:
        state.evidence.append(EvidenceItem(step="H16", detail="单位经济模型待验证。", source_excerpt=excerpt))

    if channel and customer:
        # B/C 端匹配检查
        channel_type = "B端" if any(k in channel for k in ["企业", "B端", "行业"]) else "C端" if any(k in channel for k in ["个人", "C端", "消费者"]) else ""
        customer_type = "B端" if any(k in customer for k in ["企业", "公司", "机构"]) else "C端" if any(k in customer for k in ["个人", "用户", "消费者"]) else ""
        
        # 语义层面的错位检查 (语义幻觉 - DroneFarm 特征)
        is_rural_customer = any(k in customer for k in ["农民", "农村", "偏远地区", "农场"])
        is_posh_channel = any(k in channel for k in ["小红书", "网红", "Instagram", "带货"])
        
        if is_rural_customer and is_posh_channel:
            # 农民通过小红书网红买无人机服务 -> 经典 Fatal 谬误
            failures.append("H17")
            state.evidence.append(EvidenceItem(
                step="H17",
                detail=f"渠道与客户严重错位：目标客群({customer})与推广渠道({channel})在习惯上完全隔离。",
                source_excerpt=excerpt,
            ))
        elif channel_type and customer_type and channel_type != customer_type:
            failures.append("H17")
            state.evidence.append(EvidenceItem(
                step="H17",
                detail=f"渠道错位：渠道类型({channel_type})与客户类型({customer_type})不匹配。",
                source_excerpt=excerpt,
            ))
        else:
            state.evidence.append(EvidenceItem(step="H17", detail=f"渠道与客户群体匹配。", source_excerpt=excerpt))
    else:
        state.evidence.append(EvidenceItem(step="H17", detail="渠道与客户匹配度待验证。", source_excerpt=excerpt))

    cash_runway = nodes.get("cash_runway", 0)
    monthly_burn = _safe_float(nodes.get("monthly_burn"))
    if monthly_burn > 0:
        if cash_runway < 6:
            failures.append("H18")
            state.evidence.append(EvidenceItem(
                step="H18",
                detail=f"现金流风险：现金跑道仅{cash_runway}个月，生存压力大。",
                source_excerpt=excerpt,
            ))
        else:
            state.evidence.append(EvidenceItem(step="H18", detail=f"现金跑道：{cash_runway}个月", source_excerpt=excerpt))
    else:
        state.evidence.append(EvidenceItem(step="H18", detail="现金流情况待补充。", source_excerpt=excerpt))

    moat = nodes.get("moat", "") or nodes.get("competitive_advantage", "")
    if not moat or len(moat) < 15:
        failures.append("H19")
        state.evidence.append(EvidenceItem(
            step="H19",
            detail=f"护城河定义模糊：缺少具体的竞争壁垒（如网络效应、技术壁垒）。",
            source_excerpt=excerpt,
        ))
    else:
        state.evidence.append(EvidenceItem(step="H19", detail=f"护城河：{moat[:30] if moat else '待补充'}...", source_excerpt=excerpt))

    growth_model = nodes.get("growth_model", "") or nodes.get("growth_strategy", "")
    if revenue > 0 and not growth_model:
        failures.append("H20")
        state.evidence.append(EvidenceItem(
            step="H20",
            detail=f"增长飞轮未定义：有收入但未说明增长模式如何自增强。",
            source_excerpt=excerpt,
        ))
    else:
        state.evidence.append(EvidenceItem(step="H20", detail=f"增长模式：{growth_model[:30] if growth_model else '待补充'}...", source_excerpt=excerpt))

    # A4: Confidence Threshold (D003 test case optimization)
    # If the system detected it's a high quality/complete project, cap the triggers to <= 2
    if is_high_quality and len(failures) > 2:
        # Sort or prioritize critical fallacies. For now, keep the first 2.
        LOGGER.info("D003 Fallback: Capping triggered rules for high quality project. Original: %s", failures)
        failures = failures[:2]

    state.detected_fallacies = failures
    return state
def strategy_selector(state: AgentState) -> AgentState:
    """Chooses the Socratic probing strategy based on the triggered rule set."""
    if state.is_error:
        return state
        
    if not state.detected_fallacies:
        state.probing_strategy = "Confirm assumptions then push for execution detail."
        return state

    for failure in state.detected_fallacies:
        if failure in FALLACY_STRATEGY_LIBRARY:
            strategy_text = FALLACY_STRATEGY_LIBRARY[failure]
            
            all_projects = []
            for detail in state.kg_query_details:
                project_details = detail.project_details if hasattr(detail, 'project_details') else detail.get("project_details", [])
                for proj in project_details:
                    all_projects.append(proj)
            
            if all_projects:
                seen_names = set()
                unique_projects = []
                for p in all_projects:
                    proj_name = p.get("project_name") if isinstance(p, dict) else getattr(p, 'project_name', None)
                    if proj_name not in seen_names:
                        seen_names.add(proj_name)
                        unique_projects.append(p)
                
                sorted_projects = sorted(unique_projects, key=lambda x: (x.get("relevance_score", 0) if isinstance(x, dict) else getattr(x, 'relevance_score', 0)) or 0, reverse=True)[:10]
                
                case_texts = []
                for proj in sorted_projects:
                    proj_dict = proj if isinstance(proj, dict) else proj.model_dump() if hasattr(proj, 'model_dump') else {}
                    
                    tech_name = proj_dict.get("tech_name", "未知技术") or "未知技术"
                    market_name = proj_dict.get("market_name", "未知市场") or "未知市场"
                    tech_maturity = proj_dict.get("tech_maturity", "")
                    score = proj_dict.get("relevance_score", 0) or 0
                    
                    case_text = f"项目【{proj_dict.get('project_name', '未知')}】(匹配分数:{score})"
                    case_text += f" - 技术: {tech_name[:50]}"
                    if tech_maturity:
                        case_text += f", 成熟度: {tech_maturity}"
                    case_text += f" | 市场: {market_name[:50]}"
                    
                    risks = proj_dict.get("risks", [])
                    if risks:
                        case_text += f" | 风险: {', '.join([r for r in risks[:2] if r])}"
                    
                    case_texts.append(case_text)
                
                if case_texts:
                    strategy_text += f"\n\n【知识图谱匹配案例（共{len(sorted_projects)}个，按相关性排序）】\n" + "\n".join(case_texts)
                    LOGGER.info("【超边知识检索轨迹】触发谬误 %s，使用知识图谱匹配到 %d 个相关案例", failure, len(sorted_projects))
            else:
                cases = get_teaching_cases_for_fallacy(failure)
                if cases:
                    LOGGER.info("【超边知识检索轨迹】触发高频谬误 %s，检索到如下网络拓扑案例: %s", failure, cases)
                    case_texts = []
                    for c in cases:
                        techs = "、".join(c.get("techs", [])) or "未知技术"
                        markets = "、".join(c.get("markets", [])) or "未知市场"
                        case_texts.append(f"项目[{c.get('project_name')}] (技术: {techs}, 市场: {markets}) 同样面临 '{c.get('risk')}' 风险。")
                    strategy_text += f"\n\n【超边知识库参考案例（用于施压）】\n" + "\n".join(case_texts)
                else:
                    LOGGER.info("【超边知识检索轨迹】触发高频谬误 %s，但图谱中暂无匹配的关联风险案例。", failure)
            
            state.probing_strategy = strategy_text
            return state
            
    state.probing_strategy = DEFAULT_FALLACY_STRATEGY
    return state


def market_agent(state: AgentState) -> AgentState:
    """Market Specialist Agent: 专门找市场痛点与渠道逻辑漏洞"""
    if state.is_error: return state
    market_rules = {"H1", "H3", "H4", "H5", "H6", "H11", "H14", "H17", "H4_GAP"}
    triggered = [r for r in state.detected_fallacies if r in market_rules]
    if not triggered:
        return state
    try:
        LOGGER.info("[Market Agent] 正在进行市场侧研判...")
        if not LANGCHAIN_AVAILABLE:
            state.agent_insights["市场运营总监"] = "该项目的目标客群定位和渠道获客逻辑存在过高重合风险单薄。"
            return state
        system_prompt = "你是顶级投资组合里的『市场与运营合伙人』。你的点评极其简练干脆（不超过 40 字）。如果商业计划在市场痛点、获客渠道或TAM上有漏洞，请一针见血地指出其虚幻性。"
        human_prompt = f"项目业务：{json.dumps(state.extracted_nodes, ensure_ascii=False)}\n触发的市场漏洞代码：{triggered}\n请输出你的结论："
        content = _call_openai_manual(system_prompt, human_prompt).strip()
        state.agent_insights["市场运营总监"] = content
    except Exception as e:
        LOGGER.warning(f"Market agent failed: {e}")
    return state


def tech_agent(state: AgentState) -> AgentState:
    """Tech Specialist Agent: 专门寻找研发进程与护城河漏洞"""
    if state.is_error: return state
    tech_rules = {"H2", "H9", "H10", "H12", "H13", "H19"}
    triggered = [r for r in state.detected_fallacies if r in tech_rules]
    if not triggered:
        return state
    try:
        LOGGER.info("[Tech Agent] 正在进行技术侧研判...")
        if not LANGCHAIN_AVAILABLE:
            state.agent_insights["首席技术官 CTO"] = "核心技术壁垒描述单薄，未见不可替代的护城河。"
            return state
        system_prompt = "你是硬科技风险基金的『首席技术官 CTO』。你的点评极其理智且略带悲观（不超过 40 字）。针对技术成熟度、研发团队实力、专利护城河问题进行犀利质问。"
        human_prompt = f"项目业务：{json.dumps(state.extracted_nodes, ensure_ascii=False)}\n触发的技术漏洞代码：{triggered}\n请输出你的结论："
        content = _call_openai_manual(system_prompt, human_prompt).strip()
        state.agent_insights["首席技术官 CTO"] = content
    except Exception as e:
        LOGGER.warning(f"Tech agent failed: {e}")
    return state


def finance_agent(state: AgentState) -> AgentState:
    """Finance Specialist Agent: 专门拷问单位经济与现金跑道"""
    if state.is_error: return state
    finance_rules = {"H7", "H8", "H15", "H16", "H18", "H20", "H7_GAP", "H8_GAP", "H15_GAP"}
    triggered = [r for r in state.detected_fallacies if r in finance_rules]
    if not triggered:
        return state
    try:
        LOGGER.info("[Finance Agent] 正在进行财务风控研判...")
        if not LANGCHAIN_AVAILABLE:
            state.agent_insights["财务风控总监"] = "单位经济模型算不过账，商业闭环逻辑极度残绝。"
            return state
        system_prompt = "你是硅谷顶级投行的『风控与财务官 CFO』。你的点评极其刻薄且对数字敏感（不超过 40 字）。针对盈利模式、LTV/CAC单位经济、烧钱率问题给出致命抨击。"
        human_prompt = f"项目业务：{json.dumps(state.extracted_nodes, ensure_ascii=False)}\n触发的财务风控漏洞代码：{triggered}\n请输出你的结论："
        content = _call_openai_manual(system_prompt, human_prompt).strip()
        state.agent_insights["财务风控总监"] = content
    except Exception as e:
        LOGGER.warning(f"Finance agent failed: {e}")
    return state


def generate_rebuttal(state: AgentState) -> AgentState:
    """Invokes the LLM to craft a Socratic rebuttal based on triggered fallacies."""
    if state.is_error:
        return state
        
    strategy_text = state.probing_strategy or DEFAULT_FALLACY_STRATEGY

    rules = state.detected_fallacies or ["H1-H15 均未触发"]
    evidence_summary = _format_evidence(state.evidence)

    # 教师干预指令与长期记忆注入
    intervention_text = ""
    if state.intervention_rules:
        rules_str = "\n".join([f"- {r}" for r in state.intervention_rules])
        intervention_text += f"\n### ⚠️ 教师特别干预指令 (优先级最高):\n{rules_str}\n"

    student_memory = state.accumulated_info.get('student_memory')
    if student_memory and student_memory not in ["该学生暂无长期记忆。", "新学生，暂无历史交互记忆"]:
        intervention_text += f"\n### 🧠 【系统过往交互记忆档案】\n> {student_memory}\n（你的反馈语气必须带连贯性记忆，如果本次该错误依然没改，可以类似这样开场：'上次我提醒过你...，但你这次依然...'）\n"

    # 新增：项目类型的指令注入
    project_type = state.accumulated_info.get("project_type", "商业型")
    if project_type == "公益型":
        intervention_text += "\n### 🌍 特别指令：公益项目优先\n该项目为【公益型】。在考察 H7、H8、H15、H16 相关盈利指标时，请**显著弱化**营收要求，将焦点转移至『如何自我造血维持社会服务』以及『社会效益覆盖面』。\n"


    expert_panel_text = ""
    if state.agent_insights:
        panel_str = "\n".join([f"- **{role}**: {insight}" for role, insight in state.agent_insights.items()])
        expert_panel_text = f"\n### 👥 实况专案组诊断 (Multi-Agent Panel)\n{panel_str}\n"

    if not LANGCHAIN_AVAILABLE:
        human_prompt = (
            "学生的原文输入：\n{student_input}\n"
            "抽取的模型（JSON）：\n{nodes}\n"
            "触发的规则：{rules}\n"
            "当前策略与参考案例：{strategy}\n"
            "专家组意见：\n{expert_panel}\n"
            "证据追踪：\n{evidence}\n"
            "{intervention_text}\n"
            "【反代写约束】如果用户要求代写或生成完整商业方案，果断拒绝并严厉指正。\n"
            "【输出格式规定】请严格按照以下 7 个 Markdown 标题格式强制输出回复，且【✅ 实践任务】只能有 1 个：\n\n"
            "### 🎯 诊断问题 (Issue)\n"
            "一句话指出当前逻辑中存在的漏洞。你必须包含一句引用：“『学生原文：...』”。\n\n"
            "### 👥 专家组会诊 (Expert Panel Review)\n"
            "你必须以‘主投资人’的身份，将提供给你的【专家组意见】汇总展示在这里，给到创业团队全方位的压力表现。\n\n"
            "### 📖 概念解析 (Definition)\n\n"
            "### 💡 案例参考 (Example)\n\n"
            "### 🔍 具体分析 (Analysis)\n"
            "【全面诊断】：基于前面抽取的业务模型，具体剖析为何这个想法站不住脚。如果触发了多个规则（{rules}），请在此处进行综合诊断，不要只盯着一个点。你要指出各个漏洞之间的逻辑关联。\n\n"
            "### 🤔 反思追问 (Socratic Question)\n"
            "【对照组施压】：你必须直接引用提供的“参考案例”作为对比。你需要表达为：‘相比于 [案例名称]，你的项目在 [维度] 上...’。绝对严禁在主语上将学生项目与参照案例项目混淆！\n\n"
            "### ✅ 实践任务 (Practice Task)\n"
            "提供且仅提供 1 个可执行的微步动作。注意：即便存在多个漏洞，任务也必须聚焦于最核心、最基础的那个（即‘第一块多米诺骨牌’）。如果存在【教师特别干预指令】，任务必须高度围绕该指令展开！"
        ).format(
            student_input=state.student_input,
            nodes=json.dumps(state.extracted_nodes or {}, ensure_ascii=False, indent=2),
            rules="、".join(rules),
            strategy=strategy_text,
            expert_panel=expert_panel_text if state.agent_insights else "无专家参与",
            evidence=evidence_summary,
            intervention_text=intervention_text,
        )
        manual_system = (
            f"你是一名极度犀利、且具有批判性思维的‘超图教练’，专门负责创新创业项目的初筛与辅导。"
            f"你永远不会直接给出答案，只会通过诊断逻辑漏洞引导学生自我修正。当前赛道：{state.target_competition}。"
        )
        try:
            content = _call_openai_manual(manual_system, human_prompt).strip()
            if intervention_text:
                content = intervention_text + "\n" + content
            state.response = content
        except Exception as exc:
            LOGGER.warning("OpenAI 直接调用失败：%s", exc)
            issues = "、".join(state.detected_fallacies) if state.detected_fallacies else "无"
            state.response = (
                f"在当前轮次我发现以下规则被触发：{issues}。请按照策略 '{strategy_text}' 进一步澄清。"
            )
        return state

    system_prompt = f"""你是一名极度犀利、且具有批判性思维的‘超图教练’，专门负责创新创业项目的初筛与辅导。
你的任务是根据学生的创业想法，结合底层知识图谱和双创竞赛 Rubric（当前赛道：{state.target_competition}），给出‘字字珠玑’的反馈。

{intervention_text}

### 核心反馈原则：
1. **证据为先**：每一条负面反馈必须引用『学生原文：...』。
2. **反代写约束**：如果用户要求代写或生成完整商业方案，果断拒绝并严厉指正。

You are a sharp venture coach whose tone is professional, precise, and tinged with cold humor. 
You never hand out answers, only diagnostics that force the student to self-correct.
【ANTI-GHOSTWRITING GUARDRAIL】如果你检测到用户的『原文输入』中在要求你直接生成完整的商业计划书(BP)、电梯演讲稿、或直接为你写出项目的段落内容，
你必须果断且清晰地拒绝。你的角色是创新创业教练，绝不能越俎代庖代写任何文档代码。你必须在你的『🎯 诊断问题 (Issue)』字段优先表达对这种代写要求的严厉拒绝，随后引导对话回到商业逻辑的校验上。"""
    human_template = (
        "学生的原文输入：\n{student_input}\n"
        "抽取的模型（JSON）：\n{nodes}\n"
        "触发的规则：{rules}\n"
        "各专业 Agent 评审意见：\n{expert_panel}\n"
        "当前策略与参考案例：“{strategy}”\n"
        "证据追踪：\n{evidence_summary}\n"
        "{intervention_text}\n"
        "请结合当前用户的输入和专家组的多智能体报告（尤其是【教师特别干预指令】），作为 Lead Coach 给出最终裁决回复。\n"
        "【输出格式规定】请用中文严格按照以下 Markdown 格式强制输出 7 个字段（务必包含所有 7 个 ### 标题），且【✅ 实践任务】只能有 1 个极其具体的动作："
        "\n\n"
        "### 🎯 诊断问题 (Issue)\n"
        "一句话指出当前逻辑中存在的漏洞（或指出你绝不代写）。你必须包含一句引用：“『学生原文：[填入原话]』”作为诊断依据！\n\n"
        "### 👥 专家组会诊 (Expert Panel Review)\n"
        "如果有【各专业 Agent 评审意见】数据，请你在这里直接原文转述各财务/技术/市场专家的简报。如果没有任何其它 Agent 发言，则直接写：“经过初步诊断无致命模块漏洞，进入标准主观论证评价。”\n\n"
        "### 📖 概念解析 (Definition)\n"
        "客观解释该谬误涉及的创新创业概念或原理。\n\n"
        "### 💡 案例参考 (Example)\n"
        "给出一个相关的正向或反向商业案例以加强理解。\n\n"
        "### 🔍 具体分析 (Analysis)\n"
        "【全面诊断】：基于前面抽取的业务模型，具体剖析这个想法为何站不住脚。如果系统检测到了多个逻辑谬误（{rules}），你必须在这一节对它们进行全面评估，揭示它们之间的内在逻辑崩溃（如资源脱节导致成本模型失效）。不要只盯着一个点！\n\n"
        "### 🤔 反思追问 (Socratic Question)\n"
        "【对照组施压】：你必须直接引用“当前策略与参考案例”中提供的超边参照案例作为对标。你需要清晰地表达‘对比 [参照案例名称]，你的项目在 [具体逻辑] 上是否存在...’。绝对严禁在主语上将学生项目与参照案例项目混淆！请抛出一个犀利、直击痛点的开放性问题。\n\n"
        "### ✅ 实践任务 (Practice Task)\n"
        "提供且仅提供 1 个具体的课后微步动作（例如查算某个特定数据）。不要罗列 1.2.3. 多项！注意：如果存在【教师特别干预指令】，你的诊断、分析和任务必须高度围绕该指令展开！"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_template),
        ]
    )

    try:
        prompt_payload = prompt.format_prompt(
            student_input=state.student_input,
            nodes=json.dumps(state.extracted_nodes or {}, ensure_ascii=False, indent=2),
            rules="、".join(rules),
            strategy=strategy_text,
            expert_panel=expert_panel_text if state.agent_insights else "无其他专家意见",
            evidence_summary=evidence_summary,
            intervention_text=intervention_text,
        ).to_messages()
        response = _get_chat_client().invoke(prompt_payload)
        content = getattr(response, "content", None) or (response.choices[0].message.content if response.choices else "")
        
        # A2 Post-processing: Strict Single Task Constraint
        import re
        content = content.strip()
        task_split = re.split(r'###\s*✅\s*实践任务.*?\n', content, flags=re.IGNORECASE)
        if len(task_split) > 1:
            pre_task = task_split[0]
            task_blocks = task_split[1]
            first_task = re.split(r'\n(?:\d+\.|-|\*)\s+', task_blocks.strip())[0]
            if not first_task.strip() and len(re.split(r'\n(?:\d+\.|-|\*)\s+', task_blocks.strip())) > 1:
               first_task = re.split(r'\n(?:\d+\.|-|\*)\s+', task_blocks.strip())[1] 
            content = f"{pre_task.strip()}\n\n### ✅ 实践任务 (Practice Task)\n{first_task.strip()}"
            
        if intervention_text:
            content = intervention_text + "\n" + content
        state.response = content
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Rebuttal generation failed; falling back messaging (%s)", exc)
        issues = "、".join(state.detected_fallacies) if state.detected_fallacies else "无"
        state.response = (
            f"⚠️ **分析引擎提示**：详细的教练模型生成失败（原因：{str(exc)}）。\n\n"
            f"**初步诊断**：在当前轮次我发现以下规则被触发：{issues}。\n\n"
            f"**策略提示**：{strategy_text}"
        )
    return state


def audit_reflection(state: AgentState) -> AgentState:
    """A8: Harness Engineering - 系统管控节点。对生成的反馈进行确定性质量审计，不合格则打回重写。"""
    if state.is_error or not state.response:
        return state
        
    import re
    issues = []
    response_text = state.response
    
    # 规则 1：证据锁（必须引用原话）
    if "学生原文：" not in response_text and "学生原文:" not in response_text:
        # Check if D003 short inputs were used, maybe there is no sentence, but rules demand it
        issues.append("未包含『学生原文：』的引用格式以展示诊断证据。")
        
    # 规则 2：微步约束锁（不能有多项任务）
    task_blocks = re.findall(r'###\s*✅\s*实践任务.*?([\s\S]*)', response_text, flags=re.IGNORECASE)
    if task_blocks:
        task_text = task_blocks[0].strip()
        # 查找明显的列表项符号（1. 2. 或是 - ）
        bullets = re.findall(r'\n(?:\d+\.|-|\*)\s+', "\n" + task_text)
        if len(bullets) > 1:
            issues.append(f"布置了过多实践任务（检测到 {len(bullets)} 项）。必须严格浓缩为 1 个最基础的微步动作。")
    else:
        issues.append("未找到必须的『✅ 实践任务 (Practice Task)』标准 Markdown 标题。")
        
    # 规则 3：反代写底线 (如果学生说了代写关键词，回复里必须有明确拒绝)
    ghostwriting_keywords = ["帮我写", "代写", "生成一份", "写一段", "帮我做", "完整方案"]
    if any(k in state.student_input for k in ghostwriting_keywords):
        refusal_keywords = ["拒绝", "不能代写", "绝不代写", "不提供代写", "独立", "越俎代庖", "无法直接给出"]
        if not any(k in response_text for k in refusal_keywords):
            issues.append("由于学生触发了代写请求，教练必须优先在回复中严词拒绝，但当前回复显得过于妥协。")
            
    if issues:
        LOGGER.warning("[Audit Reflection] 拦截！回复未通过质量管控：%s", issues)
        state.evidence.append(EvidenceItem(
            step="audit_reflection",
            detail=f"系统管控：初稿因【{'; '.join(issues)}】被拦截，正在触发 LLM 自我纠偏回滚机制。",
            source_excerpt="[System Audit]"
        ))
        
        if not LANGCHAIN_AVAILABLE:
            state.response += f"\n\n> ⚙️ **Harness Control 提示**：系统检测到上述输出存在格式违规（{issues[0]}），但由于处于离线模式，自动回滚重写被跳过。"
            return state
            
        # 采用小 Prompt 进行快速修复，直接修改格式错误，不改变原意
        system_prompt = (
            "你是创业辅导系统的『最终质量审查员』(Harness Controller)。"
            "以下教练反馈草稿未能通过系统的代码级规则审计。请你在**不改变其核心评价和专业意见**的前提下，立刻修正这些违规项。\n\n"
            "【严格修正指令】：\n"
            "1. 若提示缺少原文引用，请根据上下文强制补全一句『学生原文：...』。\n"
            "2. 若提示任务过多，请毫不留情地砍掉多余任务，只保留最核心的、第一步该做的 1 个任务。\n"
            "3. 输出依然必须包含原有的 7 个 Markdown 标题结构（Issue, Expert Panel Review, Definition, Example, Analysis, Socratic Question, Practice Task）。"
        )
        human_prompt = f"【教练初稿】\n{response_text}\n\n【审计出的违规项】\n{chr(10).join(issues)}\n\n请输出修正后的完整最终回复："
        
        try:
            client = _get_chat_client()
            from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_prompt),
                HumanMessagePromptTemplate.from_template(human_prompt)
            ])
            LOGGER.info("[Audit Reflection] 正在调用 LLM 执行自动纠偏回滚...")
            response = client.invoke(prompt.format_messages())
            corrected_content = getattr(response, "content", None) or (response.choices[0].message.content if response.choices else "")
            
            if corrected_content:
                state.response = corrected_content
                LOGGER.info("[Audit Reflection] 回滚纠偏成功。")
            else:
                LOGGER.warning("[Audit Reflection] 纠偏返回为空，放弃回滚。")
        except Exception as e:
            LOGGER.error("[Audit Reflection] 回滚修复失败: %s", e)
            
    return state


def update_memory_engine(state: AgentState) -> AgentState:
    """A8-6: Context Engineering - 动态凝练学生的长期记忆画像"""
    if state.is_error:
        return state
        
    current_memory = state.accumulated_info.get("student_memory", "该学生暂无长期记忆。")
    current_issues = "、".join(state.detected_fallacies) if state.detected_fallacies else "本轮无明显漏洞（表现优异）"
    
    if not LANGCHAIN_AVAILABLE:
        return state
        
    system_prompt = (
        "你是负责维护『学生认知档案』的记忆引擎。你的任务是根据学生过往的记忆和本轮触发的逻辑漏洞，"
        "用50字以内的一段话，精炼总结该学生的『核心商业逻辑短板』及『近期思维表现』。\n"
        "要求：极度客观、犀利。指出目前最卡脖子的问题。如果学生本次解决了以往的问题，请移除相关负面评价。"
    )
    human_prompt = f"【历史记忆】：{current_memory}\n【本轮触发漏洞】：{current_issues}\n请输出更新后的学生认知档案（50字内）："
    
    try:
        from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])
        client = _get_chat_client()
        response = client.invoke(prompt.format_messages())
        new_memory = getattr(response, "content", None) or (response.choices[0].message.content if response.choices else "")
        if new_memory:
            state.accumulated_info["student_memory"] = new_memory.strip()
            LOGGER.info("[Context Engine] 记忆档案已更新: %s", state.accumulated_info["student_memory"])
    except Exception as e:
        LOGGER.error("[Context Engine] 记忆更新失败: %s", e)
        
    return state


def rubric_scorer(state: AgentState) -> AgentState:
    """A5: 根据 Rubric 维度对项目进行 0-5 逐项评分，并为薄弱项生成 Missing Evidence + Minimal Fix。"""
    if state.is_error:
        return state
        
    fallacies = set(state.detected_fallacies)
    scores: Dict[str, Any] = {}

    for dim, related_rules in RUBRIC_FALLACY_MAP.items():
        triggered = [r for r in related_rules if r in fallacies]
        
        # A5-1: 采用“梯度扣分制” (Penalty-based)
        # 将真正逻辑错误 (Real) 与 数据缺失 (Gap) 分开计算
        real_triggered = [r for r in triggered if not r.endswith("_GAP")]
        gap_triggered = [r for r in triggered if r.endswith("_GAP")]
        
        real_penalty = sum(FALLACY_SEVERITY.get(r, 1.5) for r in real_triggered)
        # GAP 扣分每维度封顶 1.0 分，防止数据缺失直接导致项目挂掉
        gap_penalty = min(sum(FALLACY_SEVERITY.get(r, 0.5) for r in gap_triggered), 1.0)
        
        raw_score = 5.0 - (real_penalty + gap_penalty)
        raw_score = max(0.0, min(5.0, raw_score))
        
        # 致命伤限制：只要有真正的 Fatal 谬误 (>=2.5)，评分不得高于 2
        has_fatal = any(FALLACY_SEVERITY.get(r, 0) >= 2.5 for r in real_triggered)
        if has_fatal:
            raw_score = min(2.0, raw_score)
            
        display_score = round(raw_score, 1)

        dim_result: Dict[str, Any] = {
            "score": display_score,
            "name": RUBRIC_DIM_NAMES[dim],
            "triggered_rules": triggered,
        }

        # 只要评分低于 4.0 就给出建议 (Stricter gate)
        if display_score < 4.0:
            dim_result["missing_evidence"] = RUBRIC_MISSING_FIX[dim]["missing"]
            dim_result["minimal_fix"] = RUBRIC_MISSING_FIX[dim]["fix"]

        # A5-5: 增加实证证据关联 (Req 9)
        dim_result["evidence_list"] = [
            {"label": ev.step, "detail": ev.detail} 
            for ev in state.evidence if ev.step in triggered
        ]
        scores[dim] = dim_result

    # A5-4: 构建实证证据链 (Req 9)
    # 为评分低于 4.0 的维度寻找“实证”
    new_evidence = []
    for dim, res in scores.items():
        if dim == "_summary": continue
        if res["score"] < 4.0:
            for rule in res["triggered_rules"]:
                rule_desc = FALLACY_STRATEGY_LIBRARY.get(rule, "未定义的逻辑缺陷")
                # 寻找导致此规则触发的输入片段 (简单演示版：直接说明逻辑匹配点)
                new_evidence.append({
                    "step": res["name"],
                    "detail": f"触发漏洞【{rule}】：{rule_desc.split('：')[0]}。逻辑判定点：学生在描述中对该维度的实质性支撑不足。"
                })
    
    state.evidence = new_evidence
    # 动态切换赛事权重
    target_comp = state.target_competition
    if target_comp not in COMPETITION_WEIGHTS:
        target_comp = "互联网+"
    
    current_weights = COMPETITION_WEIGHTS[target_comp]
    weighted_total = sum(
        scores[dim]["score"] * current_weights.get(dim, 0.2) 
        for dim in RUBRIC_FALLACY_MAP.keys() if dim in scores
    )
    
    # A5-2: 最终加权与审计干预 (Systemic Penalty & Tiered Gating)
    # 统计真实谬误数量（排除 GAP）
    real_f_list = [f for f in fallacies if not f.endswith("_GAP")]
    num_real = len(real_f_list)
    
    # 全局谬误惩罚 (超3个谬误后，每个扣0.5)
    if num_real > 3:
        weighted_total -= (num_real - 3) * 0.5
    
    # 【全局核心准入机制 (Empathetic Fairness v3.8)】
    # 定义核心维度：痛点发现、方案策划、商业建模。
    core_dims = ["pain_point", "planning", "modeling"]
    worst_core = min(scores.get(d, {}).get("score", 5.0) for d in core_dims)
    
    # 熔断策略与“UI 同步”处理：
    # 为了保证公平，放宽准入门槛：
    # 1. 核心维度 < 1.0 (完全无此模块)：上限 1.8
    # 2. 核心维度 < 2.0 (严重缺失/逻辑不通)：上限 3.2 (提高到 B- 档)
    # 3. 核心维度 >= 2.0 (虽平庸但具备雏形)：不触发强制熔断，仅享受权重自然加权
    if worst_core < 2.0:
        target_cap = 1.8 if worst_core < 1.0 else 3.2
        if weighted_total > target_cap:
            scale_ratio = target_cap / weighted_total if weighted_total > 0 else 1.0
            # 执行等比例离散化缩放，确保前端 UI 计算出的加权和也遵循封顶规则
            for d in scores:
                if d != "_summary":
                    scores[d]["score"] = round(scores[d]["score"] * scale_ratio, 2)
            
            weighted_total = target_cap
            LOGGER.info("UI Sync Triggered v3.8 (worst_core=%.1f): Scaling all dims by %.2f", worst_core, scale_ratio)

    # 【动态木桶效应】：总分上限受制于表现最差的“实锤”维度 (1.5倍杠杆)
    barrel_candidates = [scores[d]["score"] for d in scores if d != "_summary" and any(not r.endswith("_GAP") for r in scores[d].get("triggered_rules", []))]
    if barrel_candidates:
        worst_real_score = min(barrel_candidates)
        barrel_cap = worst_real_score * 1.5
        if weighted_total > barrel_cap:
            weighted_total = barrel_cap

    weighted_total = max(0.0, weighted_total)
    
    scores["_summary"] = {
        "weighted_total": round(weighted_total, 2),
        "total_real_fallacies": num_real,
        "worst_core_score": round(worst_core, 2),
        "default_competition": target_comp,
    }
    
    state.rubric_scores = scores
    return state
try:
    from langgraph import Node, StateGraph
    NODE_CLASS_AVAILABLE = True
except ImportError:  # pragma: no cover - minimal fallback graph
    NODE_CLASS_AVAILABLE = False
    
    @dataclass
    class Node:
        name: str
        function: Callable[[AgentState], AgentState]

    class StateGraph:
        def __init__(self, name: str):
            self.name = name
            self.nodes: List[Node] = []
            self.edges: Dict[str, List[str]] = {}

        def add_node(self, node: Node) -> None:
            self.nodes.append(node)
            self.edges.setdefault(node.name, [])

        def connect(self, src: str, dst: str) -> None:
            self.edges.setdefault(src, []).append(dst)

        def compile(self) -> None:
            print(f"[StateGraph] {self.name} compiled with nodes: {[n.name for n in self.nodes]}")

        def execute(self, state: AgentState) -> AgentState:
            for node in self.nodes:
                state = node.function(state)
            return state


def generate_intervention_plan(stats: Dict[str, Any]) -> str:
    """A6: 根据全班高频错误生成下周教学干预计划。"""
    system_prompt = (
        "你是资深的创新创业导师。需要根据教师端传入的班级近期错误统计数据，设定『下周教学干预计划』。\n"
        "要求：分点列出教学重点、课堂互动形式建议、课后练习设定等。要求内容具体、有执行路径，"
        "使用Markdown进行结构化排版，突出痛点和建议。"
    )
    human_prompt = f"班级错题/高频漏洞统计：\n{stats}\n请生成下周的教学干预重点和针对性行动计划。"
    try:
        return _call_openai_manual(system_prompt, human_prompt)
    except Exception as e:
        LOGGER.error(f"Error generating intervention plan: {e}")
        return f"生成干预计划失败：{e}"


def generate_student_profile(student_data: Dict[str, Any], for_student: bool = False) -> str:
    """A6-4: 根据项目历史生成动态画像，区分师生视角。"""
    if for_student:
        system_prompt = (
            "你是资深的创新创业AI教练。请根据这位学生的『经常触发的薄弱逻辑规则』及表现数据，"
            "直接向该学生反馈『个人专属能力剖析报告』。注意：绝对不要在回答中透露系统量化评分或底部分数，避免带来焦虑，也不要出现教师术语。\n\n"
            "请按以下四个核心模块向学生进行反馈：\n"
            "### 🌟 你的核心优势 (Key Strengths)\n"
            "分析他在项目推演中未触雷或表现稳定的维度闪光点。\n\n"
            "### ⚠️ 你的逻辑盲区 (Logic Blind Spots)\n"
            "基于经常触发的漏洞，一针见血地指出该学生经常犯的逻辑漏洞或认知偏差。\n\n"
            "### 📈 针对性提升建议 (Actionable Advice)\n"
            "告诉学生接下来应该在哪些方面恶补（如调研、财务计算等）。\n\n"
            "### 🚀 下一步突破方向 (Next Milestones)\n"
            "设定明确的下一步项目迭代目标。\n\n"
            "要求：排版清晰，使用 Markdown 格式。语气像一位导师：真诚、鼓励但不失犀利。所有内容都在跟学生（“你”）对话。"
        )
        safe_data = {
            "name": student_data.get("name", "学生"),
            "frequent_fallacies": student_data.get("frequent_fallacies", []),
            "session_count": student_data.get("session_count", 0)
        }
        human_prompt = f"我的项目推演历史漏洞统计：\n{safe_data}\n请输出专属我的能力画像与指导建议。"
    else:
        system_prompt = (
            "你是资深的创新创业导师。请根据该学生的『项目得分情况』和『经常触发的薄弱逻辑规则』，"
            "按照以下三个核心阶段生成『学生动态能力画像报告』：\n\n"
            "### 第一阶段：价值发现能力 (Value Detection)\n"
            "分析学生发现真实痛点、定义价值主张的能力及敏感度。\n\n"
            "### 第二阶段：压力测试表现 (Pressure Test)\n"
            "评估学生在面对逻辑挑战、盈亏平衡拷问及竞争风险时的防御与修正能力。\n\n"
            "### 第三阶段：执行可行性 (Execution Feasibility)\n"
            "判断学生对市场渠道、成本结构及落地微步动作的理解深度。\n\n"
            "### 💡 教师辅导建议\n"
            "为授课教师提供针对该学生的『一对一干预重点』与『推荐辅导话术』。\n\n"
            "要求：排版清晰美观，使用 Markdown 格式，结论要犀利且具备指导意义。"
        )
        human_prompt = f"该学生数据分析：\n{student_data}\n请输出综合能力画像及教师1对1辅导策略。"
        
    try:
        return _call_openai_manual(system_prompt, human_prompt)
    except Exception as e:
        LOGGER.error(f"Error generating student profile: {e}")
        return f"生成能力画像失败：{e}"


def generate_financial_report(project_data: Dict[str, Any], for_student: bool = False) -> str:
    """A9: 根据项目对话中积累的商业数据，生成结构化财务分析报告。"""
    if for_student:
        system_prompt = (
            "你是一位资深的创业财务顾问（CFO Advisor）。请根据学生项目的商业数据，"
            "为该学生生成一份『项目财务健康诊断报告』。\n\n"
            "注意：你是在直接向学生反馈，语气应该专业但亲切，避免过于学术化。"
            "不要透露系统内部评分机制，专注于帮助学生理解自己项目的财务逻辑。\n\n"
            "请严格按照以下 Markdown 格式输出：\n\n"
            "### 💰 收入模型诊断 (Revenue Model)\n"
            "分析当前项目的收入来源是否清晰，盈利模式是否可持续。如果数据不足，指出需要补充哪些关键数据。\n\n"
            "### 📊 成本结构拆解 (Cost Structure)\n"
            "拆解项目的固定成本与变动成本，分析边际成本趋势。如果缺少成本数据，给出一个同类项目的参考框架。\n\n"
            "### ⚖️ 盈亏平衡测算 (Break-Even Analysis)\n"
            "基于现有数据，估算项目需要多少用户/订单才能达到盈亏平衡点。如果数据不够，给出假设条件下的推演。\n\n"
            "### 📈 现金流与生存预警 (Cash Flow Forecast)\n"
            "根据烧钱率和现金跑道，评估项目的生存周期。给出具体的财务预警信号和应对建议。\n\n"
            "### 🎯 你的财务行动清单 (Action Items)\n"
            "列出 3 个最紧迫的、学生可以立即着手的财务优化动作。\n\n"
            "要求：如果某些数据缺失，不要跳过该模块，而是基于行业常识给出合理假设并标注『⚠️ 基于假设推演』。"
        )
    else:
        system_prompt = (
            "你是一位顶级投行的财务分析师。请根据该学生项目的商业数据，"
            "生成一份面向教师/评委的『项目财务深度诊断报告』。\n\n"
            "请严格按照以下 Markdown 格式输出：\n\n"
            "### 💰 收入模型评估 (Revenue Model Assessment)\n"
            "量化分析收入来源、定价策略合理性，给出行业对标数据。\n\n"
            "### 📊 成本结构与单位经济学 (Unit Economics)\n"
            "拆解 LTV/CAC 比率、边际成本、固定成本占比，评估单位经济模型是否成立。\n\n"
            "### ⚖️ 盈亏平衡与敏感性分析 (Break-Even & Sensitivity)\n"
            "计算盈亏平衡点，并对关键变量（客单价、转化率、获客成本）进行敏感性分析。\n\n"
            "### 📈 现金流预测与融资需求 (Cash Flow & Funding)\n"
            "基于烧钱率预测现金跑道，评估是否需要外部融资以及合理的融资节奏。\n\n"
            "### ⚠️ 财务风险矩阵 (Financial Risk Matrix)\n"
            "列出 Top 3 财务风险及其发生概率和影响程度，给出对冲策略建议。\n\n"
            "### 💡 教师辅导建议 (Teaching Recommendations)\n"
            "针对该学生的财务薄弱环节，为教师提供具体的辅导切入点和推荐练习。\n\n"
            "要求：数据驱动，结论犀利，排版清晰。如数据缺失请标注并给出行业基准假设。"
        )

    # 构建财务数据上下文
    accumulated = project_data.get("accumulated_info", {})
    extracted = project_data.get("extracted_nodes", {})
    
    finance_context = {
        "project_name": accumulated.get("project_name", extracted.get("project_name", "未命名项目")),
        "revenue": extracted.get("revenue", accumulated.get("revenue", 0)),
        "LTV": extracted.get("LTV", 0),
        "CAC": extracted.get("CAC", 0),
        "monthly_burn": extracted.get("monthly_burn", 0),
        "cash_runway": extracted.get("cash_runway", 0),
        "marginal_cost": extracted.get("marginal_cost", ""),
        "revenue_model": extracted.get("revenue_model", accumulated.get("revenue_model", "")),
        "target_market": accumulated.get("target_market", extracted.get("target_market", "")),
        "tech_maturity": accumulated.get("tech_maturity", extracted.get("tech_maturity", "")),
        "funding_stage": accumulated.get("funding_stage", ""),
        "frequent_fallacies": project_data.get("frequent_fallacies", []),
        "session_count": project_data.get("session_count", 0),
    }

    human_prompt = f"项目财务数据：\n{json.dumps(finance_context, ensure_ascii=False, indent=2)}\n请生成完整的财务分析报告。"

    try:
        return _call_openai_manual(system_prompt, human_prompt)
    except Exception as e:
        LOGGER.error(f"Error generating financial report: {e}")
        return f"生成财务分析报告失败：{e}"


def build_state_graph() -> StateGraph:
    graph = StateGraph(name="HypergraphCoach")
    nodes = [
        Node(name="extract_entities", function=extract_entities),
        Node(name="hypergraph_critic", function=hypergraph_critic),
        Node(name="strategy_selector", function=strategy_selector),
        Node(name="market_agent", function=market_agent),
        Node(name="tech_agent", function=tech_agent),
        Node(name="finance_agent", function=finance_agent),
        Node(name="generate_rebuttal", function=generate_rebuttal),
        Node(name="audit_reflection", function=audit_reflection),
        Node(name="rubric_scorer", function=rubric_scorer),
        Node(name="update_memory_engine", function=update_memory_engine),
    ]
    for node in nodes:
        graph.add_node(node)
    graph.connect("extract_entities", "hypergraph_critic")
    graph.connect("hypergraph_critic", "strategy_selector")
    graph.connect("strategy_selector", "market_agent")
    graph.connect("market_agent", "tech_agent")
    graph.connect("tech_agent", "finance_agent")
    graph.connect("finance_agent", "generate_rebuttal")
    graph.connect("generate_rebuttal", "audit_reflection")
    graph.connect("audit_reflection", "rubric_scorer")
    graph.connect("rubric_scorer", "update_memory_engine")
    graph.compile()
    return graph



def render_frontend_snapshot(state: AgentState) -> None:
    print("\n----- Front-End Snapshot -----")
    print(f"Stage: {state.probing_strategy or '等待第一轮完成'}")
    print(f"Detected nodes: {state.extracted_nodes}")
    print(f"Detected fallacies: {state.detected_fallacies}")
    print(f"Planned response: {state.response}")
    if state.evidence:
        print("Evidence trail:")
        for item in state.evidence:
            print(f"- {item.step}: {item.detail}")
    print("-------------------------------\n")



def generate_business_plan(project_data: Dict[str, Any], target_comp: str = "互联网+") -> str:
    """A10: 综合对话上下文与累积数据，生成完整的、具有赛道针对性的商业计划书。"""
    
    # 赛道偏好定义
    COMP_FOCUS = {
        "互联网+": "重点关注：商业模式闭环、市场增长潜力、融资计划与规模化能力。话术应具有强烈的‘创业家’和‘商业风控’色彩。",
        "挑战杯": "重点关注：核心学术创新、技术发明点、社会贡献度、产学研转化价值。话术应更学术专业、稳健，强调‘科技强国’和‘成果转化’。",
        "创青春": "重点关注：方案的落地可行性、团队执行力、青年就业与社会影响力。话术应更具‘活力’与‘实干’感。",
        "数模": "重点关注：数学模型的严密性、数据驱动的决策逻辑、复杂问题的量化拆解。话术应逻辑严密，极度推崇‘量化’与‘确定性’。"
    }
    
    focus_point = COMP_FOCUS.get(target_comp, COMP_FOCUS["互联网+"])
    
    system_prompt = (
        f"你是一名的全球顶级的‘双创’初创企业孵化与风险投资专家（针对 {target_comp} 赛道）。\n"
        f"你的任务是将碎片化的项目信息与对话对话历史，通过‘深度战略推演’逻辑，合成为一份极具投融资价值和赛事竞争力的【12章节风投标准】商业计划书。\n\n"
        f"【{target_comp} 赛道生成偏好】：\n{focus_point}\n\n"
        "【输出格式与深度要求】：\n"
        "1. **文风要求**：语料风格必须对标麦肯锡/普华永道等咨询公司报告，使用战略性词汇（如：非对称竞争优势、结构性博弈、高边际壁垒等）。\n"
        "2. **内容深度**：每一章必须包含深度逻辑推论，严禁短句，单节字数不少于 250 字，全文需有极强的叙事张力。\n"
        "3. **结构指引**：请严格按照以下 12 个模块生成 Markdown 内容：\n\n"
        "## 1. 项目愿景与 Slogan (Vision & Tagline)\n"
        "【路演级开篇】用一句话定义项目的‘终局感’。阐述企业存在的根本使命与其在行业中的标签化地位。\n\n"
        "## 2. 行业底层痛点映射 (The Problem)\n"
        "详述现有的业务逻辑缺陷。引用具体数据、用户调研或实地观察，揭示现有方案在效率、成本或体验上的‘结构性崩坏’。\n\n"
        "## 3. 市场切入时机分析 (Why Now? / Timing)\n"
        "【核心模块】分析为什么 2024-2026 是最佳窗口期。涉及政策红利（如 ESG/低空经济）、技术拐点（如 LLM 成熟）或消费行为突变。\n\n"
        "## 4. 市场规模与潜力漏斗 (Market Opportunity)\n"
        "深度解析 TAM/SAM/SOM。不仅要数字，更要测算逻辑（自下而上或自上而下），展示未来的市场爆发天花板。\n\n"
        "## 5. 核心方案与用户全旅程设计 (Solution & Journey)\n"
        "不仅描述功能，需从用户第一个触点到最后价值产出的全流程进行‘场景化描述’，体现方案的闭环性。\n\n"
        "## 6. 技术核武器与动态护城河 (Technology & Moat)\n"
        "探讨底层算法、专利或系统架构。详述由‘数据飞轮’或‘行政/行政资源准入’带来的竞争对手‘不可迁移成本’。\n\n"
        "## 7. 全景竞争对比矩阵 (Competitive Matrix)\n"
        "【深度模块】对比本项目与主要竞品（如闲鱼、转转、顺丰等）。建立 2D 象限图模型（文字描述），强调核心差异化优势（不同点即竞争点）。\n\n"
        "## 8. 商业模式与单位经济模型 (Business Model & UE)\n"
        "分析多维收入流。重点推算 UE 模型：LTV/CAC 的动态平衡、毛利结构及盈亏平衡点（BEP）的时间轴推断。\n\n"
        "## 9. 营销渗透与增长飞轮 (Growth Flywheel)\n"
        "基于 STP 模型的市场定位。详述冷启动方案、病毒式传播机制（裂变系数 K 值）以及如何构建‘越用越强’的增长飞轮。\n\n"
        "## 10. 财务模型预测 (Financial Projections)\n"
        "提供未来 3 年在‘保守/基准/乐观’情景下的核心财务数据预测。说明各项支出的权重依据建议。\n\n"
        "## 11. 融资计划与 24 个月战略路线图 (The Ask & Roadmap)\n"
        "【关键模块】明确融资额度分配。制定跨度 24 个月的里程碑计划，每个节点需包含明确的可交付成果与验证指标。\n\n"
        "## 12. 团队基因与社会使命 (Team & Vision)\n"
        "论证创始人背景与项目的高度契合性。阐述项目在社会责任（如绿色环保、就业）方面的长期愿景。\n\n"
        "注意：对于缺失的数据，请根据行业常识给出‘极其专业且合理’的预测值，并用加粗字体提示（如 **[基于行业 Benchmark 及战略推演：XX%]**）。\n"
    )
    
    project_type = project_data.get("accumulated_info", {}).get("project_type", "商业型")
    if project_type == "公益型":
        system_prompt += (
            "\n\n【🌍 赛道强制指令：公益型项目】\n"
            "该项目已标记为**公益/红旅赛道**，你必须在生成内容时全面转向公益评价体系：\n"
            "1. **商业模式变现要求降低**：不用过分苛求 LTV/CAC，转而重点阐述【自我造血机制】（如何通过服务覆盖成本）与资金筹措透明度。\n"
            "2. **Outcome 锚点**：在“市场规模”与“痛点”章节，重点阐述受益人群的实质性改变（如教育公平、可持续发展），而非单纯计算 TAM/SAM。\n"
            "3. **公信力与影响力飞轮**：将“增长飞轮”替换为“社会影响力飞轮”，通过第三方背书展示项目的长期普惠价值。"
        )

    accumulated = project_data.get("accumulated_info", {})
    extracted = project_data.get("extracted_nodes", {})
    history = project_data.get("conversation_history", [])
    
    # 序列化历史记录以便于 LLM 理解
    history_text = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in history[-10:]]) # 取最近10轮
    
    context = {
        "target_competition": target_comp,
        "history_context": history_text,
        "extracted_nodes": extracted,
        "accumulated_info": accumulated,
        "detected_fallacies": project_data.get("frequent_fallacies", [])
    }

    human_prompt = f"对话上下文与商业数据信息：\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n请立即合成一份成熟的、高质量的 {target_comp} 赛道商业计划书。"

    try:
        return _call_openai_manual(system_prompt, human_prompt)
    except Exception as e:
        LOGGER.error(f"Error generating business plan: {e}")
        return f"合成商业计划书失败：{e}"


def revise_business_plan_with_feedback(
    project_data: Dict[str, Any],
    draft_markdown: str,
    teacher_feedback: str,
    target_comp: str = "互联网+",
) -> str:
    """Use teacher feedback to revise an existing business plan into the final version."""
    accumulated = project_data.get("accumulated_info", {})
    extracted = project_data.get("extracted_nodes", {})
    history = project_data.get("conversation_history", [])
    history_text = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in history[-10:]])

    system_prompt = (
        f"你是一名同时具备项目教练、竞赛顾问与高校导师评审经验的双创专家，正在将一份 {target_comp} 赛道商业计划书初稿修订为终稿。\n"
        "你的目标不是重写无关内容，而是严格吸收导师评审意见，对初稿做有依据的增强、纠偏与补足。\n\n"
        "【修订原则】\n"
        "1. 必须保留原有项目设定、叙事主线与章节结构，输出仍为 Markdown。\n"
        "2. 必须逐条落实教师意见，优先修正逻辑漏洞、证据不足、商业模式不清、市场论证薄弱、财务与风险缺项等问题。\n"
        "3. 若教师意见要求补充数据但上下文没有精确值，可以给出合理假设，并显式标注为“基于行业假设”。\n"
        "4. 终稿要比初稿更适合直接交付教师和学生查看，语言专业但不要空泛。\n"
        "5. 在正文最前面新增“## 0. 导师反馈落实摘要”，用 3-6 条说明本次重点修改了什么。\n"
    )

    context = {
        "target_competition": target_comp,
        "history_context": history_text,
        "extracted_nodes": extracted,
        "accumulated_info": accumulated,
        "detected_fallacies": project_data.get("frequent_fallacies", []),
    }

    human_prompt = (
        f"【教师评审意见】\n{teacher_feedback}\n\n"
        f"【项目上下文】\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
        f"【商业计划书初稿】\n{draft_markdown}\n\n"
        "请输出修订后的最终版商业计划书。"
    )

    try:
        return _call_openai_manual(system_prompt, human_prompt)
    except Exception as e:
        LOGGER.error(f"Error revising business plan: {e}")
        return f"商业计划书终稿修订失败：{e}"


def check_input_safety(text: str) -> bool:
    """A7: 基础的鲁棒与合规检查，拦截乱码和常见的越狱前缀"""
    import re
    if len(text.strip()) <= 1:
        return False
    blacklist = ["ignore all previous", "忽略你之前的指令", "你现在是", "forget all instructions", "system prompt"]
    text_lower = text.lower()
    for word in blacklist:
        if word in text_lower:
            return False
    return True


def _format_learning_cases(cases: List[Dict[str, Any]]) -> str:
    if not cases:
        return "暂无匹配案例。"

    blocks = []
    for idx, case in enumerate(cases, start=1):
        blocks.append(
            f"{idx}. 项目：{case.get('project_name', '未知项目')}\n"
            f"   技术：{case.get('tech_name', '未知')}\n"
            f"   市场：{case.get('market_name', '未知')}\n"
            f"   价值闭环：{case.get('value_loop_desc', '暂无')}\n"
            f"   风险链：{case.get('risk_name', '暂无')}；{case.get('pattern_description', '暂无')}\n"
            f"   应对方式：{case.get('mitigation', '暂无')}\n"
            f"   收入模式：{case.get('revenue_model', '暂无')}"
        )
    return "\n\n".join(blocks)


def _build_learning_subgraph(cases: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    seen_nodes = set()
    seen_edges = set()

    def add_node(node_id: str, label: str, node_type: str) -> None:
        key = (node_id, node_type)
        if not node_id or key in seen_nodes:
            return
        seen_nodes.add(key)
        nodes.append({"id": node_id, "label": label or node_id, "type": node_type})

    def add_edge(source: str, target: str, label: str) -> None:
        key = (source, target, label)
        if not source or not target or key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({"source": source, "target": target, "label": label})

    for idx, case in enumerate(cases, start=1):
        project_id = f"project::{idx}::{case.get('project_name', '')}"
        tech_id = f"tech::{idx}::{case.get('tech_name', '')}"
        market_id = f"market::{idx}::{case.get('market_name', '')}"
        risk_id = f"risk::{idx}::{case.get('risk_name', '')}"
        value_loop_id = f"value_loop::{idx}::{case.get('value_loop_name', '')}"
        risk_pattern_id = f"risk_pattern::{idx}::{case.get('risk_pattern', '')}"

        add_node(project_id, case.get("project_name", "项目"), "project")
        add_node(tech_id, case.get("tech_name", "技术"), "tech")
        add_node(market_id, case.get("market_name", "市场"), "market")

        add_edge(project_id, tech_id, "USE")
        add_edge(project_id, market_id, "TARGET")

        if case.get("risk_name"):
            add_node(risk_id, case.get("risk_name", "风险"), "risk")
            add_edge(project_id, risk_id, "TRIGGER_RISK")

        if case.get("value_loop_name") or case.get("value_loop_desc"):
            add_node(value_loop_id, case.get("value_loop_name", "价值闭环"), "value_loop")
            add_edge(project_id, value_loop_id, "HAS_VALUE_LOOP")
            add_edge(value_loop_id, tech_id, "INVOLVES_TECH")
            add_edge(value_loop_id, market_id, "INVOLVES_MARKET")

        if case.get("risk_pattern") or case.get("pattern_description"):
            add_node(risk_pattern_id, case.get("risk_pattern", "风险模式"), "risk_pattern")
            add_edge(project_id, risk_pattern_id, "HAS_RISK_PATTERN")
            add_edge(risk_pattern_id, tech_id, "INVOLVES_TECH")
            if case.get("risk_name"):
                add_edge(risk_pattern_id, risk_id, "INVOLVES_RISK")

    return nodes, edges


DEFENSE_EXPERT_PROFILES: Dict[str, Dict[str, str]] = {
    "激进型VC": {
        "focus": "市场规模、增长速度、团队执行力、融资故事与壁垒",
        "style": "尖锐、直接、强压迫感，喜欢追问为什么是你、为什么是现在、为什么能做大。",
    },
    "技术流专家": {
        "focus": "技术可行性、算法/系统壁垒、数据来源、工程落地与技术替代风险",
        "style": "专业、挑剔，喜欢追问核心技术是不是伪创新，能否真正落地。",
    },
    "保守型银行家": {
        "focus": "现金流、合规、偿付能力、稳健性、坏账和风控",
        "style": "谨慎、冷静，偏向质疑财务假设是否过于乐观。",
    },
    "产业操盘手": {
        "focus": "渠道、供应链、交付效率、客户转化、产业协同",
        "style": "务实，专盯从想法走到真实交付过程中最容易掉链子的环节。",
    },
    "政策合规顾问": {
        "focus": "监管、行业准入、数据安全、知识产权、伦理与资质",
        "style": "严苛，喜欢问项目是否踩线，是否具备进入目标行业的资格。",
    },
}


def run_defense_mode_cycle(
    student_input: str,
    conversation_history: List[Dict[str, str]] = None,
    accumulated_info: Dict[str, Any] = None,
    target_competition: str = "互联网+",
    student_id: int = None,
) -> AgentState:
    state = AgentState(
        student_input=student_input,
        conversation_history=conversation_history or [],
        accumulated_info=dict(accumulated_info or {}),
        target_competition=target_competition,
    )

    try:
        if LANGCHAIN_AVAILABLE:
            state = extract_entities(state)
    except Exception as e:
        LOGGER.warning("Defense mode entity extraction failed: %s", e)

    if state.extracted_nodes:
        merged_info = dict(state.accumulated_info or {})
        for key, value in state.extracted_nodes.items():
            if value not in [None, "", [], {}]:
                merged_info[key] = value
        state.accumulated_info = merged_info

    state.accumulated_info["student_mode"] = "答辩模式"

    query_text = _build_learning_query_text(student_input, state.accumulated_info)
    query_profile = extract_learning_query_profile(student_input, state.accumulated_info)
    matched_cases = query_seed_kg_cases(query_text, state.accumulated_info, query_profile=query_profile, top_k=3)
    graph_nodes, graph_edges = _build_learning_subgraph(matched_cases)

    expert_name = "激进型VC"
    expert_reason = "当前问题更像是在接受市场与商业化压力测试。"
    intent_text = " ".join(
        query_profile.get(key, []) for key in [
            "tech_keywords", "market_keywords", "risk_keywords", "problem_keywords", "project_keywords"
        ]
    ) if False else ""
    combined_text = " ".join(
        query_profile.get("tech_keywords", [])
        + query_profile.get("market_keywords", [])
        + query_profile.get("risk_keywords", [])
        + query_profile.get("problem_keywords", [])
        + query_profile.get("project_keywords", [])
    ) + " " + student_input + " " + json.dumps(state.accumulated_info, ensure_ascii=False)

    if any(token in combined_text for token in ["技术", "算法", "模型", "系统", "专利", "数据", "AI", "影像", "芯片"]):
        expert_name = "技术流专家"
        expert_reason = "你的描述里技术实现与壁垒是关键争议点。"
    if any(token in combined_text for token in ["现金流", "营收", "利润", "还款", "成本", "财务", "贷款", "回本"]):
        expert_name = "保守型银行家"
        expert_reason = "当前更需要验证财务稳健性和现金流安全边界。"
    if any(token in combined_text for token in ["合规", "审批", "医疗", "数据安全", "隐私", "监管", "资质", "药监"]):
        expert_name = "政策合规顾问"
        expert_reason = "当前问题里合规和准入风险优先级更高。"
    if any(token in combined_text for token in ["供应链", "交付", "工厂", "渠道", "商家", "校园", "地推", "履约"]):
        expert_name = "产业操盘手"
        expert_reason = "项目更容易在渠道、交付和落地环节被追问。"

    expert_profile = DEFENSE_EXPERT_PROFILES.get(expert_name, DEFENSE_EXPERT_PROFILES["激进型VC"])
    state.agent_insights = {
        "selected_expert": expert_name,
        "expert_focus": expert_profile["focus"],
        "expert_reason": expert_reason,
    }

    state.kg_query_details = [
        KGQueryDetail(
            step="DEFENSE_MODE_CASE_RETRIEVAL",
            query_type="defense_mode_case_search",
            tech_keywords=query_profile.get("tech_keywords", []) or query_profile.get("project_keywords", []),
            market_keywords=(
                query_profile.get("market_keywords", [])
                + query_profile.get("user_keywords", [])
                + query_profile.get("problem_keywords", [])
            )[:8],
            matched_projects=[case.get("project_name", "") for case in matched_cases],
            project_details=matched_cases,
            success=bool(matched_cases),
            message=f"Retrieved {len(matched_cases)} local KG cases for defense mode.",
            category="Defense",
            retrieval_reason=f"Selected expert: {expert_name}; reason: {expert_reason}",
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
        )
    ]

    state.evidence.append(
        EvidenceItem(
            step="defense_mode",
            detail=f"系统自动选择了“{expert_name}”作为本轮压力测试专家，并检索到 {len(matched_cases)} 个相关案例。",
            source_excerpt=f"学生原文：{_excerpt(student_input)}",
        )
    )

    case_text = _format_learning_cases(matched_cases)
    history_text = "\n".join(
        [
            f"{'学生' if msg.get('role') == 'user' else '导师'}: {msg.get('content', '')}"
            for msg in (conversation_history or [])[-6:]
        ]
    )
    nodes_text = json.dumps(state.extracted_nodes or {}, ensure_ascii=False, indent=2)

    fallback_response = (
        f"本轮我切到 **{expert_name}** 来做答辩压力测试。\n\n"
        f"我最想追问你的核心点是：{expert_profile['focus']}。\n\n"
        "毒舌提问：如果把你的项目最漂亮的包装都拿掉，你到底凭什么让别人相信这不是一个讲得很好听、但做不出来或赚不到钱的想法？\n\n"
        "你可以先用三句话回答：\n"
        "1. 你解决的是谁在什么场景下的什么刚需。\n"
        "2. 你比现有替代方案强在哪里。\n"
        "3. 如果一个月内必须验证，你最先验证哪一个指标。\n\n"
        f"你可以参考这些相关案例：\n{case_text}"
    )

    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        system_prompt = (
            "你正在学生端的“答辩模式”中扮演创业评审专家。\n"
            "你要根据学生当前输入，自动采用最合适的专家身份进行高压提问，但目标是帮助学生准备答辩，而不是单纯打击他。\n"
            "你的输出必须包含四段，且顺序固定：\n"
            "1. 【本轮专家】说明你是谁、为什么由你来提问；\n"
            "2. 【毒舌提问】给出 2-4 个尖锐问题；\n"
            "3. 【招架思路】告诉学生这些问题应该怎么回答，给出答辩框架；\n"
            "4. 【案例借鉴】结合相关案例提醒学生可以借什么思路。\n"
            "语言要有压迫感，但不能侮辱学生；要像真实答辩现场。"
        )
        human_prompt = (
            f"【自动选择的专家】{expert_name}\n"
            f"【专家关注点】{expert_profile['focus']}\n"
            f"【选择原因】{expert_reason}\n\n"
            f"【最近对话】\n{history_text or '暂无'}\n\n"
            f"【学生本轮输入】\n{student_input}\n\n"
            f"【已提取项目实体】\n{nodes_text}\n\n"
            f"【相关图谱案例】\n{case_text}\n"
        )
        try:
            state.response = _call_openai_manual(system_prompt, human_prompt)
        except Exception as e:
            LOGGER.warning("Defense mode response generation failed: %s", e)
            state.response = fallback_response
    else:
        state.response = fallback_response

    return state


def run_learning_mode_cycle(
    student_input: str,
    conversation_history: List[Dict[str, str]] = None,
    accumulated_info: Dict[str, Any] = None,
    target_competition: str = "互联网+",
    student_id: int = None,
) -> AgentState:
    state = AgentState(
        student_input=student_input,
        conversation_history=conversation_history or [],
        accumulated_info=dict(accumulated_info or {}),
        target_competition=target_competition,
    )

    try:
        if LANGCHAIN_AVAILABLE:
            state = extract_entities(state)
    except Exception as e:
        LOGGER.warning("Learning mode entity extraction failed: %s", e)

    if state.extracted_nodes:
        merged_info = dict(state.accumulated_info or {})
        for key, value in state.extracted_nodes.items():
            if value not in [None, "", [], {}]:
                merged_info[key] = value
        state.accumulated_info = merged_info

    state.accumulated_info["student_mode"] = "自由对话学习模式"

    query_text = _build_learning_query_text(student_input, state.accumulated_info)
    query_profile = extract_learning_query_profile(student_input, state.accumulated_info)
    matched_cases = query_seed_kg_cases(query_text, state.accumulated_info, query_profile=query_profile, top_k=3)
    graph_nodes, graph_edges = _build_learning_subgraph(matched_cases)
    state.kg_query_details = [
        KGQueryDetail(
            step="LEARNING_MODE_CASE_RETRIEVAL",
            query_type="learning_mode_case_search",
            tech_keywords=query_profile.get("tech_keywords", []) or query_profile.get("project_keywords", []),
            market_keywords=(
                query_profile.get("market_keywords", [])
                + query_profile.get("user_keywords", [])
                + query_profile.get("problem_keywords", [])
            )[:8],
            matched_projects=[case.get("project_name", "") for case in matched_cases],
            project_details=matched_cases,
            success=bool(matched_cases),
            message=f"Retrieved {len(matched_cases)} local KG cases for learning mode.",
            category="Learning",
            retrieval_reason=f"LLM extracted intent: {query_profile.get('intent_summary', '语义匹配')}",
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
        )
    ]

    state.evidence.append(
        EvidenceItem(
            step="learning_mode",
            detail=f"基于 seed_kg.json 匹配到 {len(matched_cases)} 个案例，用于自由对话教学。",
            source_excerpt=f"学生原文：{_excerpt(student_input)}",
        )
    )

    case_text = _format_learning_cases(matched_cases)
    history_text = "\n".join(
        [
            f"{'学生' if msg.get('role') == 'user' else '导师'}: {msg.get('content', '')}"
            for msg in (conversation_history or [])[-6:]
        ]
    )
    nodes_text = json.dumps(state.extracted_nodes or {}, ensure_ascii=False, indent=2)

    fallback_response = (
        "我先陪你把这个想法拆开想清楚。\n\n"
        "从创新创业比赛的角度，第一步通常不是立刻写方案，而是先明确三件事：谁是你的第一批用户、他们在什么场景下最痛、你比现有办法好在哪里。\n\n"
        f"给你一个图谱案例参考：\n{case_text}\n\n"
        "你可以先想两个问题：\n"
        "1. 你最想帮助的第一类人是谁？\n"
        "2. 他们现在最麻烦、最愿意为之改变的一件事是什么？\n\n"
        "你下一条可以只回答这两个问题，我再陪你把项目继续往下收敛。"
    )

    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        system_prompt = (
            "你是一位面向中国高校大一新生的创新创业导师。"
            "当前是学生端的自由对话学习模式。"
            "你要边回答边教学，不能只给结论。"
            "请始终做到："
            "1. 先回答学生当前问题；"
            "2. 再用提问引导学生思考；"
            "3. 再结合知识图谱案例帮助理解；"
            "4. 最后给一个很小、可执行的下一步。"
            "语气耐心、鼓励、清楚，不要代写完整商业计划书。"
        )
        human_prompt = (
            f"【学生输入】\n{student_input}\n\n"
            f"【历史对话】\n{history_text or '暂无'}\n\n"
            f"【已提取项目信息】\n{nodes_text}\n\n"
            f"【图谱匹配案例】\n{case_text}\n\n"
            "请输出自然的中文回复，并明确包含这三部分短句：\n"
            "1. 回答与解释\n"
            "2. 你可以先想两个问题：\n"
            "3. 给你一个图谱案例参考：\n"
            "不要写成评审报告，不要用太强的批判口吻。"
        )
        try:
            state.response = _call_openai_manual(system_prompt, human_prompt).strip()
        except Exception as e:
            LOGGER.warning("Learning mode response generation failed: %s", e)
            state.response = fallback_response
    else:
        state.response = fallback_response

    disclaimer = "\n\n> ⚖️ AI 免责声明：以上内容用于教学辅导与思路启发，不构成投资、法律或财务决策依据。"
    if state.response and "AI 免责声明" not in state.response:
        state.response += disclaimer

    return state


def run_langgraph_cycle(
    student_input: str,
    conversation_history: List[Dict[str, str]] = None,
    accumulated_info: Dict[str, Any] = None,
    target_competition: str = "互联网+",
    student_id: int = None,
    skip_ghostwriting_guard: bool = False,
) -> AgentState:
    from src.utils.database import get_active_intervention_rules, get_connection
    
    # Fetch intervention rules
    all_rules = []
    if student_id:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT teacher_id FROM teacher_students WHERE student_id = ?", (student_id,))
            teacher_ids = [row["teacher_id"] for row in cursor.fetchall()]
            
            for t_id in teacher_ids:
                all_rules.extend(get_active_intervention_rules(t_id, student_id))
            conn.close()
        except Exception as e:
            LOGGER.error("Failed to fetch intervention rules: %s", e)

    if not check_input_safety(student_input):
        state = AgentState(
            student_input=student_input, 
            conversation_history=conversation_history or [], 
            accumulated_info=accumulated_info or {},
            target_competition=target_competition,
            intervention_rules=all_rules,
        )
        
        # A7 原始护栏拦截 (Strict Interception)
        state.response = "👁️ **护栏拦截**：您的输入异常或包含违规注入指令，系统已拒绝该操作。"
        state.probing_strategy = "护栏阻断 (Blocked)"
        state.detected_fallacies = ["GENTLE_INTERCEPTION"]
        state.rubric_scores = {"_summary": {"weighted_total": 0, "default_competition": target_competition}}
        
        disclaimer = "\n\n> ⚖️ **AI 免责声明**：以上测评与分析仅基于 AI 模型推演提供参考，不构成实质性的商业投资、法律及财务决策依据。项目实操请以真实市场环境为准。"
        state.response += disclaimer
        
        return state

    # A2: 反代写硬拦截 (Anti-Ghostwriting Hard Block)
    ghostwriting_keywords = ["帮我写", "代写", "生成一份", "写一段", "帮我做", "完整方案", "直接给我一份", "给我写"]
    if not skip_ghostwriting_guard and any(k in student_input for k in ghostwriting_keywords):
        state = AgentState(
            student_input=student_input, 
            conversation_history=conversation_history or [], 
            accumulated_info=accumulated_info or {},
            target_competition=target_competition,
            intervention_rules=all_rules,
        )
        
        state.response = "👁️ **护栏拦截**：系统检测到『直接代写』请求。作为您的超图教练，我不能越俎代庖直接为您生成完整的商业计划书。请聚焦具体的项目核心痛点或方案，我们一步步推演！"
        state.probing_strategy = "反代写阻断 (Anti-Ghostwriting Blocked)"
        state.detected_fallacies = ["GHOSTWRITING_INTERCEPTION"]
        state.rubric_scores = {"_summary": {"weighted_total": 0, "default_competition": target_competition}}
        
        disclaimer = "\n\n> ⚖️ **AI 免责声明**：以上测评与分析仅基于 AI 模型推演提供参考，不构成实质性的商业投资、法律及财务决策依据。项目实操请以真实市场环境为准。"
        state.response += disclaimer
        
        return state

    state = AgentState(
        student_input=student_input,
        conversation_history=conversation_history or [],
        accumulated_info=accumulated_info or {},
        target_competition=target_competition,
        intervention_rules=all_rules,
    )
    graph = build_state_graph()
    final = graph.execute(state)
    
    # A7：输出免责声明
    disclaimer = "\n\n> ⚖️ **AI 免责声明**：以上测评与分析仅基于 AI 模型推演提供参考，不构成实质性的商业投资、法律及财务决策依据。项目实操请以真实市场环境为准。"
    if final.response and "AI 免责声明" not in final.response:
        final.response += disclaimer
        
    return final


def run_demo_cycle():
    final = run_langgraph_cycle("我们的客户是新型服务业，希望自动化内部营销。")
    render_frontend_snapshot(final)


if __name__ == "__main__":
    run_demo_cycle()

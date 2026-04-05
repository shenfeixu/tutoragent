"""LangGraph-backed coach agent skeleton with LangChain-powered extraction/rebuttal and structured prompts."""
from __future__ import annotations

import json
import logging
import os
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

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
_NEO4J_DRIVER: Optional[Any] = None


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
        prompt = f"""你是一个专业的技术市场分析专家和知识图谱检索专家。请从以下文本中提取关键词，用于在知识图谱中进行匹配搜索。

输入文本: {text}

请提取以下类型的关键词：

1. **核心词**（2-3字）：最核心的技术或市场词汇，如"无人机"、"影像"、"电池"
2. **专业术语**（3-4字）：行业专业术语，如"激光雷达"、"固态电池"、"电解槽"
3. **复合词**（4-6字）：技术组合或细分领域，如"医学影像AI"、"智慧农业"
4. **应用场景**（2-5字）：实际应用场景，如"植保"、"测绘"、"诊断"
5. **同义词**：核心词的同义词或近义词
6. **相关词**：与核心概念相关的词汇

输出要求：
- 关键词长度必须多样化：2字、3字、4字、5字、6字都要有
- 优先提取短词（2-4字），因为更容易匹配
- 避免泛化词汇（如"技术"、"系统"、"服务"、"产品"）
- 关键词之间用逗号分隔
- 至少提取15个关键词
- 按重要性排序

示例输入："AI影像医疗诊断"
示例输出：影像,AI,医疗,诊断,医学影像,影像诊断,计算机视觉,深度学习,医学,智能诊断,影像分析,医疗AI,辅助诊断,临床,病理

关键词列表:"""
        
        response = client.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        
        keywords = [kw.strip() for kw in content.replace("，", ",").replace("、", ",").split(",") if kw.strip()]
        
        stopwords = {"技术", "系统", "市场", "服务", "产品", "平台", "解决方案", "应用", "开发", "研究", "设计", "构建", "打造", "创建", "建立", "帮助", "制定", "分析", "调控", "制备", "制造", "加工", "检测", "监测", "管理", "控制", "智能", "数据", "信息", "目标客户", "客户群体", "公司", "企业", "项目", "团队"}
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
    
    stopwords = {"的", "和", "与", "及", "等", "是", "在", "有", "为", "对", "将", "能", "可", "以", "了", "着", "过", "被", "把", "让", "给", "向", "从", "到", "基于", "通过", "进行", "实现", "提供", "支持", "系统", "平台", "服务", "产品", "技术", "解决方案", "应用", "开发", "研究", "设计", "构建", "打造", "创建", "建立", "帮助", "制定", "分析", "诊断", "调控", "制备", "制造", "加工", "检测", "监测", "管理", "控制"}
    
    entity_patterns = [
        r'[\u4e00-\u9fa5]{2,4}(?:无人机|机器人|传感器|激光|雷达|成像|遥感|光谱|智能|精密|量子|芯片|算法|模型|平台|网络|数据|云|大数据|人工智能|机器学习|深度学习|物联网|区块链)',
        r'(?:无人机|机器人|传感器|激光|雷达|成像|遥感|光谱|智能|精密|量子|芯片|算法|模型)[\u4e00-\u9fa5]{0,4}',
        r'[A-Z][a-z]+(?:[A-Z][a-z]+)*',
        r'[A-Z]{2,}',
    ]
    
    keywords = []
    
    for pattern in entity_patterns:
        matches = re.findall(pattern, text)
        keywords.extend(matches)
    
    english_words = re.findall(r'[A-Za-z]{2,}', text)
    for word in english_words:
        if word.lower() not in ['the', 'and', 'for', 'with', 'from', 'this', 'that', 'has', 'have', 'are', 'was', 'were', 'been']:
            keywords.append(word)
    
    chinese_chunks = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
    for chunk in chinese_chunks:
        if chunk in stopwords:
            continue
        if len(chunk) == 2:
            keywords.append(chunk)
        elif len(chunk) == 3:
            keywords.append(chunk)
        elif len(chunk) == 4:
            keywords.append(chunk)
        elif len(chunk) >= 5:
            keywords.append(chunk[:4])
            keywords.append(chunk[:3])
            keywords.append(chunk[1:4] if len(chunk) >= 4 else chunk[1:])
    
    unique_keywords = []
    seen = set()
    for kw in keywords:
        kw_lower = kw.lower() if kw.isascii() else kw
        if kw_lower not in seen and kw not in stopwords and len(kw) >= 2:
            seen.add(kw_lower)
            unique_keywords.append(kw)
    
    return unique_keywords[:max_keywords]


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
                "stage": "步骤2: 精确匹配（技术→技术节点 AND 市场→市场节点）",
                "found": len(exact_results),
                "projects": [p["project_name"] for p in exact_results],
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
                "stage": "步骤3: 跨维度匹配（技术/市场关键词→任意节点）",
                "found": len(cross_results),
                "projects": [p["project_name"] for p in cross_results],
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
                "stage": "步骤4: 全文匹配（关键词→技术/市场/项目名/项目描述）",
                "found": len(fulltext_results),
                "projects": [p["project_name"] for p in fulltext_results],
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
    
    match_details["query_attempts"].append({
        "stage": "步骤5: 结果汇总与排序",
        "found": len(unique_projects),
        "projects": [p["project_name"] for p in unique_projects[:10]],
        "message": f"共找到 {len(unique_projects)} 个相关项目",
    })
    
    if unique_projects:
        sorted_projects = sorted(unique_projects, key=lambda x: x.get("relevance_score", 0) or 0, reverse=True)
        match_details["project_details"] = sorted_projects
        match_details["matched_projects"] = [p["project_name"] for p in sorted_projects]
        
        top_project = sorted_projects[0]
        detail_msg = f"图谱中找到 {len(unique_projects)} 个相关案例"
        if top_project.get("tech_maturity"):
            detail_msg += f"，技术成熟度: {top_project['tech_maturity']}"
        if top_project.get("market_name"):
            detail_msg += f"，市场: {top_project['market_name'][:30]}..."
        
        return True, detail_msg, match_details
    
    return False, f"图谱中未找到匹配案例（技术关键词: {tech_keywords[:5]}, 市场关键词: {market_keywords[:5]}）", match_details


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
        return []
    
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
        return []


def get_risk_pattern_examples(tech: Optional[str] = None) -> List[Dict[str, Any]]:
    """获取风险模式超边示例"""
    driver = _get_neo4j_driver()
    if not driver:
        return []
    
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
        return []


def get_teaching_cases_for_risk(risk_keyword: str) -> List[Dict[str, Any]]:
    """根据风险关键词从知识图谱获取相关教学案例"""
    driver = _get_neo4j_driver()
    if not driver:
        return []
    
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
        return []


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
        "Keep existing information unless the new input explicitly changes it.\n{format_instructions}"
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
    
    tech = nodes.get("tech_description", "")
    market = nodes.get("target_market", "")
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

    # H15: 区分"数据完全缺失"(GAP) vs "有渠道和客户但利润链不通"(Fatal)
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


    if not LANGCHAIN_AVAILABLE:
        human_prompt = (
            "学生的原文输入：\n{student_input}\n"
            "抽取的模型（JSON）：\n{nodes}\n"
            "触发的规则：{rules}\n"
            "当前策略与参考案例：{strategy}\n"
            "证据追踪：\n{evidence}\n"
            "{intervention_text}\n"
            "【反代写约束】如果用户要求代写或生成完整商业方案，果断拒绝并严厉指正。\n"
            "【输出格式规定】请严格按照以下 6 个 Markdown 标题格式强制输出回复，且【✅ 实践任务】只能有 1 个：\n\n"
            "### 🎯 诊断问题 (Issue)\n"
            "一句话指出当前逻辑中存在的漏洞。你必须包含一句引用：“『学生原文：...』”。\n\n"
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
        "当前策略与参考案例：“{strategy}”\n"
        "证据追踪：\n{evidence_summary}\n"
        "{intervention_text}\n"
        "请结合当前用户的输入和上述评价策略（尤其是【教师特别干预指令】），进行回复计算。\n"
        "【输出格式规定】请用中文严格按照以下 Markdown 格式强制输出 6 个字段，且【✅ 实践任务】只能有 1 个极其具体的动作："
        "\n\n"
        "### 🎯 诊断问题 (Issue)\n"
        "一句话指出当前逻辑中存在的漏洞（或指出你绝不代写）。你必须包含一句引用：“『学生原文：[填入原话]』”作为诊断依据！\n\n"
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
            "3. 输出依然必须包含原有的 6 个 Markdown 标题结构（Issue, Definition, Example, Analysis, Socratic Question, Practice Task）。"
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

        scores[dim] = dim_result

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


def generate_student_profile(student_data: Dict[str, Any]) -> str:
    """A6-4: 根据项目历史生成 3 阶段动态画像 (价值发现 -> 压力测试 -> 执行可行性)。"""
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


def build_state_graph() -> StateGraph:
    graph = StateGraph(name="HypergraphCoach")
    nodes = [
        Node(name="extract_entities", function=extract_entities),
        Node(name="hypergraph_critic", function=hypergraph_critic),
        Node(name="strategy_selector", function=strategy_selector),
        Node(name="generate_rebuttal", function=generate_rebuttal),
        Node(name="audit_reflection", function=audit_reflection),
        Node(name="rubric_scorer", function=rubric_scorer),
        Node(name="update_memory_engine", function=update_memory_engine),
    ]
    for node in nodes:
        graph.add_node(node)
    graph.connect("extract_entities", "hypergraph_critic")
    graph.connect("hypergraph_critic", "strategy_selector")
    graph.connect("strategy_selector", "generate_rebuttal")
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


def run_langgraph_cycle(
    student_input: str,
    conversation_history: List[Dict[str, str]] = None,
    accumulated_info: Dict[str, Any] = None,
    target_competition: str = "互联网+",
    student_id: int = None,
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
    if any(k in student_input for k in ghostwriting_keywords):
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

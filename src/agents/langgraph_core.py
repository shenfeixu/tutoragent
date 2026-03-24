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
    source_excerpt: str = Field(..., description="Excerpt from user input that gave rise to the evidence.")


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
    "H1": "广义竞争逻辑：你提到的技术-市场组合在图谱中没有先例。请说明你的技术壁垒能否抵御现有竞争者的快速跟进？",
    "H2": "技术成熟度逻辑：技术处于早期阶段，请说明从原型到商业化的关键里程碑和验证节点。",
    "H3": "用户画像逻辑：目标客户描述模糊。请用「谁、在什么场景、遇到什么问题」三要素重新定义。",
    "H4": "市场口径逻辑：TAM/SAM/SOM 比例异常。请说明这三个数字的计算依据和数据来源。",
    "H5": "价值主张逻辑：价值主张不够清晰。请用「我们帮助X用户解决Y问题，实现Z价值」的句式重新表述。",
    "H6": "渠道触达逻辑：获客渠道描述不足。请说明你的目标客户在哪里聚集，你如何触达他们。",
    "H7": "收入验证逻辑：收入预测缺失。请说明你的定价策略和付费意愿验证情况。",
    "H8": "单位经济逻辑：LTV/CAC 比例不健康。请说明你的客户留存率和复购周期。",
    "H9": "团队能力逻辑：团队规模偏小。请说明核心团队的技能互补性和执行力验证。",
    "H10": "融资匹配逻辑：融资阶段与技术成熟度不匹配。请说明资金用途和里程碑规划。",
    "H11": "时间规划逻辑：上市时间过于乐观。请说明关键路径和潜在瓶颈。",
    "H12": "风险识别逻辑：风险识别不充分。请列出技术、市场、团队三方面的主要风险。",
    "H13": "技术壁垒逻辑：技术描述过短。请说明你的核心技术壁垒和专利布局。",
    "H14": "市场验证逻辑：市场规模数据可能缺乏验证。请说明市场调研的方法和样本量。",
    "H15": "商业闭环逻辑：商业模式闭环不完整。请说明从获客到变现的完整路径。",
    "H16": "单位经济幻觉逻辑：你的单位经济模型存在矛盾。请说明边际成本是否随规模下降，还是固定成本占主导？",
    "H17": "渠道错位逻辑：你的渠道与目标用户群体可能不匹配。请说明渠道的用户画像与你的目标客户重合度。",
    "H18": "现金流生存逻辑：请说明你的现金跑道和盈亏平衡点。在没有新融资的情况下能支撑多久？",
    "H19": "护城河逻辑：你的竞争优势是否可持续？请说明网络效应、规模效应或品牌效应如何形成。",
    "H20": "增长飞轮逻辑：你的增长模式是否自增强？请说明新用户如何帮助获取更多用户。",
}
DEFAULT_FALLACY_STRATEGY = "继续逐条验证关键前提，别让幻觉的数字偷偷爬过审计线。"

# ── A5: 赛事 Rubric 评分维度与权重 ──────────────────────────────────
RUBRIC_FALLACY_MAP: Dict[str, List[str]] = {
    "pain_point":    ["H3", "H5"],
    "planning":      ["H6", "H11"],
    "modeling":       ["H4", "H7", "H8", "H16"],
    "leverage":       ["H9", "H10", "H18"],
    "presentation":   ["H13", "H19", "H20"],
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


def extract_keywords_with_llm(text: str, max_keywords: int = 5) -> List[str]:
    """使用 LLM 从文本中提取关键词"""
    if not text:
        return []
    
    try:
        client = _get_chat_client()
        prompt = f"""从以下文本中提取 {max_keywords} 个最重要的技术或市场关键词，用于在知识图谱中搜索匹配。

文本: {text}

要求:
1. 提取核心技术名词、市场领域名词
2. 每个关键词 2-4 个字，越短越好
3. 优先提取专有名词、技术术语、核心概念
4. 避免提取泛化词汇如"技术"、"系统"、"市场"、"服务"
5. 只输出关键词，用逗号分隔，不要其他解释

关键词:"""
        
        response = client.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        
        keywords = [kw.strip() for kw in content.replace("，", ",").split(",") if kw.strip()]
        keywords = [kw for kw in keywords if len(kw) >= 2 and kw not in ["技术", "系统", "市场", "服务", "产品", "平台", "解决方案"]]
        return keywords[:max_keywords]
    except Exception as e:
        LOGGER.warning(f"LLM 关键词提取失败，回退到本地算法: {e}")
        return extract_keywords_local(text, max_keywords)


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
    """检查技术是否适用于目标市场（基于Neo4j图谱），返回详细匹配过程"""
    match_details = {
        "tech_original": tech,
        "market_original": market,
        "tech_keywords": [],
        "market_keywords": [],
        "query_attempts": [],
        "matched_projects": [],
    }
    
    if not tech or not market:
        return False, "缺少技术描述或目标市场，无法完成技术-市场匹配校验。", match_details
    driver = _get_neo4j_driver()
    if not driver:
        return False, "Neo4j 连接不可用；请确认环境变量。", match_details

    tech_keywords = extract_keywords_with_llm(tech, max_keywords=5)
    market_keywords = extract_keywords_with_llm(market, max_keywords=5)
    
    match_details["tech_keywords"] = tech_keywords
    match_details["market_keywords"] = market_keywords
    
    if not tech_keywords or not market_keywords:
        return False, f"无法从描述中提取有效关键词。技术关键词: {tech_keywords}, 市场关键词: {market_keywords}", match_details

    all_results = []
    
    for tech_kw in tech_keywords:
        for market_kw in market_keywords:
            query = """
            MATCH (p:Project)-[:USE]->(t:Tech)
            MATCH (p)-[:TARGET]->(m:Market)
            WHERE t.name CONTAINS $tech_kw AND m.name CONTAINS $market_kw
            RETURN distinct p.name AS project
            LIMIT 3
            """
            try:
                with driver.session(database=NEO4J_DATABASE) as session:
                    result = session.run(query, tech_kw=tech_kw, market_kw=market_kw)
                    projects = [record["project"] for record in result]
                    match_details["query_attempts"].append({
                        "tech_keyword": tech_kw,
                        "market_keyword": market_kw,
                        "found": len(projects),
                        "projects": projects,
                    })
                    all_results.extend(projects)
            except Exception as e:
                LOGGER.warning(f"H1 query failed for {tech_kw}/{market_kw}: {e}")
                match_details["query_attempts"].append({
                    "tech_keyword": tech_kw,
                    "market_keyword": market_kw,
                    "found": 0,
                    "error": str(e),
                })
    
    if not all_results:
        for tech_kw in tech_keywords:
            query_tech_only = """
            MATCH (p:Project)-[:USE]->(t:Tech)
            WHERE t.name CONTAINS $tech_kw
            RETURN distinct p.name AS project
            LIMIT 3
            """
            try:
                with driver.session(database=NEO4J_DATABASE) as session:
                    result = session.run(query_tech_only, tech_kw=tech_kw)
                    projects = [record["project"] for record in result]
                    if projects:
                        match_details["query_attempts"].append({
                            "tech_keyword": tech_kw,
                            "market_keyword": "(任意)",
                            "found": len(projects),
                            "projects": projects,
                            "mode": "技术单匹配",
                        })
                        all_results.extend(projects)
            except Exception as e:
                LOGGER.warning(f"H1 tech-only query failed: {e}")
        
        for market_kw in market_keywords:
            query_market_only = """
            MATCH (p:Project)-[:TARGET]->(m:Market)
            WHERE m.name CONTAINS $market_kw
            RETURN distinct p.name AS project
            LIMIT 3
            """
            try:
                with driver.session(database=NEO4J_DATABASE) as session:
                    result = session.run(query_market_only, market_kw=market_kw)
                    projects = [record["project"] for record in result]
                    if projects:
                        match_details["query_attempts"].append({
                            "tech_keyword": "(任意)",
                            "market_keyword": market_kw,
                            "found": len(projects),
                            "projects": projects,
                            "mode": "市场单匹配",
                        })
                        all_results.extend(projects)
            except Exception as e:
                LOGGER.warning(f"H1 market-only query failed: {e}")
    
    unique_projects = list(dict.fromkeys(all_results))
    match_details["matched_projects"] = unique_projects
    
    if unique_projects:
        return True, f"图谱中找到 {len(unique_projects)} 个相关案例：{', '.join(unique_projects[:3])}", match_details
    return False, f"图谱中未找到匹配案例（技术关键词: {tech_keywords}, 市场关键词: {market_keywords}）", match_details


def check_tech_risks(tech: Optional[str]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """H12: 检查技术相关风险，返回详细匹配过程"""
    risk_details = {
        "tech_original": tech,
        "tech_keywords": [],
        "query_attempts": [],
        "risks_found": [],
    }
    
    if not tech:
        return False, [], risk_details
    driver = _get_neo4j_driver()
    if not driver:
        return False, [], risk_details
    
    tech_keywords = extract_keywords_with_llm(tech)
    risk_details["tech_keywords"] = tech_keywords
    
    if not tech_keywords:
        return False, [], risk_details

    query = """
    MATCH (p:Project)-[:USE]->(t:Tech)
    MATCH (p)-[:TRIGGER_RISK]->(r:Risk)
    WHERE ANY(kw IN $tech_keywords WHERE t.name CONTAINS kw)
    RETURN distinct r.name AS risk
    LIMIT 5
    """
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, tech_keywords=tech_keywords)
            risks = [record["risk"] for record in result]
            risk_details["risks_found"] = risks
            risk_details["query_attempts"].append({
                "keywords": tech_keywords,
                "found": len(risks),
            })
            return len(risks) > 0, risks, risk_details
    except Exception as e:
        LOGGER.warning(f"H12 check failed: {e}")
        risk_details["query_attempts"].append({
            "keywords": tech_keywords,
            "error": str(e),
        })
        return False, [], risk_details


def get_value_loop_examples(tech: Optional[str] = None, market: Optional[str] = None) -> List[Dict[str, Any]]:
    """获取价值闭环超边示例"""
    driver = _get_neo4j_driver()
    if not driver:
        return []
    
    tech_keywords = extract_keywords(tech) if tech else []
    market_keywords = extract_keywords(market) if market else []
    
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
    
    tech_keywords = extract_keywords(tech) if tech else []
    
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
        raise RuntimeError(f"实体提取失败：{exc}") from exc
    return state


def hypergraph_critic(state: AgentState) -> AgentState:
    """Runs H1-H20 checks against the hypergraph inputs."""
    failures: List[str] = []
    nodes = state.extracted_nodes
    excerpt = "[已分析]"

    tech = nodes.get("tech_description", "")
    market = nodes.get("target_market", "")
    h1_passed, h1_detail, h1_match_details = check_tech_market_match(tech, market)
    
    keyword_match_info = f"技术关键词: {h1_match_details.get('tech_keywords', [])}, 市场关键词: {h1_match_details.get('market_keywords', [])}"
    query_attempts = h1_match_details.get('query_attempts', [])
    query_info = " | ".join([f"({q['tech_keyword']}+{q['market_keyword']}→{q.get('found', 0)}条)" for q in query_attempts[:3]])
    
    state.evidence.append(EvidenceItem(
        step="H1", 
        detail=f"{h1_detail} | {keyword_match_info} | 查询: {query_info}", 
        source_excerpt=excerpt
    ))
    if not h1_passed:
        failures.append("H1")

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
    if not (tam >= sam >= som and tam > 0 and sam > 0 and som > 0):
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
    if revenue <= 0:
        failures.append("H7")
        state.evidence.append(EvidenceItem(step="H7", detail="收入预测为零或缺失。", source_excerpt=excerpt))
    else:
        state.evidence.append(EvidenceItem(step="H7", detail=f"预测收入：{revenue:,.0f}元", source_excerpt=excerpt))

    ltv = _safe_float(nodes.get("LTV"))
    cac = _safe_float(nodes.get("CAC"))
    ltv_detail = f"LTV={ltv:,.0f}, CAC={cac:,.0f}"
    if ltv < 3 * cac:
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

    if tech and len(tech) < 20:
        failures.append("H13")
        state.evidence.append(EvidenceItem(
            step="H13",
            detail=f"技术描述过短，技术壁垒不明确。",
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

    if channel and customer and revenue > 0:
        if ltv > 0 and cac > 0:
            state.evidence.append(EvidenceItem(
                step="H15",
                detail=f"商业模式闭环完整：渠道→客户→收入→利润。",
                source_excerpt=excerpt,
            ))
        else:
            failures.append("H15")
            state.evidence.append(EvidenceItem(
                step="H15",
                detail="商业模式闭环不完整，缺少LTV/CAC数据。",
                source_excerpt=excerpt,
            ))
    else:
        failures.append("H15")
        state.evidence.append(EvidenceItem(
            step="H15",
            detail="商业模式闭环不完整，缺少关键要素。",
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
        channel_keywords = ["线上", "线下", "B端", "C端", "直销", "代理", "平台"]
        customer_keywords = ["企业", "个人", "政府", "学生", "老人", "儿童"]
        channel_type = "B端" if any(k in channel for k in ["企业", "B端", "行业"]) else "C端" if any(k in channel for k in ["个人", "C端", "消费者"]) else ""
        customer_type = "B端" if any(k in customer for k in ["企业", "公司", "机构"]) else "C端" if any(k in customer for k in ["个人", "用户", "消费者"]) else ""
        if channel_type and customer_type and channel_type != customer_type:
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
    if tech and not moat:
        failures.append("H19")
        state.evidence.append(EvidenceItem(
            step="H19",
            detail=f"护城河未明确：有技术但未说明竞争壁垒如何形成。",
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

    state.detected_fallacies = failures
    return state
def strategy_selector(state: AgentState) -> AgentState:
    """Chooses the Socratic probing strategy based on the triggered rule set."""
    if not state.detected_fallacies:
        state.probing_strategy = "Confirm assumptions then push for execution detail."
        return state

    for failure in state.detected_fallacies:
        if failure in FALLACY_STRATEGY_LIBRARY:
            strategy_text = FALLACY_STRATEGY_LIBRARY[failure]
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
    strategy_text = state.probing_strategy or DEFAULT_FALLACY_STRATEGY

    rules = state.detected_fallacies or ["H1-H15 均未触发"]
    evidence_summary = _format_evidence(state.evidence)

    if not LANGCHAIN_AVAILABLE:
        human_prompt = (
            "学生的原文输入：\n{student_input}\n"
            "抽取的模型（JSON）：\n{nodes}\n"
            "触发的规则：{rules}\n"
            "当前策略与参考案例：{strategy}\n"
            "证据追踪：\n{evidence}\n"
            "【反代写约束】如果用户要求代写或生成完整商业方案，果断拒绝并严厉指正。\n"
            "【输出格式规定】请严格按照以下 6 个 Markdown 标题格式强制输出回复，且【✅ 实践任务 (Practice Task)】只能有 1 个：\n"
            "### 🎯 诊断问题 (Issue)\n"
            "### 📖 概念解析 (Definition)\n"
            "### 💡 案例参考 (Example)\n"
            "### 🔍 具体分析 (Analysis)\n"
            "### 🤔 反思追问 (Socratic Question)\n"
            "必须使用提供的超边参照案例作为弹药，进行极具压迫感的非直给追问。\n"
            "### ✅ 实践任务 (Practice Task)"
        ).format(
            student_input=state.student_input,
            nodes=json.dumps(state.extracted_nodes or {}, ensure_ascii=False, indent=2),
            rules="、".join(rules),
            strategy=strategy_text,
            evidence=evidence_summary,
        )
        manual_system = (
            "You are a sharp venture coach whose tone is professional, precise, and tinged with cold humor."
            " You do not provide solutions; you only expose logic holes."
        )
        try:
            content = _call_openai_manual(manual_system, human_prompt).strip()
            state.response = content
        except Exception as exc:
            LOGGER.warning("OpenAI 直接调用失败：%s", exc)
            issues = "、".join(state.detected_fallacies) if state.detected_fallacies else "无"
            state.response = (
                f"在当前轮次我发现以下规则被触发：{issues}。请按照策略 '{strategy_text}' 进一步澄清。"
            )
        return state

    system_prompt = (
        "You are a sharp venture coach whose tone is professional, precise, and tinged with cold humor. "
        "You never hand out answers, only diagnostics that force the student to self-correct.\n"
        "【ANTI-GHOSTWRITING GUARDRAIL】如果你检测到用户的『原文输入』中在要求你直接生成完整的商业计划书(BP)、电梯演讲稿、或直接为你写出项目的段落内容，"
        "你必须果断且清晰地拒绝。你的角色是创新创业教练，绝不能越俎代庖代写任何文档代码。你必须在你的『🎯 诊断问题 (Issue)』字段优先表达对这种代写要求的严厉拒绝，随后引导对话回到商业逻辑的校验上。"
    )
    human_template = (
        "学生的原文输入：\n{student_input}\n"
        "抽取的模型（JSON）：\n{nodes}\n"
        "触发的规则：{rules}\n"
        "当前策略与参考案例：“{strategy}”\n"
        "证据追踪：\n{evidence_summary}\n"
        "请结合当前用户的输入和上述评价策略，进行回复计算。\n"
        "【输出格式规定】请用中文严格按照以下 Markdown 格式强制输出 6 个字段，且【✅ 实践任务】只能有 1 个极其具体的动作："
        "\n\n"
        "### 🎯 诊断问题 (Issue)\n"
        "一句话指出当前逻辑中存在的漏洞（或指出你绝不代写）。\n\n"
        "### 📖 概念解析 (Definition)\n"
        "客观解释该谬误涉及的创新创业概念或原理。\n\n"
        "### 💡 案例参考 (Example)\n"
        "给出一个相关的正向或反向商业案例以加强理解。\n\n"
        "### 🔍 具体分析 (Analysis)\n"
        "基于前面抽取的业务模型，具体剖析这个想法为何站不住脚。\n\n"
        "### 🤔 反思追问 (Socratic Question)\n"
        "【压力追问闭环】：你必须直接引用“当前策略与参考案例”中提供的超边参照案例（如果有），以此为论据给学生施加真实的生存压力拷问（例如直戳现金流断裂或技术壁垒可复制的风险），抛出一个犀利、直击痛点的开放性问题。\n\n"
        "### ✅ 实践任务 (Practice Task)\n"
        "给出一个并且只能给出一个具体的课后或者后续补充任务（例如查算某个特定数据）。"
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
        ).to_messages()
        response = _get_chat_client().invoke(prompt_payload)
        content = getattr(response, "content", None) or (response.choices[0].message.content if response.choices else "")
        state.response = content.strip()
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Rebuttal generation failed; falling back messaging (%s)", exc)
        issues = "、".join(state.detected_fallacies) if state.detected_fallacies else "无"
        state.response = (
            f"在当前轮次我发现以下规则被触发：{issues}。请按照策略 '{strategy_text}' 进一步澄清。"
        )
    return state


def rubric_scorer(state: AgentState) -> AgentState:
    """A5: 根据 Rubric 维度对项目进行 0-5 逐项评分，并为薄弱项生成 Missing Evidence + Minimal Fix。"""
    fallacies = set(state.detected_fallacies)
    scores: Dict[str, Any] = {}

    for dim, related_rules in RUBRIC_FALLACY_MAP.items():
        triggered = [r for r in related_rules if r in fallacies]
        total_rules = len(related_rules)
        fail_ratio = len(triggered) / total_rules if total_rules > 0 else 0

        # 0-5 分：全部通过=5, 全部未通过=0
        raw_score = round(5 * (1 - fail_ratio))
        raw_score = max(0, min(5, raw_score))  # clamp

        dim_result: Dict[str, Any] = {
            "score": raw_score,
            "name": RUBRIC_DIM_NAMES[dim],
            "triggered_rules": triggered,
        }

        if raw_score <= 2:
            dim_result["missing_evidence"] = RUBRIC_MISSING_FIX[dim]["missing"]
            dim_result["minimal_fix"] = RUBRIC_MISSING_FIX[dim]["fix"]

        scores[dim] = dim_result

    # 按默认赛事（互联网+）计算加权综合分
    default_weights = COMPETITION_WEIGHTS["互联网+"]
    weighted_total = sum(
        scores[dim]["score"] * default_weights.get(dim, 0.2) for dim in scores
    )

    scores["_summary"] = {
        "weighted_total": round(weighted_total, 2),
        "default_competition": "互联网+",
        "available_competitions": list(COMPETITION_WEIGHTS.keys()),
    }

    state.rubric_scores = scores
    LOGGER.info("【A5 Rubric 评分】%s", {d: scores[d]["score"] for d in RUBRIC_FALLACY_MAP})
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
    """A6: 根据单个学生的历史记录和得分生成动态能力画像与教师辅导建议。"""
    system_prompt = (
        "你是资深的创新创业导师。请根据该学生的『项目得分情况』和『经常触发的薄弱逻辑规则』，"
        "定量+定性地分析其创新创业能力画像。\n"
        "要求：指出其能力优势与认知死角，最后单独分出一节为授课教师提供针对该学生的『辅导话术』与『一对一干预重点』。\n"
        "使用Markdown格式输出，排版清晰美观。"
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
        Node(name="rubric_scorer", function=rubric_scorer),
    ]
    for node in nodes:
        graph.add_node(node)
    graph.connect("extract_entities", "hypergraph_critic")
    graph.connect("hypergraph_critic", "strategy_selector")
    graph.connect("strategy_selector", "generate_rebuttal")
    graph.connect("generate_rebuttal", "rubric_scorer")
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


def run_langgraph_cycle(
    student_input: str,
    conversation_history: List[Dict[str, str]] = None,
    accumulated_info: Dict[str, Any] = None,
) -> AgentState:
    state = AgentState(
        student_input=student_input,
        conversation_history=conversation_history or [],
        accumulated_info=accumulated_info or {},
    )
    graph = build_state_graph()
    final = graph.execute(state)
    return final


def run_demo_cycle():
    final = run_langgraph_cycle("我们的客户是新型服务业，希望自动化内部营销。")
    render_frontend_snapshot(final)


if __name__ == "__main__":
    run_demo_cycle()

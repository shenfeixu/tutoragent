import streamlit as st
from datetime import datetime
from pathlib import Path
import uuid
import pypdf
import zipfile
import xml.etree.ElementTree as ET
import plotly.graph_objects as go

from src.agents.langgraph_core import (
    run_langgraph_cycle,
    run_learning_mode_cycle,
    COMPETITION_WEIGHTS, 
    RUBRIC_DIM_NAMES, 
    generate_financial_report, 
    generate_business_plan, 
    generate_intervention_plan,
    FALLACY_STRATEGY_LIBRARY,
    FALLACY_SEVERITY,
    generate_student_profile
)
from src.utils.exporters import export_markdown_to_docx
from src.utils.database import (
    authenticate_user,
    create_user,
    get_user_by_id,
    update_last_login,
    save_user_session,
    load_user_session,
    list_user_sessions,
    delete_user_session,
    get_user_memory,
    get_student_scores,
)

st.set_page_config(
    page_title="超图教练 - 创新创业教学智能体",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* =========================================================
       TUTORAGENT PREMIUM UI (v1.24)
       Elegant Dark Theme | Glassmorphism | Custom Chat Bubbles
       ========================================================= */
       
    /* 1. Global App Background (Premium Dark Gradient) */
    .stApp {
        background: radial-gradient(circle at top left, #0D1629 0%, #030712 100%);
        color: #F8FAFC !important;
    }

    /* 2. Glassmorphism Forms (For Login & Upload forms) */
    [data-testid="stForm"] {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="stForm"]:hover {
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 10px 50px rgba(99, 102, 241, 0.15);
    }
    
    /* 3. Sleek Inputs & Buttons */
    /* Form Label Colors (Crucial for Contrast) */
    [data-testid="stTextInput"] label p, [data-testid="stSelectbox"] label p {
        color: #F8FAFC !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding-bottom: 4px;
    }
    .stTextInput>div>div>input {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }
    .stTextInput>div>div>input:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Selectbox BaseWeb Override */
    div[data-baseweb="select"] > div {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }
    div[data-baseweb="select"] span {
        color: #ffffff !important;
    }
    div[data-baseweb="select"] li {
        background-color: #1E293B !important;
        color: white !important;
    }
    
    /* Tabs BaseWeb Override */
    button[data-baseweb="tab"] p {
        color: #94A3B8 !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #60A5FA !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom-color: #60A5FA !important;
    }
    
    /* Fix Chat Input Contrast & Disgusting Red Border */
    [data-testid="stChatInput"] {
        background-color: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 5px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    }
    /* Inner container of ChatInput has awful inline styles */
    [data-testid="stChatInput"] > div, [data-testid="stChatInput"] > div > div {
        background-color: transparent !important;
        border: none !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #F8FAFC !important;
        font-size: 1.05rem !important;
    }
    
    button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #2563EB 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4) !important;
    }
    
    /* Secondary buttons (Sidebar actions) */
    button[kind="secondary"] {
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        background-color: rgba(30, 41, 59, 0.5) !important;
        color: #F8FAFC !important;
        transition: all 0.2s ease !important;
    }
    button[kind="secondary"]:hover {
        background-color: rgba(51, 65, 85, 0.8) !important;
        border-color: #38BDF8 !important;
        color: #38BDF8 !important;
    }

    /* 4. Chat Bubbles Revamp (Right/Left separated, smooth visuals) */
    /* Hide the default chat background */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
    }
    
    /* User Message Container: Right aligned */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse;
        text-align: right; margin-bottom: 24px;
        animation: slideInRight 0.3s ease forwards;
    }
    /* User Message Bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 4px 18px 18px;
        display: inline-block;
        text-align: left;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.2);
    }
    /* Hide the huge default User avatar completely so it looks like WeChat */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stChatMessageAvatar {
        margin-left: 12px; margin-right: 0px;
    }

    /* Assistant Message Container: Left aligned */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        margin-bottom: 24px;
        animation: slideInLeft 0.3s ease forwards;
    }
    /* Assistant Message Bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
        background-color: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 12px 18px;
        border-radius: 4px 18px 18px 18px;
        display: inline-block;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* Animations */
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }

    /* 5. Sidebar Polish & Role Identity */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Make the Teacher identifier pop out! */
    .teacher-badge {
        background: linear-gradient(90deg, #F59E0B 0%, #EF4444 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 1.2rem;
    }
    .student-badge {
        color: #9CA3AF;
        font-weight: 600;
    }

    /* Custom Flex Dashboard Cards */
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .dash-card {
        background: linear-gradient(145deg, rgba(30,41,59,0.7), rgba(15,23,42,0.8));
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .dash-card-title {
        color: #94A3B8;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .dash-card-value {
        color: #F8FAFC;
        font-size: 1.5rem;
        font-weight: 800;
    }

    /* 6. Dashboard Metrics / Score Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: white !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #94A3B8 !important;
    }
    /* Wrap metrics in a glass container */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.4);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: transform 0.2s;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        background: rgba(30, 41, 59, 0.7);
    }
    
    /* 7. Expanders Polish */
    .streamlit-expanderHeader {
        background-color: rgba(30, 41, 59, 0.6) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Remove original ugly CSS session items and replace with native buttons */
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if "user" not in st.session_state:
        st.session_state.user = None
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_title" not in st.session_state:
        st.session_state.session_title = "新对话"
    if "accumulated_info" not in st.session_state:
        st.session_state.accumulated_info = {}
    if "view" not in st.session_state:
        st.session_state.view = "student"
    if "target_competition" not in st.session_state:
        st.session_state.target_competition = "互联网+"
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {}
    if "student_mode" not in st.session_state:
        st.session_state.student_mode = "竞赛教练模式"


def generate_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

def extract_text_from_upload(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    try:
        if filename.endswith('.pdf'):
            reader = pypdf.PdfReader(uploaded_file)
            text_parts = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    # 洗涤非法代理字符 (Surrogates)，防止数据库保存时触发 UnicodeEncodeError
                    cleaned = "".join(c for c in extracted if not ('\ud800' <= c <= '\udfff'))
                    text_parts.append(cleaned)
            return "\n".join(text_parts)
        elif filename.endswith('.docx'):
            document = zipfile.ZipFile(uploaded_file)
            xml_content = document.read('word/document.xml')
            document.close()
            tree = ET.fromstring(xml_content)
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            text = []
            for paragraph in tree.findall('.//w:p', ns):
                texts = [node.text for node in paragraph.findall('.//w:t', ns) if node.text]
                if texts:
                    text.append("".join(texts))
            return "\n".join(text)
        elif filename.endswith('.txt'):
            return uploaded_file.getvalue().decode("utf-8")
        else:
            return "不支持的文件格式。"
    except Exception as e:
        return f"文件解析错误: {e}"


def create_new_session():
    st.session_state.current_session_id = generate_session_id()
    st.session_state.messages = []
    st.session_state.session_title = "新对话"
    
    memory = None
    if getattr(st.session_state, "user", None):
        memory = get_user_memory(st.session_state.user["id"])
        
    st.session_state.accumulated_info = {}
    if memory:
        st.session_state.accumulated_info["student_memory"] = memory
    st.session_state.accumulated_info["student_mode"] = st.session_state.get("student_mode", "竞赛教练模式")


def load_session_to_state(session_id: str):
    user_id = st.session_state.user["id"]
    data = load_user_session(user_id, session_id)
    if data:
        st.session_state.current_session_id = session_id
        st.session_state.messages = data.get("messages", [])
        st.session_state.session_title = data.get("title", "新对话")
        st.session_state.accumulated_info = data.get("accumulated_info", {})
        st.session_state.student_mode = st.session_state.accumulated_info.get("student_mode", "竞赛教练模式")


def save_current_session():
    if st.session_state.user and st.session_state.current_session_id and st.session_state.messages:
        user_id = st.session_state.user["id"]
        st.session_state.accumulated_info["student_mode"] = st.session_state.get("student_mode", "竞赛教练模式")
        save_user_session(
            user_id=user_id,
            session_id=st.session_state.current_session_id,
            title=st.session_state.session_title,
            messages=st.session_state.messages,
            accumulated_info=st.session_state.accumulated_info,
        )


def delete_current_session():
    if st.session_state.user and st.session_state.current_session_id:
        user_id = st.session_state.user["id"]
        delete_user_session(user_id, st.session_state.current_session_id)
    create_new_session()


def render_login_page():
    st.markdown("<div style='margin-bottom: 8vh;'></div>", unsafe_allow_html=True)
    
    col_hero, col_margin, col_form = st.columns([1.5, 0.2, 1])
    
    with col_hero:
        st.markdown("""
        <div style="padding-top: 2rem;">
            <h1 style='font-size: 4rem; font-weight: 900; background: -webkit-linear-gradient(45deg, #60A5FA, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 20px;'>
                🎯 超图教练<br>多智能体双创辅导系统
            </h1>
            <p style='font-size: 1.3rem; color: #94A3B8; line-height: 1.6; margin-bottom: 30px;'>
                融合底层垂直业务图谱引擎，通过大模型驱动的多Agent协作管线，为师生提供对接主流赛制的量化推演与评估闭环。
            </p>
            <div style="border-left: 3px solid #6366F1; padding-left: 20px; color: #E2E8F0; font-size: 1.1rem; line-height: 1.8;">
                ✨ <b>图谱节点寻址</b> &nbsp;结构化定位项目逻辑与商业闭环<br>
                ✨ <b>多端智能体协同</b> &nbsp;技术/市场/财务专家机制交叉质询<br>
                ✨ <b>AI 动态双视角评估</b> &nbsp;学生精准诊断与教师数据看板
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_form:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.3); padding: 5px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 系统登录", "✨ 新建账户"])
        
        with tab1:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("用户名", key="login_username")
                password = st.text_input("密码", type="password", key="login_password")
                login_role = st.selectbox("身份", ["student", "teacher", "admin"], format_func=lambda x: {"student": "学生", "teacher": "教师", "admin": "管理员"}.get(x, x), key="login_role_select")
                
                st.markdown("<br>", unsafe_allow_html=True)
                submit = st.form_submit_button("登录验证", use_container_width=True)
                
                if submit:
                    if not username or not password:
                        st.error("请输入用户名和密码")
                    else:
                        user = authenticate_user(username, password)
                        if user:
                            if user["role"] == login_role:
                                st.session_state.user = user
                                update_last_login(user["id"])
                                if user["role"] == "teacher":
                                    st.session_state.view = "teacher"
                                elif user["role"] == "admin":
                                    st.session_state.view = "admin"
                                else:
                                    st.session_state.view = "student"
                                st.rerun()
                            else:
                                role_names = {"student": "学生", "teacher": "教师", "admin": "管理员"}
                                st.error(f"身份验证失败：该账号实际注册身份为【{role_names.get(user['role'], '未知')}】")
                        else:
                            st.error("用户名或密码错误")
        
        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("用户名", key="reg_username")
                new_password = st.text_input("密码", type="password", key="reg_password")
                confirm_password = st.text_input("确认密码", type="password", key="reg_confirm")
                display_name = st.text_input("显示名称 (可选)", key="reg_displayname")
                role = st.selectbox("身份", ["student", "teacher", "admin"], format_func=lambda x: {"student": "学生", "teacher": "教师", "admin": "管理员"}.get(x, x))
                
                st.markdown("<br>", unsafe_allow_html=True)
                register = st.form_submit_button("注册账户", use_container_width=True)
                
                if register:
                    if not new_username or not new_password:
                        st.error("请填写完整信息")
                    elif new_password != confirm_password:
                        st.error("两次密码不一致")
                    elif len(new_password) < 6:
                        st.error("密码为了您的安全需要大于6位")
                    else:
                        user_id = create_user(
                            username=new_username,
                            password=new_password,
                            role=role,
                            display_name=display_name or new_username,
                        )
                        if user_id:
                            st.success("✅ 账户创建成功，请切换至登录页！")
                        else:
                            st.error("用户名已存在")
                            
        st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        user = st.session_state.user
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if user["role"] == "teacher":
                st.markdown(f"<div class='teacher-badge'>👑 终身教职 | {user.get('display_name', user['username'])}</div>", unsafe_allow_html=True)
            elif user["role"] == "admin":
                st.markdown(f"<div class='teacher-badge'>🛡️ 超级管理 | {user.get('display_name', user['username'])}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='student-badge'>👨‍🎓 学生创客 | {user.get('display_name', user['username'])}</div>", unsafe_allow_html=True)
        with col2:
            if st.button("🚪", help="退出登录"):
                st.session_state.user = None
                st.session_state.current_session_id = None
                st.session_state.messages = []
                st.rerun()
        
        st.caption(f"身份: {'教师' if user['role'] == 'teacher' else ('管理员' if user['role'] == 'admin' else '学生')}")
        
        st.divider()

        if user["role"] == "student":
            st.markdown("**🧭 学习模式**")
            st.session_state.student_mode = st.radio(
                "选择当前对话方式：",
                ["竞赛教练模式", "自由对话学习模式"],
                index=0 if st.session_state.get("student_mode", "竞赛教练模式") == "竞赛教练模式" else 1,
            )
            st.caption("自由对话模式会边回答边追问，并结合图谱案例帮助理解。")
            st.divider()
        
        st.markdown("**🏆 赛事目标设置**")
        st.session_state.target_competition = st.selectbox(
            "选择打分基准：",
            ["互联网+", "挑战杯", "创青春", "数模"],
            index=["互联网+", "挑战杯", "创青春", "数模"].index(st.session_state.target_competition)
        )
        st.session_state.project_type = st.selectbox(
            "选择项目类型：",
            ["商业型", "公益型"],
            index=0 if st.session_state.get("project_type", "商业型") == "商业型" else 1
        )
        
        st.divider()
        
        st.divider()
        
        if user["role"] == "student":
            st.markdown("**🤖 AI 专属测评**")
            if st.button("✨ 生成并查看我的能力画像报告", use_container_width=True):
                with st.spinner("AI 正在提纯商战能力..."):
                    weights = COMPETITION_WEIGHTS.get(st.session_state.target_competition, COMPETITION_WEIGHTS["互联网+"])
                    all_scores = get_student_scores(weights)
                    my_score = next((s for s in all_scores if s["user_id"] == user["id"]), None)
                    
                    if my_score and len(my_score["sessions"]) > 0:
                        from src.agents.langgraph_core import generate_student_profile
                        student_data = {
                            "name": user.get("display_name", user["username"]),
                            "total_score": round(my_score["total_score"], 1),
                            "risk_level": my_score["risk_level"],
                            "rubric_scores": {
                                "pain_point": my_score["pain_point_score"],
                                "planning": my_score["planning_score"],
                                "modeling": my_score["modeling_score"],
                                "leverage": my_score["leverage_score"],
                                "presentation": my_score["presentation_score"],
                            },
                            "frequent_fallacies": my_score["fallacies"],
                            "session_count": len(my_score["sessions"])
                        }
                        profile_md = generate_student_profile(student_data, for_student=True)
                        st.session_state["my_ai_profile_content"] = profile_md
                        st.session_state["show_student_profile"] = True
                        st.rerun()
                    else:
                        st.error("您暂无历史会话或记录过少，请先与教练切磋后再生成能力画像！")
            
            st.markdown("**💰 AI 财务诊断**")
            if st.button("📊 生成项目财务分析报告", use_container_width=True):
                with st.spinner("AI 正在进行财务深度诊断..."):
                    # 从当前对话中收集财务数据
                    finance_data = {
                        "accumulated_info": st.session_state.get("accumulated_info", {}),
                        "extracted_nodes": {},
                        "frequent_fallacies": [],
                        "session_count": 0,
                    }
                    # 从最新一轮对话的 state 中提取实体
                    for msg in reversed(st.session_state.get("messages", [])):
                        if msg.get("role") == "assistant" and msg.get("state"):
                            finance_data["extracted_nodes"] = msg["state"].get("extracted_nodes", {})
                            finance_data["frequent_fallacies"] = msg["state"].get("detected_fallacies", [])
                            break
                    
                    # 补充历史会话数据
                    weights = COMPETITION_WEIGHTS.get(st.session_state.target_competition, COMPETITION_WEIGHTS["互联网+"])
                    all_scores = get_student_scores(weights)
                    my_score = next((s for s in all_scores if s["user_id"] == user["id"]), None)
                    if my_score:
                        finance_data["session_count"] = len(my_score["sessions"])
                        finance_data["frequent_fallacies"] = my_score.get("fallacies", [])
                    
                    if finance_data["extracted_nodes"] or finance_data["accumulated_info"]:
                        report_md = generate_financial_report(finance_data, for_student=True)
                        st.session_state["finance_report_content"] = report_md
                        st.session_state["show_finance_report"] = True
                        st.rerun()
                    else:
                        st.error("暂无足够的项目数据，请先与教练进行至少一轮对话，描述您的商业模式后再生成财务报告！")
            
            st.markdown("**✨ 深度智慧合成**")
            # 条件：对话轮数 >= 3 (即消息数 >= 6)
            msg_count = len(st.session_state.get("messages", []))
            if msg_count >= 6:
                if st.button("🌟 合成完整项目商业计划书", use_container_width=True):
                    with st.status("🚀 深度智慧合成中 (全程可能需要60-180秒)...", expanded=True) as status:
                        st.write("🟢 [项目教练] 正在提取对话上下文特征...")
                        import time
                        time.sleep(1.0)
                        st.write("🟢 [知识图谱] 正在构建专家风险匹配子图与纠缠度网络...")
                        time.sleep(1.5)
                        st.write("🟢 [竞赛顾问] 介入，根据项目类型自适应打磨专属商业叙事...")
                        time.sleep(0.5)
                        st.write("⏳ 正在等待大模型推理合成 12 章节标准计划书，请耐心等待...")
                        
                        bp_data = {
                            "accumulated_info": st.session_state.get("accumulated_info", {}),
                            "conversation_history": st.session_state.get("messages", []),
                            "extracted_nodes": {},
                            "frequent_fallacies": [],
                        }
                        bp_data["accumulated_info"]["project_type"] = st.session_state.get("project_type", "商业型")
                        # 提取最新状态
                        for msg in reversed(st.session_state.messages):
                            if msg.get("role") == "assistant" and msg.get("state"):
                                bp_data["extracted_nodes"] = msg["state"].get("extracted_nodes", {})
                                bp_data["frequent_fallacies"] = msg["state"].get("detected_fallacies", [])
                                break
                        
                        full_bp_md = generate_business_plan(bp_data, target_comp=st.session_state.target_competition)
                        st.session_state["full_bp_content"] = full_bp_md
                        st.session_state["show_full_bp"] = True
                        
                        # 自动将生成的BP存档，供教师端审阅
                        import json
                        from pathlib import Path
                        bp_file = Path("data/student_bps.json")
                        bps_data = {}
                        if bp_file.exists():
                            try:
                                bps_data = json.loads(bp_file.read_text(encoding="utf-8"))
                            except:
                                pass
                        bps_data[user.get("display_name", user["username"])] = full_bp_md
                        bp_file.write_text(json.dumps(bps_data, ensure_ascii=False), encoding="utf-8")
                        
                        st.rerun()
            else:
                rounds_left = (6 - msg_count + 1) // 2
                st.info(f"💡 再进行 {rounds_left} 轮深度对话，即可解锁‘商业计划书自动合成’功能！", icon="🔒")
        
        # [NEW] 教师视图/班级画像入口 (Req 7, 9)
        if user["role"] in ["teacher", "admin"]:
            st.divider()
            st.markdown("**👩‍🏫 教师/助教控制塔**")
            if st.button("📊 进入班级能力画像看板", use_container_width=True, type="secondary"):
                st.session_state.view = "teacher_dashboard"
                st.rerun()
        
        st.divider()

        if st.button("➕ 新建对话", use_container_width=True):
            save_current_session()
            create_new_session()
            st.rerun()
        
        st.divider()
        
        sessions = list_user_sessions(user["id"])
        
        if sessions:
            st.markdown("**历史对话**")
            for session in sessions:
                is_active = session["session_id"] == st.session_state.current_session_id
                
                with st.container():
                    col_title, col_delete = st.columns([4, 1])
                    with col_title:
                        if st.button(
                            f"{'▶ ' if is_active else ''}{session['title']}",
                            key=f"session_{session['session_id']}",
                            use_container_width=True,
                        ):
                            save_current_session()
                            load_session_to_state(session["session_id"])
                            st.rerun()
                    with col_delete:
                        if st.button("🗑️", key=f"del_{session['session_id']}", help="删除"):
                            delete_user_session(user["id"], session["session_id"])
                            if is_active:
                                create_new_session()
                            st.rerun()
                    
                    st.caption(f"{session['message_count']} 条消息 · {session['updated_at'][:10] if session.get('updated_at') else ''}")
        else:
            st.info("暂无历史对话，开始新对话吧！")
        
        st.divider()
        
        memory = None
        if "accumulated_info" in st.session_state and st.session_state.accumulated_info:
            memory = st.session_state.accumulated_info.get("student_memory")
        if not memory and getattr(st.session_state, "user", None):
            memory = get_user_memory(st.session_state.user["id"])
            
        if memory:
            st.markdown("🧠 **教练长期记忆档案**")
            st.info(memory)
            st.caption("跨会话跟踪中...")
            st.divider()
        
        with st.expander("⚙️ 调试面板"):
            if st.session_state.messages:
                last_msg = st.session_state.messages[-1]
                if last_msg.get("state"):
                    state = last_msg["state"]
                    st.metric("触发逻辑数量", len(state.get("detected_fallacies", [])))
                    
                    with st.expander("抽取实体"):
                        st.json(state.get("extracted_nodes", {}))
            else:
                st.write("等待对话开始...")


def render_kg_query_visualization(kg_query_details: list):
    """渲染知识图谱查询过程可视化组件"""
    if not kg_query_details:
        return

    def build_detail_subgraph(detail: dict):
        nodes = []
        edges = []
        seen_nodes = set()
        seen_edges = set()

        def add_node(node_id, label, node_type):
            if not node_id or node_id in seen_nodes:
                return
            seen_nodes.add(node_id)
            nodes.append({"id": node_id, "label": label, "type": node_type})

        def add_edge(source, target, label):
            key = (source, target, label)
            if not source or not target or key in seen_edges:
                return
            seen_edges.add(key)
            edges.append({"source": source, "target": target, "label": label})

        project_details = detail.get("project_details", []) or []
        if project_details:
            for idx, proj in enumerate(project_details[:5], start=1):
                project_name = proj.get("project_name", f"项目{idx}")
                project_id = f"project::{idx}::{project_name}"
                add_node(project_id, project_name, "project")

                tech_name = proj.get("tech_name")
                if tech_name:
                    tech_id = f"tech::{idx}::{tech_name}"
                    add_node(tech_id, tech_name, "tech")
                    add_edge(project_id, tech_id, "USE")

                market_name = proj.get("market_name")
                if market_name:
                    market_id = f"market::{idx}::{market_name}"
                    add_node(market_id, market_name, "market")
                    add_edge(project_id, market_id, "TARGET")

                for risk_name in (proj.get("risks", []) or [])[:2]:
                    risk_id = f"risk::{idx}::{risk_name}"
                    add_node(risk_id, risk_name, "risk")
                    add_edge(project_id, risk_id, "TRIGGER_RISK")

                if proj.get("value_loop_name") or proj.get("value_loop_desc"):
                    value_loop_name = proj.get("value_loop_name") or "价值闭环"
                    value_loop_id = f"value_loop::{idx}::{value_loop_name}"
                    add_node(value_loop_id, value_loop_name, "value_loop")
                    add_edge(project_id, value_loop_id, "HAS_VALUE_LOOP")
                    if tech_name:
                        add_edge(value_loop_id, f"tech::{idx}::{tech_name}", "INVOLVES_TECH")
                    if market_name:
                        add_edge(value_loop_id, f"market::{idx}::{market_name}", "INVOLVES_MARKET")

        for idx, risk in enumerate((detail.get("risk_details", []) or [])[:5], start=1):
            risk_name = risk.get("risk_name")
            if not risk_name:
                continue
            risk_id = f"risk_detail::{idx}::{risk_name}"
            add_node(risk_id, risk_name, "risk")

            risk_pattern = risk.get("risk_pattern")
            if risk_pattern:
                pattern_id = f"risk_pattern::{idx}::{risk_pattern}"
                add_node(pattern_id, risk_pattern, "risk_pattern")
                add_edge(pattern_id, risk_id, "INVOLVES_RISK")

            for proj_name in (risk.get("related_projects", []) or [])[:3]:
                project_id = f"risk_project::{idx}::{proj_name}"
                add_node(project_id, proj_name, "project")
                add_edge(project_id, risk_id, "TRIGGER_RISK")
                if risk_pattern:
                    add_edge(project_id, f"risk_pattern::{idx}::{risk_pattern}", "HAS_RISK_PATTERN")

        if not nodes:
            for idx, proj_name in enumerate((detail.get("matched_projects", []) or [])[:5], start=1):
                add_node(f"matched_project::{idx}::{proj_name}", proj_name, "project")

        return nodes, edges

    def render_subgraph(graph_nodes, graph_edges, title="**🕸️ 相关案例子图**"):
        if not graph_nodes:
            return

        st.markdown(title)
        color_map = {
            "project": "#38bdf8",
            "tech": "#22c55e",
            "market": "#f59e0b",
            "risk": "#ef4444",
            "value_loop": "#a855f7",
            "risk_pattern": "#f97316",
        }
        type_order = {
            "project": 0,
            "tech": 1,
            "market": 2,
            "value_loop": 3,
            "risk_pattern": 4,
            "risk": 5,
        }

        grouped = {}
        for node in graph_nodes:
            grouped.setdefault(node.get("type", "other"), []).append(node)

        positions = {}
        for node_type, nodes_in_type in grouped.items():
            x = type_order.get(node_type, len(type_order))
            count = len(nodes_in_type)
            for j, node in enumerate(nodes_in_type):
                y = 0 if count == 1 else (count - 1) / 2 - j
                positions[node["id"]] = (x, y)

        edge_x, edge_y = [], []
        for edge in graph_edges:
            src = positions.get(edge.get("source"))
            dst = positions.get(edge.get("target"))
            if not src or not dst:
                continue
            edge_x.extend([src[0], dst[0], None])
            edge_y.extend([src[1], dst[1], None])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=1.5, color="#64748b"),
            hoverinfo="skip", showlegend=False,
        ))

        for node_type, nodes_in_type in grouped.items():
            xs, ys, labels = [], [], []
            for node in nodes_in_type:
                x, y = positions[node["id"]]
                xs.append(x)
                ys.append(y)
                labels.append(node.get("label", node["id"]))
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers+text", text=labels, textposition="top center",
                marker=dict(size=28, color=color_map.get(node_type, "#94a3b8")),
                name=node_type, hovertemplate="%{text}<extra></extra>",
            ))

        for edge in graph_edges:
            src = positions.get(edge.get("source"))
            dst = positions.get(edge.get("target"))
            if not src or not dst:
                continue
            fig.add_annotation(
                x=(src[0] + dst[0]) / 2,
                y=(src[1] + dst[1]) / 2,
                text=edge.get("label", ""),
                showarrow=False,
                font=dict(size=10, color="#cbd5e1"),
                bgcolor="rgba(15,23,42,0.75)",
            )

        fig.update_layout(
            height=460,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🔍 知识图谱查询轨迹")
    
    for idx, detail in enumerate(kg_query_details):
        with st.container():
            step = detail.get("step", "未知查询")
            query_type = detail.get("query_type", "")
            success = detail.get("success", False)
            is_learning_mode = query_type == "learning_mode_case_search"
            
            status_icon = "✅" if success else "❌"
            status_color = "#10a37f" if success else "#ef4444"
            
            st.markdown(f"""
            <div style="background-color: #2b313e; padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.5rem; border-left: 4px solid {status_color};">
                <h4 style="margin: 0; color: white;">{status_icon} {step}</h4>
                <p style="color: #9ca3af; margin: 0.5rem 0 0 0;">{detail.get('message', '')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                tech_keywords = detail.get("tech_keywords", [])
                if tech_keywords:
                    st.markdown("**🔧 提取的技术关键词:**")
                    kw_html = " ".join([f'<span style="background-color: #3b82f6; color: white; padding: 0.2rem 0.5rem; border-radius: 0.25rem; margin: 0.1rem; display: inline-block;">{kw}</span>' for kw in tech_keywords])
                    st.markdown(f'<div style="margin-bottom: 0.5rem;">{kw_html}</div>', unsafe_allow_html=True)
            
            with col2:
                market_keywords = detail.get("market_keywords", [])
                if market_keywords:
                    st.markdown("**🎯 提取的市场关键词:**")
                    kw_html = " ".join([f'<span style="background-color: #8b5cf6; color: white; padding: 0.2rem 0.5rem; border-radius: 0.25rem; margin: 0.1rem; display: inline-block;">{kw}</span>' for kw in market_keywords])
                    st.markdown(f'<div style="margin-bottom: 0.5rem;">{kw_html}</div>', unsafe_allow_html=True)

            graph_nodes = detail.get("graph_nodes", [])
            graph_edges = detail.get("graph_edges", [])
            if not graph_nodes:
                graph_nodes, graph_edges = build_detail_subgraph(detail)
            if graph_nodes:
                render_subgraph(
                    graph_nodes,
                    graph_edges,
                    title="**🕸️ 相关案例子图**" if is_learning_mode else "**🕸️ 图谱命中案例子图**",
                )

            query_attempts = detail.get("query_attempts", [])
            if query_attempts and not is_learning_mode:
                st.markdown("**📊 查询阶段记录:**")
                for attempt in query_attempts:
                    stage = attempt.get("stage", "查询")
                    found = attempt.get("found", 0)
                    error = attempt.get("error", "")
                    
                    if error:
                        st.markdown(f"- ❌ **{stage}**: 错误 - {error}")
                    else:
                        projects = attempt.get("projects", [])
                        st.markdown(f"- ✅ **{stage}**: 找到 **{found}** 个匹配项目")
                        if projects:
                            project_tags = " ".join([f'<span style="background-color: #374151; color: #10a37f; padding: 0.1rem 0.3rem; border-radius: 0.25rem; margin: 0.1rem; display: inline-block; font-size: 0.85rem;">📁 {p}</span>' for p in projects[:5]])
                            st.markdown(f'<div style="margin-left: 1rem; margin-bottom: 0.5rem;">{project_tags}</div>', unsafe_allow_html=True)
            
            project_details = detail.get("project_details", [])
            if project_details:
                st.markdown(f"**📋 匹配项目详情 ({len(project_details)}个):**")
                for proj in project_details[:5]:
                    title = f"📁 {proj.get('project_name', '未知项目')}"
                    if not is_learning_mode:
                        score = detail.get("match_scores", {}).get(proj.get('project_name', ''), 0)
                        title = f"{title} (分数: {score})"
                    with st.expander(title, expanded=False):
                        cols = st.columns(2)
                        with cols[0]:
                            st.markdown(f"**技术:** {proj.get('tech_name', 'N/A')}")
                            st.markdown(f"**成熟度:** {proj.get('tech_maturity', 'N/A')}")
                            st.markdown(f"**壁垒:** {proj.get('tech_barrier', 'N/A')}")
                        with cols[1]:
                            st.markdown(f"**市场:** {proj.get('market_name', 'N/A')}")
                            tam = proj.get('tam', 0)
                            sam = proj.get('sam', 0)
                            som = proj.get('som', 0)
                            if tam:
                                st.markdown(f"**TAM:** {tam:,.0f} 元")
                                st.markdown(f"**SAM:** {sam:,.0f} 元")
                                st.markdown(f"**SOM:** {som:,.0f} 元")
                        
                        risks = proj.get('risks', [])
                        if risks:
                            st.markdown(f"**相关风险:** {', '.join(risks[:3])}")
                        
                        value_loop = proj.get('value_loop_name', '')
                        if value_loop:
                            st.markdown(f"**价值闭环:** {value_loop}")
                            st.markdown(f"**描述:** {proj.get('value_loop_desc', 'N/A')}")
                            st.markdown(f"**LTV/CAC:** {proj.get('ltv', 0):,.0f} / {proj.get('cac', 0):,.0f}")
                            st.markdown(f"**收入模式:** {proj.get('revenue_model', 'N/A')}")
                
                match_scores = detail.get("match_scores", {})
                if match_scores and not is_learning_mode:
                    st.markdown("**📈 匹配分数排序:**")
                    sorted_scores = sorted(match_scores.items(), key=lambda x: x[1], reverse=True)
                    score_text = " > ".join([f"{name}({score})" for name, score in sorted_scores[:5]])
                    st.markdown(f"<div style='color: #10a37f;'>{score_text}</div>", unsafe_allow_html=True)
            
            matched_projects = detail.get("matched_projects", [])
            if matched_projects and not project_details:
                st.markdown("**📁 匹配到的项目案例:**")
                project_tags = " ".join([f'<span style="background-color: #374151; color: #10a37f; padding: 0.2rem 0.5rem; border-radius: 0.25rem; margin: 0.1rem; display: inline-block;">📁 {p}</span>' for p in matched_projects[:5]])
                st.markdown(f'<div>{project_tags}</div>', unsafe_allow_html=True)
            
            risk_details = detail.get("risk_details", [])
            if risk_details:
                st.markdown("**⚠️ 风险详情:**")
                for risk in risk_details[:3]:
                    with st.expander(f"⚡ {risk.get('risk_name', '未知风险')}", expanded=False):
                        severity = risk.get('severity', 'N/A')
                        severity_color = "#ef4444" if severity == "高" else "#f59e0b" if severity == "中" else "#10a37f"
                        st.markdown(f"**严重程度:** <span style='color: {severity_color}'>{severity}</span>", unsafe_allow_html=True)
                        
                        related_projs = risk.get('related_projects', [])
                        if related_projs:
                            st.markdown(f"**相关项目:** {', '.join(related_projs[:3])}")
                        
                        risk_pattern = risk.get('risk_pattern', '')
                        if risk_pattern:
                            st.markdown(f"**风险模式:** {risk_pattern}")
                            st.markdown(f"**模式描述:** {risk.get('pattern_description', 'N/A')}")
                        
                        mitigation = risk.get('mitigation', '')
                        if mitigation:
                            st.markdown(f"**缓解措施:** {mitigation}")
            
            risks_found = detail.get("risks_found", [])
            if risks_found and not risk_details:
                st.markdown("**⚠️ 发现的相关风险:**")
                risk_tags = " ".join([f'<span style="background-color: #7f1d1d; color: #fca5a5; padding: 0.2rem 0.5rem; border-radius: 0.25rem; margin: 0.1rem; display: inline-block;">⚡ {r}</span>' for r in risks_found[:5]])
                st.markdown(f'<div>{risk_tags}</div>', unsafe_allow_html=True)
            
            related_projects = detail.get("related_projects", [])
            if related_projects and related_projects != matched_projects:
                st.markdown("**🔗 相关项目:**")
                st.markdown(f"{', '.join(related_projects[:5])}")


def render_teacher_dashboard():
    """👩‍🏫 教师控制塔：班级画像看板与干预计划 (Req 7, 9)"""
    st.title("👩‍🏫 教师控制塔 - 班级能力画像看板")
    if st.button("🔙 返回导师工作台"):
        st.session_state.view = "student"
        st.rerun()
    
    st.divider()
    
    # 1. 聚合数据
    weights = COMPETITION_WEIGHTS.get(st.session_state.target_competition, COMPETITION_WEIGHTS["互联网+"])
    all_scores = get_student_scores(weights)
    
    if not all_scores:
        st.warning("暂无学生数据。")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 班级整体能力分布")
        # 简单计算平均分
        avg_scores = {
            "pain_point": sum(s["pain_point_score"] for s in all_scores) / len(all_scores),
            "planning": sum(s["planning_score"] for s in all_scores) / len(all_scores),
            "modeling": sum(s["modeling_score"] for s in all_scores) / len(all_scores),
            "leverage": sum(s["leverage_score"] for s in all_scores) / len(all_scores),
            "presentation": sum(s["presentation_score"] for s in all_scores) / len(all_scores),
        }
        
        # 恢复简洁统计表格
        st.markdown("**班级维度得分总览：**")
        st.table(avg_scores)
        
        st.write(f"**班级加权全均分：{sum(s['total_score'] for s in all_scores)/len(all_scores):.1f}**")
        
        st.subheader("👥 学生能力排行榜 (风险预警)")
        display_list = []
        for s in all_scores:
            display_list.append({
                "学生名": s["display_name"],
                "总得分": s["total_score"],
                "风险等级": s["risk_level"],
                "活跃度": f"{s['session_count']} 轮对话",
                "高频谬误": " | ".join(s["fallacies"][:3])
            })
        st.dataframe(display_list, use_container_width=True)
        
        st.subheader("✉️ 下发赛事指导意见")
        student_names = [s["学生名"] for s in display_list]
        if student_names:
            selected_student = st.selectbox("选择评审对象：", student_names)
            feedback_text = st.text_area("导师点评 (学生登录后将在顶部看到该反馈)：")
            if st.button("发送评审结论"):
                import json
                feedback_file = Path("data/teacher_feedback.json")
                feedbacks = {}
                if feedback_file.exists():
                    try:
                        feedbacks = json.loads(feedback_file.read_text(encoding="utf-8"))
                    except:
                        pass
                feedbacks[selected_student] = feedback_text
                feedback_file.write_text(json.dumps(feedbacks, ensure_ascii=False), encoding="utf-8")
                st.success(f"✅ 已成功将评估意见下发给 {selected_student}")

    with col2:
        st.subheader("🧠 AI 教学干预建议")
        if st.button("🚀 生成针对性干预计划", use_container_width=True):
            common_fallacies = {}
            for s in all_scores:
                for f in s["fallacies"]:
                    common_fallacies[f] = common_fallacies.get(f, 0) + 1
            
            top_fallacies = sorted(common_fallacies.items(), key=lambda x: x[1], reverse=True)[:5]
            stats = {"top_errors": top_fallacies, "avg_scores": avg_scores}
            
            with st.spinner("AI 正在分析班级共性痛点..."):
                plan = generate_intervention_plan(stats)
                st.markdown(plan)


def render_chat_message(role: str, content: str, state: dict = None):
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)
            
            if state:
                # Hide detailed analysis and rubric if it's a strict interception
                fallacies = state.get("detected_fallacies", [])
                student_mode = state.get("accumulated_info", {}).get("student_mode", "竞赛教练模式")
                is_learning_mode = student_mode == "自由对话学习模式"
                if "GENTLE_INTERCEPTION" in fallacies or "GHOSTWRITING_INTERCEPTION" in fallacies:
                    return
                
                with st.expander("📊 详细分析", expanded=False):
                    if is_learning_mode:
                        st.markdown("**抽取的项目线索**")
                        st.json(state.get("extracted_nodes", {}))
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**抽取的商业实体**")
                            st.json(state.get("extracted_nodes", {}))
                        with col2:
                            st.markdown("**触发的超图逻辑**")
                            if fallacies:
                                st.warning("已触发: " + ", ".join(fallacies))
                            else:
                                st.success("逻辑通畅")
                    
                    # 知识图谱诊断轨迹 (Req 4, 5) - 独立展示，不嵌套在else中
                    if state and state.get("kg_query_details"):
                        st.markdown("---")
                        st.markdown("**🌐 知识图谱检索结果**" if is_learning_mode else "**🌐 知识图谱 / 超图检索轨迹**")
                        for detail in state["kg_query_details"]:
                            cat_icon = "📍"
                            if "Market" in detail.get("category", ""): cat_icon = "📊"
                            elif "Tech" in detail.get("category", ""): cat_icon = "🛠️"
                            elif "Risk" in detail.get("category", ""): cat_icon = "⚠️"
                            
                            st.markdown(f"**{cat_icon} {detail['step']}**")
                            st.caption(f"🔍 检索逻辑：{detail.get('retrieval_reason', '语义关联匹配')}")
                            st.write(f"_{detail['message']}_")
                    
                    # 证据链溯源
                    if state.get("evidence"):
                        st.markdown("---")
                        st.markdown("**🧬 学习辅助线索**" if is_learning_mode else "**🧬 逻辑溯源（超图审计证据）**")
                        for ev in state["evidence"]:
                            st.markdown(f"**{ev['step']}**: {ev['detail']}")
                
                # ── 评估细则与过程 (Req 4, 5) ──
                if not is_learning_mode:
                    with st.expander("📋 评估细则与过程（点击展开查看完整规则）", expanded=False):
                        from src.agents.langgraph_core import FALLACY_STRATEGY_LIBRARY, FALLACY_SEVERITY
                        
                        st.markdown("#### 📖 超图评估规则库（H1-H20）")
                        st.caption("以下为系统内置的全部逻辑审计规则，每条规则均通过与大模型反复讨论后确定。")
                        
                        severity_groups = {
                            "🔴 致命伤 (Fatal)": [],
                            "🟠 重大问题 (Major)": [],
                            "🟡 显著影响 (Significant)": [],
                            "🟢 轻微瑕疵 (Minor)": [],
                        }
                        for rule_id, desc in FALLACY_STRATEGY_LIBRARY.items():
                            severity = FALLACY_SEVERITY.get(rule_id, 0.5)
                            if severity >= 2.5:
                                severity_groups["🔴 致命伤 (Fatal)"].append((rule_id, desc))
                            elif severity >= 1.5:
                                severity_groups["🟠 重大问题 (Major)"].append((rule_id, desc))
                            elif severity >= 1.2:
                                severity_groups["🟡 显著影响 (Significant)"].append((rule_id, desc))
                            else:
                                severity_groups["🟢 轻微瑕疵 (Minor)"].append((rule_id, desc))
                        
                        for group_name, rules in severity_groups.items():
                            if rules:
                                st.markdown(f"**{group_name}**")
                                for rule_id, desc in rules:
                                    triggered = "⚡" if state and rule_id in state.get("detected_fallacies", []) else ""
                                    st.markdown(f"- `{rule_id}` {triggered} {desc}")
                        
                        st.markdown("---")
                        st.markdown("#### 🔄 本轮评估过程")
                        fallacies = state.get("detected_fallacies", []) if state else []
                        total_rules = len(FALLACY_STRATEGY_LIBRARY)
                        st.markdown(f"- 共检查 **{total_rules}** 条规则")
                        st.markdown(f"- 本轮触发 **{len(fallacies)}** 条：{', '.join(fallacies) if fallacies else '无'}")
                        st.markdown(f"- 评估方式：LLM实体提取 → 知识图谱对标 → 超图逻辑审计 → Rubric评分")

                    rubric = state.get("rubric_scores", {})
                    if rubric:
                        with st.expander("🏆 赛事 Rubric 评分", expanded=True):
                            default_comp = st.session_state.get("target_competition", "互联网+")
                            comp_options = list(COMPETITION_WEIGHTS.keys())
                            try:
                                default_idx = comp_options.index(default_comp)
                            except ValueError:
                                default_idx = 0

                            comp_name = st.selectbox(
                                "选择赛事权重",
                                comp_options,
                                index=default_idx,
                                key=f"comp_{id(state)}_{st.session_state.target_competition}",
                            )
                            weights = COMPETITION_WEIGHTS.get(comp_name, COMPETITION_WEIGHTS["互联网+"])
                            
                            score_cols = st.columns(5)
                            weighted_total = 0.0
                            for idx, (dim, dim_name) in enumerate(RUBRIC_DIM_NAMES.items()):
                                dim_data = rubric.get(dim, {})
                                score = dim_data.get("score", 0)
                                w = weights.get(dim, 0.2)
                                weighted_total += score * w
                                with score_cols[idx]:
                                    color = "🔴" if score <= 2 else ("🟡" if score <= 3 else "🟢")
                                    st.metric(f"{color} {dim_name}", f"{score}/5")
                            
                            st.metric(f"📊 {comp_name} 加权综合分", f"{weighted_total:.2f}/5.00")
                            
                            weak_dims = [
                                (dim, rubric.get(dim, {}))
                                for dim in RUBRIC_DIM_NAMES
                                if rubric.get(dim, {}).get("score", 5) <= 2
                            ]
                            if weak_dims:
                                st.divider()
                                st.markdown("**⚠️ 薄弱项行动建议**")
                                for dim, dim_data in weak_dims:
                                    st.error(
                                        f"**{RUBRIC_DIM_NAMES[dim]}** (得分 {dim_data.get('score', 0)}/5)\n\n"
                                        f"❌ 缺失证据: {dim_data.get('missing_evidence', 'N/A')}\n\n"
                                        f"✅ 最小修复: {dim_data.get('minimal_fix', 'N/A')}"
                                    )
                
                kg_query_details = state.get("kg_query_details", [])
                if kg_query_details:
                    with st.expander("🔍 知识图谱查询轨迹", expanded=True):
                        render_kg_query_visualization(kg_query_details)

def main():
    init_session_state()
    
    if not st.session_state.user:
        render_login_page()
        return
    
    if st.session_state.view == "teacher":
        if st.session_state.user.get("role") != "teacher":
            st.error("🛑 403 Forbidden: 此页面仅限教师访问。")
            st.stop()
        from Instructor_View import main as teacher_main
        teacher_main()
        return
        
    if st.session_state.view == "admin":
        if st.session_state.user.get("role") != "admin":
            st.error("🛑 403 Forbidden: 您没有管理员权限访问此页面。")
            st.stop()
        from Admin_View import main as admin_main
        admin_main()
        return
    
    if st.session_state.view == "teacher_dashboard":
        if st.session_state.user.get("role") not in ["teacher", "admin"]:
            st.session_state.view = "student"
            st.rerun()
        render_teacher_dashboard()
        return

    render_sidebar()
    
    if st.session_state.get("show_student_profile", False) and st.session_state.view == "student":
        st.title("🎓 学生专属动态能力画像")
        if st.button("🔙 返回当前对话列表", type="primary"):
            st.session_state["show_student_profile"] = False
            st.rerun()
            
        # 已移除雷达图，直接展示剖析报告
        
        st.markdown(st.session_state.get("my_ai_profile_content", "生成失败"))
        return
    
    if st.session_state.get("show_finance_report", False) and st.session_state.view == "student":
        st.title("💰 项目财务健康诊断报告")
        st.caption("基于 AI 多维度财务分析引擎生成")
        if st.button("🔙 返回当前对话列表", type="primary"):
            st.session_state["show_finance_report"] = False
            st.rerun()
        
        st.divider()
        st.markdown(st.session_state.get("finance_report_content", "生成失败"))
        return

    if st.session_state.get("show_full_bp", False) and st.session_state.view == "student":
        st.title(f"✨ {st.session_state.target_competition} 商业计划书 (AI 合成版)")
        st.caption("基于全量对话逻辑深度合成，仅供参赛参考")
        
        col_b1, col_b2 = st.columns([1, 4])
        with col_b1:
            if st.button("🔙 返回对话列表", type="primary", use_container_width=True):
                st.session_state["show_full_bp"] = False
                st.rerun()
        with col_b2:
            st.info("💡 提示：您可以直接全选、复制以下内容到您的正式 Word/PPT 文档中。")
            
        st.divider()
        st.markdown(st.session_state.get("full_bp_content", "生成失败"))
        
        # [NEW] Word 下载功能 (Req 10)
        st.divider()
        docx_bytes = export_markdown_to_docx(
            st.session_state["full_bp_content"], 
            title=f"{st.session_state.target_competition} 商业计划书",
            subtitle=f"项目名称：{st.session_state.accumulated_info.get('project_name', '未命名')} | 生成时间：{datetime.now().strftime('%Y-%m-%d')}"
        )
        st.download_button(
            label="📄 下载完整版商业计划书 (.docx)",
            data=docx_bytes,
            file_name=f"商业计划书_{st.session_state.accumulated_info.get('project_name', '未命名')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
        return
    
    col_t1, col_t2 = st.columns([3, 1.2])
    with col_t1:
        st.title("🎯 创新创业教学智能体")
        st.markdown("基于知识图谱与超图推理的创业项目诊断助手。")
        
        # 加载教师评审反馈
        import json
        feedback_file = Path("data/teacher_feedback.json")
        if feedback_file.exists():
            try:
                feedbacks = json.loads(feedback_file.read_text(encoding="utf-8"))
                display_name = st.session_state.user.get("display_name", st.session_state.user["username"])
                if display_name in feedbacks:
                    st.info(f"**👨‍🏫 你的商业计划书已收到主教练最新评审意见：**\n\n{feedbacks[display_name]}")
            except:
                pass
    with col_t2:
        st.write("")
        st.write("")
        with st.popover("📎 上传完整计划书分析", use_container_width=True):
            st.markdown("💡 **AI 深度解析引擎**\n\n支持拖拽 **PDF / Word / TXT**，AI 会全景扫描提取所有业务节点！")
            with st.form("file_upload_form"):
                uploaded_file = st.file_uploader("请在这里拖拽文档", type=["pdf", "docx", "txt"], label_visibility="collapsed")
                upload_submitted = st.form_submit_button("🔥 确认提交并开始分析")
    
    if st.session_state.accumulated_info:
        with st.expander("📋 项目核心情报大盘 (实时映射)", expanded=False):
            info = st.session_state.accumulated_info
            st.markdown(f"""
            <div class="dashboard-grid">
                <div class="dash-card">
                    <div class="dash-card-title">🚀 项目名称</div>
                    <div class="dash-card-value">{info.get("project_name", "未命名")}</div>
                </div>
                <div class="dash-card">
                    <div class="dash-card-title">⚙️ 技术成熟度</div>
                    <div class="dash-card-value">{info.get("tech_maturity", "未判定")}</div>
                </div>
                <div class="dash-card">
                    <div class="dash-card-title">🎯 目标市场</div>
                    <div class="dash-card-value">{info.get("target_market", "未判定")}</div>
                </div>
                <div class="dash-card">
                    <div class="dash-card-title">💰 融资与营收</div>
                    <div class="dash-card-value">{info.get("funding_stage", "早期")} | {f"{info.get('revenue', 0):,.0f}元" if info.get('revenue') else "前置"}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
                
            # student_memory moved to sidebar
    
    st.divider()
    
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.messages:
            render_chat_message(
                msg.get("role"),
                msg.get("content"),
                msg.get("state") if msg.get("role") == "assistant" else None
            )
            
    # Handle File Upload Submission logic
    current_mode = st.session_state.get("student_mode", "竞赛教练模式")
    chat_placeholder = (
        "请描述你的创业想法，或与我交流..."
        if current_mode == "竞赛教练模式"
        else "可以直接提问：比如我不知道怎么找用户痛点、怎么选赛道、怎么做商业模式..."
    )
    prompt = st.chat_input(chat_placeholder)
    
    # Process either a file upload OR text chat input
    input_text = None
    if upload_submitted and uploaded_file:
        with st.spinner("正在解析文档提取文本..."):
            doc_text = extract_text_from_upload(uploaded_file)
            
        if "文件解析错误" in doc_text or "不支持的文件" in doc_text:
            st.error(doc_text)
        elif len(doc_text.strip()) < 10:
            st.warning(f"⚠️ 警告：从文档中提取到的有效文字极少（仅{len(doc_text)}字）。请确认该 PDF 是否为扫描图片格式，或者是加密文件。")
        else:
            st.success(f"✅ 解析成功！检测到 {len(doc_text)} 个字符。")
            with st.expander("🔍 查看解析出的文本预览 (前 500 字)"):
                st.text(doc_text[:500] + "...")
            input_text = f"【用户上传了附件：{uploaded_file.name}】\n\n请评估以下文件内容中的商业逻辑与项目信息：\n\n{doc_text}"
            
    # Regular Chat prompt takes precedence if both happen instantly (unlikely but safe)
    if prompt:
        input_text = prompt

    if input_text:
        if not st.session_state.current_session_id:
            create_new_session()
        
        # Determine the display string (we don't want to show the giant raw text to the user in the UI)
        display_text = prompt if prompt else f"📎 [上传了长文档：{uploaded_file.name}] 请全面分析其中的项目信息与商业漏洞。"
        
        st.session_state.messages.append({
            "role": "user",
            "content": display_text, # We store visual representation
            "raw_payload": input_text, # Keep raw input for AI
            "timestamp": datetime.now().isoformat(),
        })
        
        with chat_container:
            render_chat_message("user", display_text)
        
        conversation_history = []
        for msg in st.session_state.messages[:-1]:
            conversation_history.append({
                "role": msg.get("role"), 
                "content": msg.get("raw_payload", msg.get("content"))
            })
        
        st.session_state.accumulated_info["student_mode"] = current_mode

        if current_mode == "自由对话学习模式":
            spinner_text = "AI 导师正在结合知识图谱组织讲解与案例..."
            cycle_runner = run_learning_mode_cycle
        else:
            spinner_text = "AI 教练正在全量分析档案内容..."
            cycle_runner = run_langgraph_cycle

        with st.spinner(spinner_text):
            state = cycle_runner(
                input_text, # We feed the giant input text directly to backend
                conversation_history=conversation_history,
                accumulated_info=st.session_state.accumulated_info,
                target_competition=st.session_state.target_competition,
                student_id=st.session_state.user["id"],
            )

        st.session_state.accumulated_info = state.accumulated_info
        
        assistant_msg = {
            "role": "assistant",
            "content": state.response,
            "timestamp": datetime.now().isoformat(),
            "state": state.model_dump() if hasattr(state, "model_dump") else state.dict(),
        }
        st.session_state.messages.append(assistant_msg)
        
        with chat_container:
            render_chat_message("assistant", state.response, assistant_msg["state"])
        
        if len(st.session_state.messages) == 2:
            title_source = prompt if prompt else (uploaded_file.name if uploaded_file else "新对话")
            st.session_state.session_title = title_source[:30] + ("..." if len(title_source) > 30 else "")
        
        save_current_session()
        st.rerun()


if __name__ == "__main__":
    main()

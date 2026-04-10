import streamlit as st
from datetime import datetime
import uuid
import pypdf
import zipfile
import xml.etree.ElementTree as ET

from src.agents.langgraph_core import run_langgraph_cycle, COMPETITION_WEIGHTS, RUBRIC_DIM_NAMES
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


def generate_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

def extract_text_from_upload(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    try:
        if filename.endswith('.pdf'):
            reader = pypdf.PdfReader(uploaded_file)
            text = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text.append(extracted)
            return "\n".join(text)
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


def load_session_to_state(session_id: str):
    user_id = st.session_state.user["id"]
    data = load_user_session(user_id, session_id)
    if data:
        st.session_state.current_session_id = session_id
        st.session_state.messages = data.get("messages", [])
        st.session_state.session_title = data.get("title", "新对话")
        st.session_state.accumulated_info = data.get("accumulated_info", {})


def save_current_session():
    if st.session_state.user and st.session_state.current_session_id and st.session_state.messages:
        user_id = st.session_state.user["id"]
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
        
        st.markdown("**🏆 赛事目标设置**")
        st.session_state.target_competition = st.selectbox(
            "选择打分基准：",
            ["互联网+", "挑战杯", "创青春", "数模"],
            index=["互联网+", "挑战杯", "创青春", "数模"].index(st.session_state.target_competition)
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
    
    st.markdown("---")
    st.markdown("### 🔍 知识图谱查询轨迹")
    
    for idx, detail in enumerate(kg_query_details):
        with st.container():
            step = detail.get("step", "未知查询")
            query_type = detail.get("query_type", "")
            success = detail.get("success", False)
            
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
            
            query_attempts = detail.get("query_attempts", [])
            if query_attempts:
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
                    score = detail.get("match_scores", {}).get(proj.get('project_name', ''), 0)
                    with st.expander(f"📁 {proj.get('project_name', '未知项目')} (分数: {score})", expanded=False):
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
                if match_scores:
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
                if "GENTLE_INTERCEPTION" in fallacies or "GHOSTWRITING_INTERCEPTION" in fallacies:
                    return
                
                with st.expander("📊 详细分析", expanded=False):
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
                    
                    with st.expander("证据链追溯"):
                        for item in state.get("evidence", []):
                            st.markdown(f"**{item.get('step')}**: {item.get('detail')}")
                
                # ── A5: 赛事 Rubric 评分面板 ──
                rubric = state.get("rubric_scores", {})
                if rubric:
                    with st.expander("🏆 赛事 Rubric 评分", expanded=True):
                        # Use the sidebar's choice as the default to ensure reactivity
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
                        
                        # Missing Evidence & Minimal Fix
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
    
    render_sidebar()
    
    if st.session_state.get("show_student_profile", False) and st.session_state.view == "student":
        st.title("🎓 学生专属动态能力画像")
        if st.button("🔙 返回当前对话列表", type="primary"):
            st.session_state["show_student_profile"] = False
            st.rerun()
            
        st.markdown(st.session_state.get("my_ai_profile_content", "生成失败"))
        return
    
    col_t1, col_t2 = st.columns([3, 1.2])
    with col_t1:
        st.title("🎯 创新创业教学智能体")
        st.markdown("基于知识图谱与超图推理的创业项目诊断助手。")
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
    prompt = st.chat_input("请描述你的创业想法，或与我交流...")
    
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
        
        with st.spinner("AI 教练正在全量分析档案内容..."):
            state = run_langgraph_cycle(
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

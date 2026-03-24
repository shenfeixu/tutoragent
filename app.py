import streamlit as st
from datetime import datetime
import uuid

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
)

st.set_page_config(
    page_title="超图教练 - 创新创业教学智能体",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.assistant {
        background-color: #343541;
    }
    .session-item {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
    }
    .session-item:hover {
        background-color: #343541;
    }
    .session-item.active {
        background-color: #343541;
        border-left: 3px solid #10a37f;
    }
    div[data-testid="stSidebar"] {
        min-width: 280px;
    }
    .login-container {
        max-width: 400px;
        margin: 100px auto;
        padding: 2rem;
        border-radius: 1rem;
        background-color: #1e1e1e;
    }
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


def generate_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def create_new_session():
    st.session_state.current_session_id = generate_session_id()
    st.session_state.messages = []
    st.session_state.session_title = "新对话"
    st.session_state.accumulated_info = {}


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
    st.markdown("<h1 style='text-align: center;'>🎯 超图教练</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>创新创业教学智能体</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["登录", "注册"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("用户名", key="login_username")
                password = st.text_input("密码", type="password", key="login_password")
                submit = st.form_submit_button("登录", use_container_width=True)
                
                if submit:
                    if not username or not password:
                        st.error("请输入用户名和密码")
                    else:
                        user = authenticate_user(username, password)
                        if user:
                            st.session_state.user = user
                            update_last_login(user["id"])
                            st.rerun()
                        else:
                            st.error("用户名或密码错误")
        
        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("用户名", key="reg_username")
                new_password = st.text_input("密码", type="password", key="reg_password")
                confirm_password = st.text_input("确认密码", type="password", key="reg_confirm")
                display_name = st.text_input("显示名称", key="reg_displayname")
                role = st.selectbox("身份", ["student", "teacher", "admin"], format_func=lambda x: {"student": "学生", "teacher": "教师", "admin": "管理员"}.get(x, x))
                register = st.form_submit_button("注册", use_container_width=True)
                
                if register:
                    if not new_username or not new_password:
                        st.error("请填写用户名和密码")
                    elif new_password != confirm_password:
                        st.error("两次密码不一致")
                    elif len(new_password) < 6:
                        st.error("密码至少6位")
                    else:
                        user_id = create_user(
                            username=new_username,
                            password=new_password,
                            role=role,
                            display_name=display_name or new_username,
                        )
                        if user_id:
                            st.success("注册成功，请登录")
                        else:
                            st.error("用户名已存在")


def render_sidebar():
    with st.sidebar:
        user = st.session_state.user
        
        col1, col2 = st.columns([3, 1])
        with col1:
            role_text = "👨‍🏫" if user["role"] == "teacher" else "👨‍🎓"
            st.markdown(f"### {role_text} {user.get('display_name', user['username'])}")
        with col2:
            if st.button("🚪", help="退出登录"):
                st.session_state.user = None
                st.session_state.current_session_id = None
                st.session_state.messages = []
                st.rerun()
        
        st.caption(f"身份: {'教师' if user['role'] == 'teacher' else ('管理员' if user['role'] == 'admin' else '学生')}")
        
        if user["role"] == "teacher":
            if st.button("📊 教师端", use_container_width=True):
                st.session_state.view = "teacher"
                st.rerun()
                
        if user["role"] == "admin":
            if st.button("👑 管理端", use_container_width=True):
                st.session_state.view = "admin"
                st.rerun()
        
        st.divider()
        
        st.markdown("**🏆 赛事目标设置**")
        st.session_state.target_competition = st.selectbox(
            "选择打分基准：",
            ["互联网+", "挑战杯", "创青春", "数模"],
            index=["互联网+", "挑战杯", "创青春", "数模"].index(st.session_state.target_competition)
        )
        
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


def render_chat_message(role: str, content: str, state: dict = None):
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)
            
            if state:
                with st.expander("📊 详细分析", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**抽取的商业实体**")
                        st.json(state.get("extracted_nodes", {}))
                    with col2:
                        st.markdown("**触发的超图逻辑**")
                        fallacies = state.get("detected_fallacies", [])
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
    
    st.title("🎯 创新创业教学智能体")
    st.markdown("基于知识图谱与超图推理的创业项目诊断助手。输入你的创业想法，AI 教练将帮你分析商业逻辑。")
    
    if st.session_state.accumulated_info:
        with st.expander("📋 已收集的项目信息", expanded=False):
            info = st.session_state.accumulated_info
            cols = st.columns(3)
            with cols[0]:
                st.metric("项目名称", info.get("project_name", "未命名"))
                st.metric("技术成熟度", info.get("tech_maturity", "未设定"))
            with cols[1]:
                st.metric("目标市场", info.get("target_market", "未设定")[:15] + "..." if info.get("target_market") else "未设定")
                st.metric("团队规模", f"{info.get('team_size', 0)}人")
            with cols[2]:
                st.metric("融资阶段", info.get("funding_stage", "未设定"))
                st.metric("预计收入", f"{info.get('revenue', 0):,.0f}元" if info.get("revenue") else "未设定")
    
    st.divider()
    
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.messages:
            render_chat_message(
                msg.get("role"),
                msg.get("content"),
                msg.get("state") if msg.get("role") == "assistant" else None
            )
    
    if prompt := st.chat_input("请描述你的创业想法，或补充更多信息..."):
        if not st.session_state.current_session_id:
            create_new_session()
        
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat(),
        })
        
        with chat_container:
            render_chat_message("user", prompt)
        
        conversation_history = [
            {"role": msg.get("role"), "content": msg.get("content")}
            for msg in st.session_state.messages[:-1]
        ]
        
        with st.spinner("AI 教练正在分析..."):
            state = run_langgraph_cycle(
                prompt,
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
            st.session_state.session_title = prompt[:30] + ("..." if len(prompt) > 30 else "")
        
        save_current_session()
        st.rerun()


if __name__ == "__main__":
    main()

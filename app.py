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
    get_user_memory,
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
                login_role = st.selectbox("身份", ["student", "teacher", "admin"], format_func=lambda x: {"student": "学生", "teacher": "教师", "admin": "管理员"}.get(x, x), key="login_role_select")
                submit = st.form_submit_button("登录", use_container_width=True)
                
                if submit:
                    if not username or not password:
                        st.error("请输入用户名和密码")
                    else:
                        user = authenticate_user(username, password)
                        if user:
                            if user["role"] == login_role:
                                st.session_state.user = user
                                update_last_login(user["id"])
                                st.rerun()
                            else:
                                role_names = {"student": "学生", "teacher": "教师", "admin": "管理员"}
                                st.error(f"身份验证失败：该账号实际注册身份为【{role_names.get(user['role'], '未知')}】，请选择正确的身份登录！")
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

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from src.agents.langgraph_core import get_teaching_cases_for_risk, COMPETITION_WEIGHTS
from src.utils.database import (
    get_class_fallacy_stats,
    get_student_scores,
    get_teacher_students,
    get_all_students,
    add_student_to_teacher,
    get_student_sessions_for_teacher,
    get_all_intervention_rules,
    add_intervention_rule,
    delete_intervention_rule,
    create_class,
    get_teacher_classes,
    get_class_by_id,
    update_class,
    delete_class,
    add_student_to_class,
    add_students_to_class_batch,
    remove_student_from_class,
    get_class_students,
    get_students_not_in_class,
    get_all_students_for_teacher,
)

st.set_page_config(
    page_title="教师端 - 超图教练",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* =========================================================
       INSTRUCTOR PREMIUM UI
       ========================================================= */
    .stApp {
        background: radial-gradient(circle at top left, #0D1629 0%, #030712 100%);
    }

    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #2563EB 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 600 !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    
    /* Metrics / Dashboard Cards wrapper in pure CSS */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(30,41,59,0.5), rgba(15,23,42,0.8));
        padding: 20px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        background: linear-gradient(145deg, rgba(30,41,59,0.7), rgba(15,23,42,0.9));
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #F8FAFC !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        color: #94A3B8 !important;
        margin-bottom: 5px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(30, 41, 59, 0.6) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    .risk-high { background-color: rgba(239, 68, 68, 0.2); color: #FCA5A5; padding: 2px 8px; border-radius: 4px; border: 1px solid #EF4444;}
    .risk-medium { background-color: rgba(245, 158, 11, 0.2); color: #FCD34D; padding: 2px 8px; border-radius: 4px; border: 1px solid #F59E0B; }
    .risk-low { background-color: rgba(16, 163, 127, 0.2); color: #6EE7B7; padding: 2px 8px; border-radius: 4px; border: 1px solid #10A37F; }
    
    .hero-title {
        font-size: 2.8rem; font-weight: 900; 
        background: -webkit-linear-gradient(45deg, #60A5FA, #A78BFA); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

FALLACY_DESCRIPTIONS = {
    "H1": "技术-市场匹配度不足",
    "H2": "技术成熟度低",
    "H3": "目标客户模糊",
    "H4": "市场规模数据不合理",
    "H5": "价值主张不清晰",
    "H6": "获客渠道不明确",
    "H7": "收入预测缺失",
    "H8": "单位经济不健康(LTV/CAC)",
    "H9": "团队规模不足",
    "H10": "融资阶段不匹配",
    "H11": "上市时间过于乐观",
    "H12": "风险识别不充分",
    "H13": "技术壁垒不高",
    "H14": "市场规模验证不足",
    "H15": "商业模式闭环不完整",
    "H16": "单位经济幻觉",
    "H17": "渠道与用户群体错位",
    "H18": "现金流生存风险",
    "H19": "护城河未明确",
    "H20": "增长飞轮未定义",
}

RUBRIC_DIMENSIONS = {
    "pain_point_score": "痛点发现",
    "planning_score": "方案策划",
    "modeling_score": "商业建模",
    "leverage_score": "资源杠杆",
    "presentation_score": "路演表达",
}


def render_sidebar(user):
    with st.sidebar:
        st.markdown(f"### 👨‍🏫 {user.get('display_name', user['username'])}")
        st.caption("教师端")
        st.divider()
        
        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state.user = None
            st.session_state.view = None
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.session_state.accumulated_info = {}
            st.rerun()
        
        st.divider()
        
        page = st.radio(
            "导航",
            ["📊 班级概览", "👥 教学班管理", "📈 详细分析", "✨ 动态能力画像", "🛠 教学干预", "📚 教学案例"],
            key="teacher_page",
        )
        
        return page


def render_class_overview():
    st.markdown("<div class='hero-title'>📊 班级全景洞察矩阵</div>", unsafe_allow_html=True)
    
    stats = get_class_fallacy_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总会话数", stats["total_sessions"])
    with col2:
        st.metric("错误模式种类", len(stats["fallacy_counts"]))
    with col3:
        avg_errors = sum(stats["fallacy_counts"].values()) / max(1, stats["total_sessions"])
        st.metric("平均错误数/会话", f"{avg_errors:.1f}")
    
    st.divider()
    
    if stats["top_5"]:
        st.subheader("🔴 Top 5 错误模式")
        
        top5_data = []
        for fallacy, count in stats["top_5"]:
            percentage = count / max(1, stats["total_sessions"]) * 100
            top5_data.append({
                "规则": fallacy,
                "描述": FALLACY_DESCRIPTIONS.get(fallacy, "未知"),
                "触发次数": count,
                "触发率": f"{percentage:.1f}%",
            })
        
        df_top5 = pd.DataFrame(top5_data)
        st.dataframe(df_top5, use_container_width=True, hide_index=True)
        
        fig = px.bar(
            df_top5,
            x="规则",
            y="触发次数",
            color="触发次数",
            color_continuous_scale="Reds",
            title="Top 5 错误模式分布",
            text="触发率",
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # A6: 教学干预计划生成
        st.divider()
        with st.expander("✨ AI 自动生成：下周教学干预计划", expanded=False):
            if st.button("生成/刷新干预计划"):
                with st.spinner("AI 正在分析班级数据，生成教学计划中..."):
                    from src.agents.langgraph_core import generate_intervention_plan
                    stats_data = {
                        "total_sessions": stats.get("total_sessions", 0),
                        "top_5_fallacies": [{"rule": f, "desc": FALLACY_DESCRIPTIONS.get(f, ""), "count": c} for f, c in stats.get("top_5", [])]
                    }
                    plan_md = generate_intervention_plan(stats_data)
                    st.session_state.intervention_plan = plan_md
            
            if "intervention_plan" in st.session_state:
                st.markdown(st.session_state.intervention_plan)
    else:
        st.info("暂无错误数据，请等待学生完成对话。")
    
    st.divider()
    
    st.subheader("📊 全部错误模式分布")
    if stats["fallacy_counts"]:
        all_data = []
        for fallacy, count in sorted(stats["fallacy_counts"].items(), key=lambda x: x[1], reverse=True):
            all_data.append({
                "规则": fallacy,
                "描述": FALLACY_DESCRIPTIONS.get(fallacy, "未知"),
                "触发次数": count,
            })
        
        df_all = pd.DataFrame(all_data)
        fig_all = px.pie(
            df_all,
            values="触发次数",
            names="规则",
            title="错误模式占比",
            hole=0.4,
        )
        st.plotly_chart(fig_all, use_container_width=True)


def render_student_management(user):
    st.header("👥 教学班管理")
    
    tab1, tab2 = st.tabs(["我的教学班", "创建教学班"])
    
    with tab1:
        classes = get_teacher_classes(user["id"])
        
        if classes:
            for cls in classes:
                with st.expander(f"📚 {cls['name']} ({cls['student_count']}名学生)", expanded=False):
                    col_info, col_actions = st.columns([3, 1])
                    with col_info:
                        st.markdown(f"**描述**: {cls['description'] or '无描述'}")
                        st.caption(f"创建时间: {cls['created_at'][:10] if cls['created_at'] else '-'}")
                    with col_actions:
                        if st.button("🗑️ 删除班级", key=f"del_class_{cls['id']}"):
                            if delete_class(cls['id']):
                                st.success("班级已删除")
                                st.rerun()
                            else:
                                st.error("删除失败")
                    
                    st.divider()
                    
                    class_students = get_class_students(cls['id'])
                    if class_students:
                        st.markdown("**班级学生**")
                        df_students = pd.DataFrame([
                            {
                                "姓名": s["display_name"],
                                "用户名": s["username"],
                                "邮箱": s.get("email", "-"),
                                "加入时间": s.get("joined_at", "-")[:10] if s.get("joined_at") else "-",
                            }
                            for s in class_students
                        ])
                        st.dataframe(df_students, use_container_width=True, hide_index=True)
                    else:
                        st.info("该班级暂无学生")
                    
                    st.divider()
                    
                    st.markdown("**添加学生到班级**")
                    available_students = get_students_not_in_class(user["id"], cls['id'])
                    
                    if available_students:
                        student_options = {f"{s['display_name']} ({s['username']})": s["id"] for s in available_students}
                        
                        col_select, col_btn = st.columns([3, 1])
                        with col_select:
                            selected_students = st.multiselect(
                                "选择学生（可多选）",
                                list(student_options.keys()),
                                key=f"multiselect_{cls['id']}"
                            )
                        with col_btn:
                            st.write("")
                            st.write("")
                            if st.button("批量添加", key=f"add_batch_{cls['id']}"):
                                if selected_students:
                                    student_ids = [student_options[s] for s in selected_students]
                                    count = add_students_to_class_batch(cls['id'], student_ids)
                                    st.success(f"成功添加 {count} 名学生")
                                    st.rerun()
                                else:
                                    st.warning("请先选择学生")
                    else:
                        st.info("所有学生已在该班级中，或暂无可添加的学生")
        else:
            st.info("暂无教学班，请在「创建教学班」标签页创建。")
    
    with tab2:
        st.subheader("创建新教学班")
        with st.form("create_class_form"):
            class_name = st.text_input("班级名称", placeholder="例如：2024春季创业基础班")
            class_desc = st.text_area("班级描述", placeholder="可选：描述该班级的课程信息、学期等")
            
            st.divider()
            st.markdown("**可选：批量导入学生**")
            all_students = get_all_students_for_teacher()
            
            if all_students:
                student_options = {f"{s['display_name']} ({s['username']})": s["id"] for s in all_students}
                pre_select_students = st.multiselect(
                    "选择要导入的学生（可多选，也可创建班级后再添加）",
                    list(student_options.keys())
                )
            else:
                student_options = {}
                pre_select_students = []
                st.info("暂无可导入的学生")
            
            submit = st.form_submit_button("创建教学班", use_container_width=True)
            
            if submit:
                if not class_name.strip():
                    st.error("请输入班级名称")
                else:
                    class_id = create_class(user["id"], class_name.strip(), class_desc.strip() if class_desc else None)
                    if class_id:
                        if pre_select_students:
                            student_ids = [student_options[s] for s in pre_select_students]
                            count = add_students_to_class_batch(class_id, student_ids)
                            st.success(f"教学班创建成功！已导入 {count} 名学生")
                        else:
                            st.success("教学班创建成功！")
                        st.rerun()
                    else:
                        st.error("创建失败，请重试")


def render_detailed_analysis(user):
    st.header("📈 详细分析")
    
    # 赛事权重选择
    col_a, col_b = st.columns([2, 3])
    with col_a:
        target_comp = st.selectbox(
            "评估基准：",
            list(COMPETITION_WEIGHTS.keys()),
            key="teacher_target_comp"
        )
    weights = COMPETITION_WEIGHTS[target_comp]
    
    student_scores = get_student_scores(weights)
    
    if not student_scores:
        st.info("暂无学生数据，请等待学生完成对话。")
        return
    
    tab1, tab2, tab3 = st.tabs(["量化评分", "高风险项目", "证据链追溯"])
    
    with tab1:
        st.subheader("基于 Rubric 的量化评价")
        
        score_data = []
        for s in student_scores:
            score_data.append({
                "学生": s["display_name"],
                "痛点发现": s["pain_point_score"],
                "方案策划": s["planning_score"],
                "商业建模": s["modeling_score"],
                "资源杠杆": s["leverage_score"],
                "路演表达": s["presentation_score"],
                "综合得分": round(s["total_score"], 1),
                "风险等级": s["risk_level"],
            })
        
        df_scores = pd.DataFrame(score_data)
        st.dataframe(df_scores, use_container_width=True, hide_index=True)
        
        radar_data = []
        for dim, name in RUBRIC_DIMENSIONS.items():
            avg_score = sum(s[dim] for s in student_scores) / len(student_scores)
            radar_data.append({"维度": name, "平均分": avg_score})
        
        df_radar = pd.DataFrame(radar_data)
        fig_radar = px.line_polar(
            df_radar,
            r="平均分",
            theta="维度",
            line_close=True,
            title="班级能力雷达图",
            range_r=[0, 100],
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab2:
        st.subheader("⚠️ 高风险项目清单")
        
        high_risk = [s for s in student_scores if s["risk_level"] == "高"]
        medium_risk = [s for s in student_scores if s["risk_level"] == "中"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔴 高风险项目**")
            if high_risk:
                for s in high_risk:
                    with st.expander(f"{s['display_name']} - 综合得分: {s['total_score']:.1f}"):
                        st.write(f"触发规则: {', '.join(s['fallacies'])}")
            else:
                st.success("暂无高风险项目")
        
        with col2:
            st.markdown("**🟡 中风险项目**")
            if medium_risk:
                for s in medium_risk:
                    with st.expander(f"{s['display_name']} - 综合得分: {s['total_score']:.1f}"):
                        st.write(f"触发规则: {', '.join(s['fallacies'])}")
            else:
                st.success("暂无中风险项目")
    
    with tab3:
        st.subheader("📋 证据链追溯")
        
        student_options = {s["display_name"]: s for s in student_scores}
        selected_student = st.selectbox("选择学生", list(student_options.keys()))
        
        if selected_student:
            student = student_options[selected_student]
            
            if student["sessions"]:
                for i, session in enumerate(student["sessions"]):
                    with st.expander(f"会话 {i+1}: {session['title']}"):
                        if session["evidence"]:
                            for ev in session["evidence"]:
                                st.markdown(f"**{ev.get('step', 'N/A')}**: {ev.get('detail', 'N/A')}")
                        else:
                            st.write("暂无证据记录")
            else:
                st.info("该学生暂无会话记录。")
                
    
def render_dynamic_profile(user):
    st.header("✨ 学生动态能力画像评估")
    st.markdown("基于全量真实交互上下文，由 AI 自动生成深度的创新创业核心能力剖析。")
    
    target_comp = st.selectbox(
        "评估基准：",
        list(COMPETITION_WEIGHTS.keys()),
        key="teacher_profile_sync_comp"
    )
    weights = COMPETITION_WEIGHTS[target_comp]
    student_scores = get_student_scores(weights)
    
    if not student_scores:
        st.info("暂无学生数据。")
        return
        
    student_options = {s["display_name"]: s for s in student_scores}
    selected_student = st.selectbox("🎯 选择要深入解剖的学生", list(student_options.keys()), key="profile_student")
    
    if selected_student:
        student = student_options[selected_student]
        st.markdown(f"**当前量化得分：{round(student['total_score'], 1)}** · **风险定级：**`{student['risk_level']}`")
        
        if st.button("🚀 提纯并生成最新 AI 画像"):
            with st.spinner(f"正在全维检索并深扒 {selected_student} 的交互历史与商战盲区..."):
                from src.agents.langgraph_core import generate_student_profile
                
                student_data = {
                    "name": student["display_name"],
                    "total_score": round(student["total_score"], 1),
                    "risk_level": student["risk_level"],
                    "rubric_scores": {
                        "pain_point": student["pain_point_score"],
                        "planning": student["planning_score"],
                        "modeling": student["modeling_score"],
                        "leverage": student["leverage_score"],
                        "presentation": student["presentation_score"],
                    },
                    "frequent_fallacies": student["fallacies"],
                    "session_count": len(student["sessions"])
                }
                
                profile_md = generate_student_profile(student_data)
                st.session_state[f"profile_{selected_student}"] = profile_md
        
        profile_key = f"profile_{selected_student}"
        if profile_key in st.session_state:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(st.session_state[profile_key])


def render_teaching_cases():
    st.header("📚 教学案例推荐")
    
    stats = get_class_fallacy_stats()
    
    st.markdown("""
    根据班级薄弱项，从知识图谱中调取匹配的教学案例。
    
    **案例来源**: Neo4j 知识图谱中的项目数据
    """)
    
    st.divider()
    
    if stats["top_5"]:
        st.subheader("🎯 针对性教学建议")
        
        for fallacy, count in stats["top_5"]:
            with st.container():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**{fallacy}**: {FALLACY_DESCRIPTIONS.get(fallacy, '未知')}")
                    st.caption(f"触发率: {count / max(1, stats['total_sessions']) * 100:.1f}%")
                with col2:
                    cases = get_teaching_cases_for_risk(FALLACY_DESCRIPTIONS.get(fallacy, ""))
                    if cases:
                        for case in cases:
                            st.info(f"**{case['project_name']}**: 技术: {', '.join(case['techs'])} | 市场: {', '.join(case['markets'])}")
                            st.caption(f"风险: {case['risk']}")
                    else:
                        default_tips = {
                            "H1": "建议讲解技术-市场匹配分析方法",
                            "H2": "建议讲解技术成熟度评估框架",
                            "H3": "建议讲解用户画像方法论",
                            "H4": "建议讲解市场规模估算方法",
                            "H5": "建议讲解价值主张设计",
                            "H6": "建议讲解增长黑客方法论",
                            "H7": "建议讲解收入模型设计",
                            "H8": "建议讲解单位经济模型",
                            "H9": "建议讲解团队建设策略",
                            "H10": "建议讲解融资策略",
                            "H11": "建议讲解项目管理方法",
                            "H12": "建议讲解风险管理框架",
                            "H13": "建议讲解技术壁垒构建",
                            "H14": "建议讲解市场调研方法",
                            "H15": "建议讲解商业模式画布",
                            "H16": "建议讲解单位经济验证方法，避免数据幻觉",
                            "H17": "建议讲解渠道与用户匹配分析方法",
                            "H18": "建议讲解现金流管理与生存分析",
                            "H19": "建议讲解竞争壁垒与护城河构建",
                            "H20": "建议讲解增长飞轮与自增强机制设计",
                        }
                        st.info(default_tips.get(fallacy, "暂无相关教学建议"))
    else:
        st.info("暂无班级薄弱项数据，请等待学生完成对话。")


def render_teacher_intervention(user):
    st.header("🛠 教学干预中心")
    st.markdown("在此下发针对全班或特定学生的“AI 指令”，实时干预 AI 教练的反馈偏好。")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("➕ 新增干预规则")
        with st.form("add_rule_form"):
            students = get_teacher_students(user["id"])
            student_options = {"全班 (Class-wide)": None}
            for s in students:
                student_options[f"{s['display_name']} ({s['username']})"] = s["id"]
            
            target = st.selectbox("针对对象", list(student_options.keys()))
            rule_content = st.text_area("指令内容", placeholder="例如：本周重点考核商业模式的闭环性，对财务数据不全的项目请严厉指出。")
            
            if st.form_submit_button("发布指令"):
                if rule_content.strip():
                    add_intervention_rule(user["id"], rule_content, student_options[target])
                    st.success("指令发布成功！")
                    st.rerun()
                else:
                    st.error("请输入指令内容")
    
    with col2:
        st.subheader("📋 当前有效指令")
        rules = get_all_intervention_rules(user["id"])
        if rules:
            for rule in rules:
                target_str = f"🎯 {rule['student_name']}" if rule['student_id'] else "📢 全班"
                status_color = "green" if rule['is_active'] else "gray"
                
                with st.expander(f"{target_str} | {rule['created_at'][:16]}", expanded=True):
                    st.markdown(f"**指令**: {rule['content']}")
                    if st.button("删除指令", key=f"del_{rule['id']}"):
                        delete_intervention_rule(rule["id"])
                        st.rerun()
        else:
            st.info("当前没有活跃的干预指令。")


def main():
    if "user" not in st.session_state or not st.session_state.user:
        st.error("请先登录")
        st.stop()
    
    user = st.session_state.user
    
    if user["role"] != "teacher":
        st.error("此页面仅限教师访问")
        if st.button("返回学生端"):
            st.session_state.view = "student"
            st.rerun()
        st.stop()
    
    page = render_sidebar(user)
    
    if page == "📊 班级概览":
        render_class_overview()
    elif page == "👥 教学班管理":
        render_student_management(user)
    elif page == "📈 详细分析":
        render_detailed_analysis(user)
    elif page == "✨ 动态能力画像":
        render_dynamic_profile(user)
    elif page == "🛠 教学干预":
        render_teacher_intervention(user)
    elif page == "📚 教学案例":
        render_teaching_cases()


if __name__ == "__main__":
    main()

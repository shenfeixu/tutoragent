import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from src.agents.langgraph_core import get_teaching_cases_for_risk
from src.utils.database import (
    get_class_fallacy_stats,
    get_student_scores,
    get_teacher_students,
    get_all_students,
    add_student_to_teacher,
    get_student_sessions_for_teacher,
)

st.set_page_config(
    page_title="教师端 - 超图教练",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .risk-high { color: #ff4b4b; }
    .risk-medium { color: #ffa500; }
    .risk-low { color: #4caf50; }
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
        
        if st.button("🏠 返回学生端", use_container_width=True):
            st.session_state.view = "student"
            st.rerun()
        
        st.divider()
        
        page = st.radio(
            "导航",
            ["📊 班级概览", "👥 学生管理", "📈 详细分析", "📚 教学案例"],
            key="teacher_page",
        )
        
        return page


def render_class_overview():
    st.header("📊 班级能力分布与预警")
    
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
    st.header("👥 学生管理")
    
    tab1, tab2 = st.tabs(["我的学生", "添加学生"])
    
    with tab1:
        my_students = get_teacher_students(user["id"])
        
        if my_students:
            df_students = pd.DataFrame([
                {
                    "姓名": s["display_name"],
                    "用户名": s["username"],
                    "邮箱": s.get("email", "-"),
                    "注册时间": s.get("created_at", "-")[:10] if s.get("created_at") else "-",
                }
                for s in my_students
            ])
            st.dataframe(df_students, use_container_width=True, hide_index=True)
        else:
            st.info("暂无学生，请在「添加学生」标签页添加。")
    
    with tab2:
        all_students = get_all_students()
        my_student_ids = [s["id"] for s in get_teacher_students(user["id"])]
        available_students = [s for s in all_students if s["id"] not in my_student_ids]
        
        if available_students:
            student_options = {f"{s['display_name']} ({s['username']})": s["id"] for s in available_students}
            
            selected = st.selectbox("选择学生", list(student_options.keys()))
            
            if st.button("添加学生"):
                student_id = student_options[selected]
                if add_student_to_teacher(user["id"], student_id):
                    st.success("添加成功！")
                    st.rerun()
                else:
                    st.error("添加失败，请重试。")
        else:
            st.info("没有可添加的学生。")


def render_detailed_analysis(user):
    st.header("📈 详细分析")
    
    student_scores = get_student_scores()
    
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
                        }
                        st.info(default_tips.get(fallacy, "暂无相关教学建议"))
    else:
        st.info("暂无班级薄弱项数据，请等待学生完成对话。")


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
    elif page == "👥 学生管理":
        render_student_management(user)
    elif page == "📈 详细分析":
        render_detailed_analysis(user)
    elif page == "📚 教学案例":
        render_teaching_cases()


if __name__ == "__main__":
    main()

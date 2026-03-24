import streamlit as st
import pandas as pd
from src.utils.database import get_system_stats, get_all_users

def render_sidebar():
    with st.sidebar:
        st.write(f"管理员端")
        st.divider()
        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state.user = None
            st.session_state.view = None
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.session_state.accumulated_info = {}
            st.rerun()

def main():
    if "user" not in st.session_state or not st.session_state.user:
        st.error("请先登录。")
        st.stop()
        
    user = st.session_state.user
    if user.get("role") != "admin":
        st.error("🛑 403 Forbidden: 您没有管理员权限访问此页面。")
        st.stop()
        
    st.title(f"👑 管理员控制台，欢迎 {user['display_name']}")
    render_sidebar()
    
    stats = get_system_stats()
    
    st.subheader("📊 全局数据概览")
    cols = st.columns(4)
    cols[0].metric("👨‍🎓 学生总数", stats["student_count"])
    cols[1].metric("👩‍🏫 导师总数", stats["teacher_count"])
    cols[2].metric("💬 总会话数", stats["session_count"])
    cols[3].metric("📈 预估消息数", stats["estimated_messages"])
    
    st.divider()
    
    # ── A6-5: 全局健康度与漏洞看板 ──
    from src.utils.database import get_global_fallacy_stats, get_global_health_metrics
    import plotly.express as px
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🛡️ 全局教练健康度")
        health = get_global_health_metrics()
        st.metric("全局平均分 (Global Health Index)", f"{health['avg_total']:.2f} / 100")
        
        # 维度细分
        avg_dims = health["avg_dims"]
        dim_labels = {
            "pain_point": "痛点发现",
            "planning": "方案策划",
            "modeling": "商业建模",
            "leverage": "资源杠杆",
            "presentation": "路演表达"
        }
        health_data = [{"维度": dim_labels[k], "得分": v} for k, v in avg_dims.items()]
        fig_health = px.line_polar(pd.DataFrame(health_data), r="得分", theta="维度", line_close=True, range_r=[0, 100])
        st.plotly_chart(fig_health, use_container_width=True)

    with col2:
        st.subheader("🚩 TOP 5 逻辑漏洞榜单")
        f_stats = get_global_fallacy_stats()
        if f_stats["top_5"]:
            df_fallacy = pd.DataFrame(f_stats["top_5"], columns=["漏洞代码", "触发频次"])
            fig_fallacy = px.bar(
                df_fallacy, 
                x="触发频次", 
                y="漏洞代码", 
                orientation='h',
                color="触发频次",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_fallacy, use_container_width=True)
        else:
            st.info("暂无漏洞统计数据")
    
    st.divider()
    st.subheader("👥 用户列表")
    
    users = get_all_users()
    if users:
        df_users = pd.DataFrame(users)
        df_users["角色"] = df_users["role"].map({"admin": "👑 管理员", "teacher": "👩‍🏫 教师", "student": "👨‍🎓 学生"})
        df_users = df_users.rename(columns={
            "id": "ID", "username": "用户名", "display_name": "显示名称", 
            "email": "邮箱", "created_at": "注册时间", "last_login": "上次登录"
        })
        st.dataframe(df_users[["ID", "用户名", "显示名称", "角色", "注册时间", "上次登录"]], use_container_width=True, hide_index=True)
    else:
        st.info("暂无用户数据")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
from src.utils.database import (
    get_system_stats, 
    get_all_users, 
    delete_user, 
    create_user,
    get_global_fallacy_stats, 
    get_global_health_metrics
)
import plotly.express as px

def render_sidebar():
    with st.sidebar:
        st.write("👑 管理员端")
        st.divider()
        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state.user = None
            st.session_state.view = None
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.session_state.accumulated_info = {}
            st.rerun()

st.markdown("""
<style>
    /* =========================================================
       ADMIN PREMIUM UI
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
    
    /* Metrics / Dashboard Cards wrapper */
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
    }
    
    .hero-title {
        font-size: 2.5rem; font-weight: 900; 
        background: -webkit-linear-gradient(45deg, #60A5FA, #A78BFA); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    if "user" not in st.session_state or not st.session_state.user:
        st.error("请先登录。")
        st.stop()
        
    user = st.session_state.user
    if user.get("role") != "admin":
        st.error("🛑 403 Forbidden: 您没有管理员权限访问此页面。")
        st.stop()
        
    st.title(f"👑 管理员控制台")
    st.caption(f"欢迎回来, {user['display_name']} | 系统当前状态：运行中")
    render_sidebar()
    
    tab_dashboard, tab_users = st.tabs(["📊 数据看板", "👥 用户管理"])
    
    with tab_dashboard:
        stats = get_system_stats()
        
        st.subheader("全局数据概览")
        cols = st.columns(4)
        cols[0].metric("👨‍🎓 学生总数", stats["student_count"])
        cols[1].metric("👩‍🏫 导师总数", stats["teacher_count"])
        cols[2].metric("💬 总会话数", stats["session_count"])
        cols[3].metric("📈 预估消息数", stats["estimated_messages"])
        
        st.divider()
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("🛡️ 全局教练健康度")
            health = get_global_health_metrics()
            st.metric("全局平均分 (Global Health Index)", f"{health['avg_total']:.2f} / 100")
            
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

    with tab_users:
        st.subheader("🔑 账号管理")
        
        # 新增用户表单
        with st.expander("➕ 开通新账号", expanded=False):
            with st.form("admin_create_user"):
                c1, c2 = st.columns(2)
                new_un = c1.text_input("用户名")
                new_dn = c1.text_input("显示名称（姓名）")
                new_pw = c2.text_input("初始密码", type="password")
                new_role = c2.selectbox("角色", ["student", "teacher", "admin"], format_func=lambda x: {"student": "学生", "teacher": "教师", "admin": "管理员"}.get(x))
                
                if st.form_submit_button("立即创建"):
                    if new_un and new_pw:
                        uid = create_user(new_un, new_pw, new_role, new_dn)
                        if uid:
                            st.success(f"✅ 账号 {new_un} 创建成功！")
                            st.rerun()
                        else:
                            st.error("❌ 创建失败：用户名可能已存在。")
                    else:
                        st.error("请填写必填项。")

                        st.error("请填写必填项。")

        # 新增批量导入表单
        with st.expander("📂 批量导入账号 (CSV/Excel 模拟)", expanded=False):
            st.markdown("💡 仅支持包含 `username, password, role, display_name` 四列的标准 CSV 文件。")
            
            # 提供模板下载模拟
            csv_template = "username,password,role,display_name\nstu01,123456,student,学生张三\ntea01,123456,teacher,导师李四"
            st.download_button(label="📥 下载 CSV 导入模板", data=csv_template, file_name="账号批量导入模板.csv", mime="text/csv")
            
            uploaded_csv = st.file_uploader("上传已填写的分配表", type=["csv"], label_visibility="collapsed")
            if st.button("🚀 执行批量分配写入"):
                if uploaded_csv:
                    try:
                        import io
                        df_import = pd.read_csv(io.StringIO(uploaded_csv.getvalue().decode("utf-8")))
                        success_cnt = 0
                        fails = 0
                        for _, row in df_import.iterrows():
                            un = str(row.get("username", "")).strip()
                            pw = str(row.get("password", "123456")).strip()
                            rl = str(row.get("role", "student")).strip()
                            dn = str(row.get("display_name", un)).strip()
                            
                            if un and pw:
                                uid = create_user(un, pw, rl, dn)
                                if uid: 
                                    success_cnt += 1
                                else:
                                    fails += 1
                        
                        st.success(f"✅ 成功批量分配并写入 {success_cnt} 个账号！(忽略/覆盖 {fails} 个)")
                    except Exception as e:
                        st.error(f"❌ 读取 CSV 文件失败，请检查格式约束。详情: {e}")
                else:
                    st.warning("请先上传文件。")

        st.divider()
        
        # 用户列表与删除
        users = get_all_users()
        if users:
            df_users = pd.DataFrame(users)
            role_map = {"admin": "👑 管理员", "teacher": "👩‍🏫 教师", "student": "👨‍🎓 学生"}
            
            # 使用列表显示用户，方便放置删除按钮
            for _, row in df_users.iterrows():
                with st.container():
                    c_info, c_action = st.columns([4, 1])
                    with c_info:
                        st.markdown(f"**{row['display_name']}** (@{row['username']})")
                        st.caption(f"角色: {role_map.get(row['role'], row['role'])} | 注册时间: {row['created_at'][:16]}")
                    with c_action:
                        # 禁止自删
                        if row["id"] == st.session_state.user["id"]:
                            st.button("当前登录", disabled=True, key=f"self_{row['id']}")
                        else:
                            if st.button("🗑️ 删除", key=f"del_u_{row['id']}", type="secondary"):
                                if delete_user(row["id"]):
                                    st.toast(f"已删除用户: {row['username']}")
                                    st.rerun()
                                else:
                                    st.error("删除失败")
                st.divider()
        else:
            st.info("系统中尚无其他用户")

if __name__ == "__main__":
    main()

#!/bin/bash
# TutorAgent v1.55 一键部署脚本

echo "🚀 开始部署 TutorAgent v1.55..."

# 1. 确保安装了新依赖
echo "📦 检查并更新绘图依赖 (plotly)..."
pip install plotly --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 同步代码
echo "🔄 拉取 GitHub 最新 v1.55 稳定版..."
git fetch origin
git checkout main
git pull origin main
git checkout v1.55

# 3. 管理进程
echo "🛑 正在停止旧版本服务..."
# 杀掉所有 streamlit 进程
pkill -f streamlit
sleep 2

# 4. 后台启动新版本
echo "🔥 正在启动 v1.55 服务 (Port: 8501)..."
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > tutor_v155.log 2>&1 &

echo "✨ 部署完成！"
echo "💡 提示：您可以使用 'tail -f tutor_v155.log' 查看实时运行日志。"

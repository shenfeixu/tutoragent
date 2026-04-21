#!/bin/bash
echo "=================================================="
echo "  TutorAgent v1.51 - 生产环境稳定版部署开始"
echo "  现代化改造完成版 | 纯净文字审计列表 & 评价流转"
echo "=================================================="

# 1. 从 GitHub 推送到本地后，服务器端建议先备份
echo "[1/4] 从 GitHub 拉取主线代码与标签..."
git fetch origin main
git fetch origin v1.51
git reset --hard origin/main
echo "  OK: 代码已同步至最新 v1.51 生产状态 (Git ID: $(git rev-parse --short HEAD))"

# 2. 清理旧进程 (Port 8240)
echo "[2/4] 清理 8240 端口旧进程..."
OLD_PID=$(fuser 8240/tcp 2>/dev/null)
if [ ! -z "$OLD_PID" ]; then
    fuser -k 8240/tcp
    echo "  OK: 已杀死旧进程 $OLD_PID"
else
    echo "  OK: 端口 8240 目前空闲"
fi

# 3. 环境加固
echo "[3/4] 正在安装/同步依赖环境 (排除 Plotly)..."
pip install -r requirements.txt --quiet
echo "  OK: 生产环境依赖已就绪"

# 4. 启动服务 (后台运行)
echo "[4/4] 启动 v1.51 Streamlit 服务 (Port 8240)..."
nohup streamlit run app.py --server.port 8240 --server.address 0.0.0.0 > server_v1.51_production.log 2>&1 &

echo "=================================================="
echo "  部署成功！"
echo "  访问地址: http://<您的服务器IP>:8240"
echo "  运行日志见: server_v1.51_production.log"
echo "=================================================="

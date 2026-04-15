#!/bin/bash
echo "=================================================="
echo "  TutorAgent v1.35 - VC-Grade 史诗版部署开始"
echo "  12章节专业BP引擎 + Unicode 鲁棒加固"
echo "=================================================="

# 1. 检查 Neo4j
echo "[1/4] 检查 Neo4j Docker 容器状态..."
if docker ps | grep -q "group24-neo4j"; then
    echo "  OK: group24-neo4j 正在运行 (Port 8224)"
else
    echo "  ERROR: Neo4j 容器未运行，尝试启动..."
    docker start group24-neo4j
fi

# 2. 清理旧进程
echo "[2/4] 清理 8240 端口旧进程..."
OLD_PID=$(fuser 8240/tcp 2>/dev/null)
if [ ! -z "$OLD_PID" ]; then
    fuser -k 8240/tcp
    echo "  OK: 已杀死旧进程 $OLD_PID"
else
    echo "  OK: 端口 8240 目前空闲"
fi

# 3. 安装依赖 (以防万一)
echo "[3/4] 检查环境依赖..."
pip install -r requirements.txt --quiet
echo "  OK: 依赖环境已就绪"

# 4. 启动服务
echo "[4/4] 启动 Streamlit 服务 (Port 8240)..."
nohup streamlit run app.py --server.port 8240 --server.address 0.0.0.0 > server_v1.35.log 2>&1 &

echo "=================================================="
echo "  部署完成！"
echo "  访问地址: http://<SERVER_IP>:8240"
echo "  运行日志见: server_v1.35.log"
echo "=================================================="

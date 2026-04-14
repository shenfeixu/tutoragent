#!/bin/bash
# ============================================================
# TutorAgent v1.30 部署脚本
# 部署环境: Ubuntu 22.04 (inspur-NF5468M6)
# Neo4j: Docker 容器 group24-neo4j (端口映射 8224:7687)
# Streamlit: 端口 8240
# ============================================================

echo "=================================================="
echo "  TutorAgent v1.30 - 部署开始"
echo "  财务分析引擎 + 199案例知识图谱"
echo "=================================================="

PORT=8240

# ── 1. 确认 Neo4j Docker 容器状态 ──
echo ""
echo "[1/5] 检查 Neo4j Docker 容器状态..."
if docker ps --format '{{.Names}}' | grep -q "group24-neo4j"; then
    echo "  OK: group24-neo4j 容器正在运行"
else
    echo "  WARNING: group24-neo4j 容器未运行，尝试启动..."
    docker start group24-neo4j 2>/dev/null || echo "  ERROR: 无法启动容器，请手动检查 docker ps -a"
fi

# ── 2. 确认 .env 配置正确 ──
echo ""
echo "[2/5] 校验 .env 配置..."
if grep -q "localhost:8224" .env; then
    echo "  OK: Neo4j 端口已正确配置为 8224"
else
    echo "  FIX: 修正 Neo4j 端口为 8224..."
    sed -i 's|NEO4J_URI=.*|NEO4J_URI=bolt://localhost:8224|' .env
fi
echo "  当前配置:"
grep "NEO4J_" .env | sed 's/^/    /'

# ── 3. 安装/更新依赖 ──
echo ""
echo "[3/5] 检查 Python 依赖..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet 2>/dev/null

# ── 4. 清理旧进程 ──
echo ""
echo "[4/5] 清理端口 $PORT 的旧进程..."
fuser -k $PORT/tcp 2>/dev/null || true
pkill -f "streamlit run app.py --server.port $PORT" 2>/dev/null || true
sleep 2

# ── 5. 启动服务 ──
echo ""
echo "[5/5] 启动 Streamlit 服务 (端口 $PORT)..."
nohup streamlit run app.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    > server_v1.30.log 2>&1 &

sleep 3

# ── 验证 ──
if fuser $PORT/tcp >/dev/null 2>&1; then
    echo ""
    echo "=================================================="
    echo "  DONE! v1.30 部署成功"
    echo "  访问地址: http://$(hostname -I | awk '{print $1}'):$PORT"
    echo "  日志文件: server_v1.30.log"
    echo "=================================================="
else
    echo ""
    echo "  ERROR: 启动失败，请检查日志:"
    echo "  tail -f server_v1.30.log"
fi

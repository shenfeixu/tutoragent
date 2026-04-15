#!/bin/bash
# ============================================================
# TutorAgent v1.32 部署脚本 (紧急修复版)
# 部署环境: Ubuntu 22.04 (inspur-NF5468M6)
# ============================================================

echo "=================================================="
echo "  TutorAgent v1.32 - 部署开始"
echo "  [FIX] 修复 PDF 解析 Unicode 编码崩溃问题"
echo "  [NEW] 商业计划书自动合成功能支持"
echo "=================================================="

PORT=8240

# ── 1. 确认 .env ──
echo ""
echo "[1/4] 检查环境配置..."
if grep -q "localhost:8224" .env; then
    echo "  OK: Neo4j 端口配置正确"
fi

# ── 2. 安装/更新依赖 ──
echo ""
echo "[2/4] 检查 Python 依赖..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet 2>/dev/null

# ── 3. 清理旧进程 ──
echo ""
echo "[3/4] 清理旧进程..."
fuser -k $PORT/tcp 2>/dev/null || true
sleep 2

# ── 4. 启动服务 ──
echo ""
echo "[4/4] 启动服务 (v1.32)..."
nohup streamlit run app.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    > server_v1.32.log 2>&1 &

sleep 3

if fuser $PORT/tcp >/dev/null 2>&1; then
    echo ""
    echo "=================================================="
    echo "  DONE! v1.32 部署并修复完成"
    echo "  日志文件: server_v1.32.log"
    echo "=================================================="
else
    echo "  ERROR: 启动失败"
fi

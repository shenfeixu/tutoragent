#!/bin/bash
echo "=================================================="
echo "🚀 正在升级并重启 TutorAgent (v1.22 业务引擎适配版)"
echo "=================================================="

# 1. 确保虚拟环境存在
if [ -d "venv/bin" ]; then
    source venv/bin/activate
else
    python3 -m venv venv
    source venv/bin/activate
fi

# 2. 安装最新代码依赖
echo "📥 正在检查并更新依赖..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet

# 3. 后台重启服务
echo "⚙️ 正在关闭旧服务..."
fuser -k 8501/tcp 2>/dev/null || true
pkill -f "streamlit run app.py" 2>/dev/null || true
sleep 3

echo "▶️ 重新启动 8501 端口服务..."
# 启动时确保只挂起前端环境
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > server.log 2>&1 &

echo "=================================================="
echo "🎯 【v1.22 升级完成】业务防漏拦算法已激活！"
echo "=================================================="

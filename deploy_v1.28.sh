#!/bin/bash
echo "=================================================="
echo "🚀 正在部署 TutorAgent v1.28 (海量案例增强版)"
echo "=================================================="

# 1. 端口配置
PORT=8240
echo "目标端口: $PORT"

# 2. 激活虚拟环境
if [ -d "venv/bin" ]; then
    source venv/bin/activate
elif [ -d "venv/Scripts" ]; then
    # Windows 兼容性 (针对用户提到的拷贝环境)
    source venv/Scripts/activate
else
    echo "⚠️ 未发现 venv 环境，尝试从系统环境运行..."
fi

# 3. 更新依赖
echo "📥 正在检查依赖..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet

# 4. 清理旧进程
echo "⚙️ 正在清理端口 $PORT 的既有进程..."
fuser -k $PORT/tcp 2>/dev/null || true
pkill -f "streamlit run app.py --server.port $PORT" 2>/dev/null || true
sleep 2

# 5. 后台启动
echo "▶️ 正在启动服务..."
nohup streamlit run app.py --server.port $PORT --server.address 0.0.0.0 > server_v1.28.log 2>&1 &

echo "=================================================="
echo "🎯 【部署完成】服务已映射至 $PORT 端口"
echo "日志文件: server_v1.28.log"
echo "=================================================="

#!/bin/bash
# TutorAgent v1.18 自动化升级/部署脚本

echo "🚀 开始升级 TutorAgent 至 v1.18..."

# 1. 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3，请先安装 Python。"
    exit 1
fi

# 2. 拉取最新代码
echo "🔄 正在从 GitHub 拉取 v1.18 代码..."
git pull origin main
git fetch --tags
git checkout v1.18

# 3. 激活虚拟环境 (假设名为 venv)
echo "📦 正在激活虚拟环境..."
if [ -d "venv/bin" ]; then
    source venv/bin/activate
else
    echo "⚠️ 未找到虚拟环境 venv，正在尝试创建..."
    if python3 -m venv venv 2>/dev/null; then
        source venv/bin/activate
    else
        echo "❌ 虚拟环境创建失败（可能是缺少 python3-venv 模块）。"
        echo "💡 将跳过虚拟环境，改用 --user 模式直接安装依赖..."
    fi
fi

# 4. 更新依赖
echo "📥 正在检查并更新依赖 (这可能需要几分钟)..."
pip install --upgrade pip --user --quiet
pip install -r requirements.txt --user --quiet

# 5. 配置环境变量 (如果已有 .env 则跳过)
if [ ! -f .env ]; then
    echo "⚙️ 测试到没有 .env，正在创建默认配置..."
    cat > .env << EOF
# OpenAI 配置
OPENAI_API_KEY=sk-iGOC0IM8zyx5Maxj2Uw04xzQOhx9JQvFU3FooyflN49acySx
OPENAI_API_BASE=https://api.openai-proxy.org/v1
LLM_MODEL=gpt-4o-mini

# Neo4j 配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
NEO4J_DATABASE=neo4j
EOF
else
    echo "✅ .env 配置文件已存在，保持原样。"
fi

# 6. 提示完成
echo "------------------------------------------------"
echo "✅ v1.18 升级部署完成！"
echo "💡 提示: 请确保您的 Neo4j 数据库仍正常运行。"
echo "------------------------------------------------"
echo "▶️ 若无在后台运行，请使用以下指令启动服务:"
echo "   nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > server.log 2>&1 &"
echo "   (如果你在使用 screen/tmux 或 systemd，请按你原来的方式重启服务)"

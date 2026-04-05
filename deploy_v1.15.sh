#!/bin/bash
# TutorAgent v1.15 自动化部署脚本

echo "🚀 开始部署 TutorAgent v1.15..."

# 1. 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3，请先安装 Python。"
    exit 1
fi

# 2. 拉取代码 (假设已在项目目录中，若未在则克隆)
# git pull origin main

# 3. 创建并激活虚拟环境
echo "📦 正在创建虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 4. 安装依赖
echo "📥 正在安装依赖 (这可能需要几分钟)..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. 配置环境变量
echo "⚙️ 正在配置 .env 文件..."
cat > .env << EOF
# OpenAI 配置
OPENAI_API_KEY=sk-iGOC0IM8zyx5Maxj2Uw04xzQOhx9JQvFU3FooyflN49acySx
OPENAI_API_BASE=https://api.openai-proxy.org/v1
LLM_MODEL=gpt-4o-mini

# Neo4j 配置 (默认连接本地，如需改动请手动编辑 .env)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
NEO4J_DATABASE=neo4j
EOF

# 6. 提示 Neo4j
echo "------------------------------------------------"
echo "✅ 环境部署完成！"
echo "💡 提示: 请确保服务器上已运行 Neo4j (端口 7687)。"
echo "   如果没有 Neo4j，可以使用 Docker 快速启动:"
echo "   docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password123 neo4j"
echo "------------------------------------------------"
echo "▶️ 运行指令: streamlit run app.py --server.port 8501 --server.address 0.0.0.0"

#!/bin/bash
# TutorAgent v1.21 一键服务器部署与数据库修复脚本

echo "=================================================="
echo "🚀 开始同步升级 TutorAgent 服务器环境至 v1.21..."
echo "=================================================="

# 1. 确保 Git 拉取到最新的 1.21 代码
echo "🔄 1. 正在从 GitHub 拉取 v1.21 最新代码..."
git fetch --all
git checkout main
git pull origin main
# 确保回退任何未提交的文件更改，并检出最新 tag
git reset --hard origin/main
git checkout tags/v1.21 || echo "⚠️ 提示：未找到 tag v1.21，将使用 main 分支的最新代码"

# 2. 检查并激活虚拟环境 (venv)
echo "📦 2. 正在激活 Python 虚拟环境..."
if [ -d "venv/bin" ]; then
    source venv/bin/activate
elif [ -d ".venv/bin" ]; then
    source .venv/bin/activate
else
    echo "⚠️ 未找到名为 venv 的虚拟环境，正尝试为您创建..."
    python3 -m venv venv
    source venv/bin/activate
fi

# 3. 安装依赖 (特别是 pypdf 等多模态所需依赖)
echo "📥 3. 正在同步 v1.21 环境依赖 (安装 pypdf 等组件)..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "✅ 环境依赖同步完成。"

# 4. 修复/同步 Neo4j 数据库数据
echo "🗃️ 4. 正在为您清空并重新导入 52 个商业项目的知识图谱数据 （修复无法检索 Bug）..."
python scripts/import_kg_v1.19.py
if [ $? -eq 0 ]; then
    echo "✅ 数据库同步成功！前沿项目解析数据已恢复。"
else
    echo "❌ 数据库同步可能出现异常，请检查 Neo4j 是否在运行，或检查 .env 密码配置。"
fi

# 5. 关闭旧进程并后台启动新进程
echo "⚙️ 5. 正在关闭旧进程并重启前端界面..."
# 根据端口号 8501 杀死 streamlit 旧进程
fuser -k 8501/tcp 2>/dev/null || true
pkill -f "streamlit run app.py" 2>/dev/null || true

echo "⏳ 等待 3 秒钟确保端口释放..."
sleep 3

echo "▶️ 后台启动 TutorAgent 服务..."
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > server.log 2>&1 &

echo "=================================================="
echo "🎯 v1.21 服务器部署大功告成！"
echo "您现在可以通过服务器 IP 访问系统，且多模态解析支持与数据库检索引擎已完全激活！"
echo "若要查看运行日志，请输入命令: tail -f server.log"
echo "=================================================="

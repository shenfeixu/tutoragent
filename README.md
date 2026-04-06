# 超图教练 - 创新创业教学智能体

基于知识图谱与超图推理的创业项目诊断助手，支持学生端和教师端双重身份。

## 功能特性

### 🌟 v1.21 新特性：多模态全文解析引擎
- **支持长文档分析**：全量支持 PDF / Word / TXT 格式的商业计划书提取与精准评估。
- **防止超长幻觉**：AI 会在前端自动萃取文字层结构，支持后台大模型超宽上下文推演。

### 学生端
- 🎯 **智能对话** - AI 教练帮助学生分析创业想法
- 📊 **H1-H15 规则检查** - 15 项商业逻辑自动诊断
- 💬 **多轮对话** - 支持信息补充和重新评估
- 📋 **证据链追溯** - 每个判断都有据可查

### 教师端
- 📊 **班级概览** - Top 5 错误模式图表
- 👥 **学生管理** - 添加和管理学生
- 📈 **量化评分** - 五大维度自动评分
- 📚 **教学案例** - 从知识图谱获取针对性案例

## 技术栈

- **前端**: Streamlit
- **后端**: LangGraph + LangChain
- **LLM**: OpenAI API
- **数据库**: SQLite + Neo4j
- **可视化**: Plotly

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/tutoragent.git
cd tutoragent

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 配置

1. 复制环境变量模板：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入你的配置：
```env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j
```

## 运行

```bash
streamlit run app.py
```

访问 http://localhost:8501

## 项目结构

```
tutoragent/
├── app.py                    # 主应用（学生端）
├── Instructor_View.py        # 教师端页面
├── requirements.txt          # 依赖列表
├── .env.example              # 环境变量模板
├── src/
│   ├── agents/
│   │   └── langgraph_core.py # 核心逻辑（H1-H15 检查）
│   └── utils/
│       ├── database.py       # 数据库管理
│       └── session_manager.py# 会话管理
└── data/                     # 数据目录（自动创建）
    ├── tutoragent.db         # SQLite 数据库
    └── sessions/             # 会话文件
```

## H1-H15 检查规则

| 规则 | 检查内容 |
|-----|---------|
| H1 | 技术-市场匹配度 |
| H2 | 技术成熟度 |
| H3 | 目标客户定义 |
| H4 | 市场规模一致性 |
| H5 | 价值主张匹配 |
| H6 | 获客渠道 |
| H7 | 收入预测 |
| H8 | 单位经济 (LTV/CAC) |
| H9 | 团队规模 |
| H10 | 融资阶段匹配 |
| H11 | 上市时间 |
| H12 | 风险识别 |
| H13 | 技术壁垒 |
| H14 | 市场规模验证 |
| H15 | 商业模式闭环 |

## Neo4j 知识图谱结构

```cypher
CREATE (p:Project {name: '项目名称'})
CREATE (t:Tech {name: '技术描述'})
CREATE (m:Market {name: '目标市场'})
CREATE (r:Risk {name: '风险描述'})
CREATE (p)-[:USE]->(t)
CREATE (p)-[:TARGET]->(m)
CREATE (p)-[:TRIGGER_RISK]->(r)
```

## 许可证

MIT License

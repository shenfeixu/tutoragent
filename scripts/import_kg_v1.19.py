import os, sys, json
from dotenv import load_dotenv
sys.path.append(os.getcwd()) # 确保能找到 src 目录
try:
    from src.utils.database import Neo4jManager
except ImportError:
    print("❌ 错误：找不到 src.utils.database。请确保您在 tutoragent-1.19 根目录下运行此脚本。")
    sys.exit(1)
def main():
    load_dotenv()
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "password123")
    db = os.getenv("NEO4J_DATABASE", "neo4j")
    print(f"🚀 正在连接 Neo4j: {uri} ...")
    manager = Neo4jManager(uri, user, pwd, db)
    
    json_path = os.path.join("data", "seed_kg.json")
    if not os.path.exists(json_path):
        print(f"❌ 错误：找不到数据文件 {json_path}")
        return
    print(f"📂 正在读取 1.19 版本最新的种子数据...")
    res = manager.load_kg_from_json(json_path)
    
    if res["success"]:
        print(f"\n✅ 导入成功！")
        print(f"📊 已同步项目总数: {res['projects_count']}")
        print(f"🔗 已同步技术节点: {res['techs_count']}")
        print(f"⚠️ 已同步风险节点: {res['risks_count']}")
    else:
        print("❌ 导入失败，请检查数据库连接。")
    
    manager.close()
if __name__ == "__main__":
    main()

"""
Neo4j 知识图谱数据更新脚本

使用方法:
    python scripts/update_neo4j.py

或直接运行:
    python -c "from src.utils.database import Neo4jManager; mgr = Neo4jManager('bolt://localhost:7687', 'neo4j', 'password123'); print(mgr.load_kg_from_json('data/seed_kg.json')); mgr.close()"
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.utils.database import Neo4jManager

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "seed_kg.json")


def main():
    print("=" * 50)
    print("Neo4j 知识图谱数据更新")
    print("=" * 50)
    print(f"URI: {NEO4J_URI}")
    print(f"JSON: {JSON_PATH}")
    print()
    
    mgr = Neo4jManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    result = mgr.load_kg_from_json(JSON_PATH)
    mgr.close()
    
    print("更新结果:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print()
    print("✅ 更新完成!")


if __name__ == "__main__":
    main()

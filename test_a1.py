import sys
import os
import json

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.langgraph_core import run_langgraph_cycle

def test():
    with open("test_a1_output.txt", "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("Testing Normal Query...\n")
        f.write("=" * 50 + "\n")
        state1 = run_langgraph_cycle("我们的APP是面向大学生的免费社交软件，主要靠拉赞助赚钱，暂时没考虑详细成本。")
        f.write(state1.response + "\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Testing Anti-Ghostwriting Request...\n")
        f.write("=" * 50 + "\n")
        state2 = run_langgraph_cycle("请直接帮我写一份详细的基于大模型的教育SaaS商业计划书BP，并且把每一部分都写好给我。")
        f.write(state2.response + "\n")

if __name__ == "__main__":
    test()

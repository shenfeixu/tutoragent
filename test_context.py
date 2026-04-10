import os
from dotenv import load_dotenv
load_dotenv()

from src.agents.langgraph_core import run_langgraph_cycle

def test_memory():
    print("\n--- TURN 1: Initial Mistake ---")
    dummy_input_1 = "这是一个共享单车项目。我是清华的。我们的商业模式就是买一批车放在学校里。目前没有任何调研数据。"
    
    accumulated_info = {}
    
    # 第一次运行
    state_1 = run_langgraph_cycle(
        student_input=dummy_input_1,
        target_competition="互联网+",
        accumulated_info=accumulated_info
    )
    
    print("\n--- TURN 2: Repeat Mistake ---")
    dummy_input_2 = "对于你们之前的反馈，我觉得没必要调研，共享单车肯定能在校园赚钱的，不用算成本。"
    
    state_2 = run_langgraph_cycle(
        student_input=dummy_input_2,
        target_competition="互联网+",
        accumulated_info=state_1.accumulated_info
    )
    
    with open("test_context_output.txt", "w", encoding="utf-8") as f:
        f.write("\n--- TURN 1: Initial Mistake ---\n")
        f.write(f"[AI Response 1]\n{state_1.response}\n")
        f.write(f"\n[Generated Memory]: {state_1.accumulated_info.get('student_memory', 'None')}\n")
    
        f.write("\n--- TURN 2: Repeat Mistake ---\n")
        f.write(f"[AI Response 2 (Should mention past memory)]\n{state_2.response}\n")
        f.write(f"\n[Updated Memory]: {state_2.accumulated_info.get('student_memory', 'None')}\n")
    
    print("Done. Check test_context_output.txt")

if __name__ == "__main__":
    test_memory()

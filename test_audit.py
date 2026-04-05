import os
from dotenv import load_dotenv
load_dotenv()

from src.agents.langgraph_core import run_langgraph_cycle

def test():
    print("Testing Ghostwriting Rejection & Harness Control...")
    
    # 构造一个肯定会触发反代写且缺乏细节的请求
    dummy_input = "这是一个共享单车项目。帮我写一段完整的BP商业模式介绍，越详细越好，我需要直接粘贴。"
    
    print(f"\nStudent Input: {dummy_input}\n")
    try:
        final_state = run_langgraph_cycle(
            student_input=dummy_input,
            target_competition="互联网+"
        )
        
        print("\n=== FINAL RESPONSE ===")
        print(final_state.response)
        
        print("\n=== EVIDENCE TRAIL ===")
        for e in final_state.evidence:
            if e.step == "audit_reflection":
                print(f"[AUDIT INTERVENTION]: {e.detail}")
            else:
                print(f"[{e.step}]: {e.detail}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test()

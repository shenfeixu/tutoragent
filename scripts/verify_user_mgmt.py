import os
import sys
from dotenv import load_dotenv

# Ensure the root directory is in the path
sys.path.append(os.getcwd())

from src.utils.database import create_user, delete_user, get_system_stats, get_all_users

def test_user_management():
    load_dotenv()
    
    print("--- 1. 获取初始状态 ---")
    initial_stats = get_system_stats()
    initial_count = initial_stats["student_count"]
    print(f"初始学生数: {initial_count}")
    
    print("\n--- 2. 创建测试学生 'test_cleanup_user' ---")
    new_uid = create_user("test_cleanup_user", "password123", "student", "测试清理用户")
    if new_uid:
        print(f"成功创建用户，ID: {new_uid}")
    else:
        print("创建失败（可能已存在），尝试直接运行后续逻辑")
        # Find existing user if needed, but for clean test we expect a new one
        users = get_all_users()
        new_uid = next((u["id"] for u in users if u["username"] == "test_cleanup_user"), None)

    post_create_stats = get_system_stats()
    print(f"创建后学生数: {post_create_stats['student_count']}")
    
    if post_create_stats['student_count'] != initial_count + 1:
        print("❌ 统计未实时更新（增加）")
    else:
        print("✅ 统计已实时更新（增加）")

    print("\n--- 3. 删除测试学生 ---")
    if new_uid:
        success = delete_user(new_uid)
        if success:
            print("成功执行 delete_user")
        else:
            print("delete_user 执行失败")
    
    final_stats = get_system_stats()
    print(f"删除后学生数: {final_stats['student_count']}")
    
    if final_stats['student_count'] == initial_count:
        print("✅ 统计已实时更新（重置）")
    else:
        print("❌ 统计未实时更新（重置）")

if __name__ == "__main__":
    test_user_management()

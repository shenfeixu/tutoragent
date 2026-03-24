import sys

def convert():
    try:
        # Try different encodings
        content = ""
        for enc in ['utf-16', 'utf-8', 'gbk']:
            try:
                with open('创新创业智能体测试文档.txt', 'r', encoding=enc) as f:
                    content = f.read()
                if content:
                    print(f"Success with {enc}")
                    break
            except:
                continue
        
        if content:
            with open('req_cleaned.md', 'w', encoding='utf-8') as f:
                f.write(content)
            return "SUCCESS"
        return "FAILED TO READ"
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    print(convert())

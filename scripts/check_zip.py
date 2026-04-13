import zipfile
import sys

def check_zip(path):
    try:
        with zipfile.ZipFile(path, 'r') as zf:
            return zf.namelist()
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_zip.py <path>")
    else:
        print(check_zip(sys.argv[1]))

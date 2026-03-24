import zipfile
import xml.etree.ElementTree as ET
import sys

def get_docx_text(path):
    """
    Extracts text from a docx file using built-in libraries.
    """
    try:
        document = zipfile.ZipFile(path)
        xml_content = document.read('word/document.xml')
        document.close()
        
        tree = ET.fromstring(xml_content)
        
        # Word XML uses namespaces
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        text = []
        for paragraph in tree.findall('.//w:p', ns):
            texts = [node.text for node in paragraph.findall('.//w:t', ns) if node.text]
            if texts:
                text.append("".join(texts))
        
        return "\n".join(text)
    except Exception as e:
        return f"Error extracting text: {e}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_docx.py <path_to_docx>")
    else:
        print(get_docx_text(sys.argv[1]))

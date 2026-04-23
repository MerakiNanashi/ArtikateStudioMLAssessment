import os
import yaml
from pathlib import Path
from PyPDF2 import PdfReader


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(path="config.yaml"):
    config_path = os.path.join(BASE_DIR, path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_pdfs(data_path):
    subdata_path = load_config()["subdata_path"]
    data_path_comp = os.path.join(data_path, subdata_path)
    data_dir = os.path.join(BASE_DIR, data_path_comp)

    docs = []
    for pdf_file in Path(data_dir).glob("*.pdf"):
        reader = PdfReader(str(pdf_file))
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                docs.append({
                    "text": text,
                    "page": i + 1,
                    "document": pdf_file.name
                })
    return docs

if __name__ == "__main__":
    print("This is a helper module. Please import functions as needed.")
    config = load_config()
    data_path = config["data_path"]
    subdata_path = config["subdata_path"]
    data_path_comp = os.path.join(data_path, subdata_path)
    data_dir = os.path.join(BASE_DIR, data_path_comp)
    print("BASE DIR:", BASE_DIR)
    print("DATA PATH:", data_path)
    print("SUBDATA PATH:", subdata_path)
    print("RESOLVED:", data_dir)
    print("FILES FOUND:", list(Path(data_dir).glob("*.pdf")))
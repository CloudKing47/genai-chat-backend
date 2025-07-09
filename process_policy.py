import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# === STEP 1: Load text from .txt or .pdf (PDF fallback for future use) ===
def load_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        try:
            reader = PdfReader(filepath)
            if not reader.pages:
                raise ValueError("PDF file has no readable pages.")

            full_text = ""
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

            if not full_text.strip():
                raise ValueError("No readable text extracted from PDF.")
            return full_text

        except Exception as e:
            raise ValueError(f"Error reading PDF file: {e}")

    else:
        raise ValueError("Unsupported file type. Only .txt (and optionally .pdf) allowed.")

# === STEP 2: Split into chunks ===
def split_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

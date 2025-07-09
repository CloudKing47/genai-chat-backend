import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from process_policy import load_file, split_text

# === Settings ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_DIR = os.path.join(BASE_DIR, "data/policies")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.index")
CHUNK_STORE_PATH = os.path.join(BASE_DIR, "chunk_data.pkl")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ALLOWED_EXTENSIONS = {".txt", ".pdf"}

# === Helpers ===
def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

# === Step 1: Load and chunk all documents ===
print("[1] Loading and chunking policy documents...")
all_text = ""
for fname in os.listdir(POLICY_DIR):
    if allowed_file(fname):
        path = os.path.join(POLICY_DIR, fname)
        print(f"  - Adding {fname}")
        all_text += load_file(path) + "\n"

chunks = split_text(all_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# === Step 2: Generate embeddings ===
print(f"[2] Generating embeddings for {len(chunks)} chunks...")
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(chunks, show_progress_bar=True)

# === Step 3: Create and save FAISS index ===
print("[3] Creating FAISS index...")
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, FAISS_INDEX_PATH)
with open(CHUNK_STORE_PATH, "wb") as f:
    pickle.dump(chunks, f)

print(f"[✅] Done! Stored {len(chunks)} chunks in FAISS index.")

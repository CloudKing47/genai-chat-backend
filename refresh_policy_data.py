import os
import subprocess
from process_policy import load_file, split_text
from sentence_transformers import SentenceTransformer
import pickle
import faiss

# === Settings ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_DIR = os.path.join(BASE_DIR, "data/policies")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.index")
CHUNK_STORE_PATH = os.path.join(BASE_DIR, "chunk_data.pkl")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
ALLOWED_EXTENSIONS = {".txt", ".pdf"}

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

print("🔄 Refreshing company policy data...")

# Step 1: Load and chunk
print("📄 Loading and chunking policy documents...")
all_text = ""
for fname in os.listdir(POLICY_DIR):
    if allowed_file(fname):
        full_path = os.path.join(POLICY_DIR, fname)
        try:
            content = load_file(full_path)
            if content.strip():
                print(f"  - Adding {fname}")
                all_text += content + "\n"
            else:
                print(f"⚠️ Skipping empty file: {fname}")
        except Exception as e:
            print(f"❌ Failed to load {fname}: {e}")

chunks = split_text(all_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# Step 2: Embedding
print(f"🧠 Generating embeddings using model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(chunks, show_progress_bar=True)

# Step 3: Create and save FAISS index
print("💾 Creating and saving FAISS index...")
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, FAISS_INDEX_PATH)

# Save chunk data
with open(CHUNK_STORE_PATH, "wb") as f:
    pickle.dump(chunks, f)

print(f"✅ Done! Indexed {len(chunks)} chunks from your documents.\n")

# Optional: Restart Flask API in a new terminal window
subprocess.Popen(["python", "api.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)
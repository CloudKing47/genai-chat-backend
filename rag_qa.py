# rag_qa.py

import pickle
import faiss
from sentence_transformers import SentenceTransformer
import subprocess

# --- Settings ---
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
FAISS_INDEX_PATH = "faiss_index.index"
CHUNK_STORE_PATH = "chunk_data.pkl"
TOP_K = 3
OLLAMA_MODEL = "mistral"  # make sure this is downloaded

# --- Load FAISS index and chunks ---
print("[1] Loading index and chunks...")
index = faiss.read_index(FAISS_INDEX_PATH)
with open(CHUNK_STORE_PATH, "rb") as f:
    chunks = pickle.load(f)

# --- Load embedding model ---
print("[2] Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

# --- Ask user for question ---
question = input("\n🧠 Ask a question about your company policy: ")

# --- Convert question to embedding ---
q_embedding = model.encode([question])

# --- Search in FAISS index ---
D, I = index.search(q_embedding, TOP_K)
retrieved_chunks = [chunks[i] for i in I[0]]

# --- Build prompt ---
context = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])
prompt = f"""
You are a helpful assistant trained on company policy.
Based on the below context, answer the question briefly and clearly.

Context:
{context}

Question:
{question}

Answer:
"""

# --- Send prompt to Ollama LLM ---
print("\n🤖 Sending to LLM...\n")

process = subprocess.Popen(
    ["ollama", "run", OLLAMA_MODEL],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

stdout, stderr = process.communicate(input=prompt)

if stderr:
    print("⚠️ Error:", stderr)
else:
    print("✅ Answer:")
    print(stdout.strip())

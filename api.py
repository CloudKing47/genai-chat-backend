from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import re
import pickle
import faiss
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# === Load FAISS index and document chunks ===
index = faiss.read_index("faiss_index.index")
with open("chunk_data.pkl", "rb") as f:
    chunks = pickle.load(f)

# === Load sentence transformer model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Utility to clean ANSI escape codes from LLM response ===
def clean_output(text):
    return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text).strip()

# === Subprocess runner for Ollama model ===
def run_ollama_prompt(prompt):
    try:
        process = subprocess.run(
            ["ollama", "run", "gemma"],  # 👈 Change "mistral" to "gemma" if needed
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        raw_output = process.stdout.decode("utf-8", errors="ignore")
        return clean_output(raw_output)

    except subprocess.TimeoutExpired:
        return "❌ LLM took too long to respond. Try again."
    except Exception as e:
        return f"❌ Failed to connect to model: {str(e)}"

# === Caching Ollama responses for repeated prompts ===
@lru_cache(maxsize=128)
def cached_response(prompt: str) -> str:
    return run_ollama_prompt(prompt)

# === Caching question embeddings ===
@lru_cache(maxsize=256)
def cached_embedding(question: str):
    return model.encode([question])[0]

# === API route for asking questions ===
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    print("📥 Incoming data:", data)
    question = data.get("question", "")

    if not question.strip():
        return jsonify({"answer": "❗ Please provide a valid question."})

    # === Retrieve similar chunks ===
    q_vector = np.array([cached_embedding(question)])
    _, I = index.search(q_vector, 3)
    top_chunks = [chunks[i] for i in I[0]]
    context = "\n\n".join(top_chunks)
    print("📚 Context:\n", context)

    # === Build prompt ===
    prompt = f"""
You are a helpful assistant trained on internal company policy.

Answer clearly using only the information below.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    # === Get LLM answer from cached prompt ===
    answer = cached_response(prompt)
    print("🤖 Answer returned ===>", answer)

    return jsonify({ "answer": answer })


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import re
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import os
from werkzeug.utils import secure_filename
from process_policy import load_file, split_text

# === Setup ===
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "./data/policies"
ALLOWED_EXTENSIONS = {".txt", ".pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Initial Load ===
index = faiss.read_index("faiss_index.index")
with open("chunk_data.pkl", "rb") as f:
    chunks = pickle.load(f)
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Helpers ===
def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def clean_output(text):
    return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text).strip()

def run_ollama_prompt(prompt):
    try:
        process = subprocess.run(
            ["ollama", "run", "gemma"],  # or "mistral" if desired
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        raw_output = process.stdout.decode("utf-8", errors="ignore")
        return clean_output(raw_output)
    except subprocess.TimeoutExpired:
        return "LLM took too long to respond. Try again."
    except Exception as e:
        return f"Failed to connect to model: {str(e)}"

# === Question Endpoint ===
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")

    if not question.strip():
        return jsonify({"answer": "Please provide a valid question."})

    q_vector = model.encode([question])
    _, I = index.search(q_vector, 3)
    top_chunks = [chunks[i] for i in I[0]]
    context = "\n\n".join(top_chunks)

    prompt = f"""
You are a helpful assistant trained on internal company policy.

Answer clearly using only the information below.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    answer = run_ollama_prompt(prompt)
    return jsonify({ "answer": answer })

# === Upload Endpoint ===
@app.route("/upload", methods=["POST"])
def upload():
    global index, chunks  # So we can update them in memory

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Only .txt or .pdf allowed."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        print("🔄 Rebuilding FAISS index from uploaded files...")
        all_text = ""
        for fname in os.listdir(UPLOAD_FOLDER):
            if allowed_file(fname):
                full_path = os.path.join(UPLOAD_FOLDER, fname)
                try:
                    content = load_file(full_path)
                    if content.strip():
                        print(f"  ✅ Added: {fname}")
                        all_text += content + "\n"
                    else:
                        print(f"  ⚠️ Skipped empty file: {fname}")
                except Exception as e:
                    print(f"  ❌ Skipped unreadable file {fname}: {e}")

        if not all_text.strip():
            return jsonify({"error": "No valid content found in uploaded files"}), 400

        # Split into chunks
        chunks = split_text(all_text, chunk_size=500, chunk_overlap=50)

        # Embed
        embeddings = model.encode(chunks, show_progress_bar=True)
        dimension = embeddings[0].shape[0]

        # Create FAISS index
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Save to disk
        faiss.write_index(index, "faiss_index.index")
        with open("chunk_data.pkl", "wb") as f:
            pickle.dump(chunks, f)

        return jsonify({"message": f"✅ {filename} uploaded and indexed successfully!"})

    except Exception as e:
        print("Error during upload/indexing:", e)
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

# === Run Server ===
if __name__ == "__main__":
    app.run(debug=True)
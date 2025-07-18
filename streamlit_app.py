import streamlit as st
import os
import shutil
from PyPDF2 import PdfReader
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# --- CONFIG ---
# === SETTINGS: Change these to use different models or settings ===
UPLOAD_DIR = "uploads"
EMBED_DIR = "pdf_embeddings"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence Transformers embedding model
QA_MODEL_NAME = "deepset/roberta-base-squad2"  # (legacy, not used for generative)
# GEN_LLM_NAME = "microsoft/phi-2"  # Local generative LLM for full-sentence answers
GEN_LLM_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# CHUNK_SIZE = 1000  # Larger chunk size for more

MAX_GEN_TOKENS = 64
CHUNK_SIZE = 300
N_CONTEXT_CHUNKS = 3

# Example: If you need an API key for cloud LLMs, use environment variables:
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# (Set this in your .env file, never hardcode in code)
# ===============================================

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)

# Load models once
@st.cache_resource(show_spinner=False)
def load_models():
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(GEN_LLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(GEN_LLM_NAME, torch_dtype="auto")
    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    return embedder, gen_pipe

embedder, gen_pipe = load_models()

# --- PDF & EMBEDDING UTILS ---
def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def get_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def embed_texts(texts):
    return embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def save_embeddings(filename, embeddings, chunks):
    np.save(os.path.join(EMBED_DIR, f"{filename}_embeddings.npy"), embeddings)
    with open(os.path.join(EMBED_DIR, f"{filename}_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f)

def load_embeddings(filename):
    emb_path = os.path.join(EMBED_DIR, f"{filename}_embeddings.npy")
    chunks_path = os.path.join(EMBED_DIR, f"{filename}_chunks.json")
    if not os.path.exists(emb_path) or not os.path.exists(chunks_path):
        return None, None
    embeddings = np.load(emb_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return embeddings, chunks

# --- STREAMLIT UI ---
st.set_page_config(page_title="Talk to your PDF", layout="centered")
st.title("Talk to your PDF :books:")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "filename" not in st.session_state:
    st.session_state.filename = None

st.write("Upload a PDF and chat with it using AI!")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)
    # Extract text & embed
    with st.spinner("Processing PDF and creating embeddings..."):
        text = get_pdf_text(file_path)
        chunks = chunk_text(text)
        if len(chunks) == 0:
            st.error("No extractable text found in PDF.")
        else:
            embeddings = embed_texts(chunks)
            save_embeddings(uploaded_file.name, embeddings, chunks)
            st.session_state.filename = uploaded_file.name
            st.success(f"Uploaded and processed: {uploaded_file.name}")
            st.session_state.messages = []

if st.session_state.filename:
    st.write(f"**Current PDF:** {st.session_state.filename}")
    # Chat UI
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "AI"
        st.markdown(f"**{role}:** {msg['content']}")

    user_input = st.text_input("Ask a question about your PDF:", key="input")
    if st.button("Send") and user_input:
        # Retrieve context
        embeddings, chunks = load_embeddings(st.session_state.filename)
        if embeddings is None or chunks is None:
            st.error("PDF not processed or not found.")
        else:
            q_emb = embed_texts([user_input])[0]
            # Compute cosine similarities
            similarities = np.dot(embeddings, q_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8)
            top_idx = np.argsort(similarities)[-N_CONTEXT_CHUNKS:][::-1]
            context = "\n\n".join([chunks[i] for i in top_idx])
            # Build prompt for generative LLM
            prompt = (
                f"You are an assistant for question answering. "
                f"Given the following context from a PDF, answer the user's question in a detailed, helpful, and concise way. "
                f"Quote relevant sections and summarize as needed.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {user_input}\nAnswer:"
            )
            with st.spinner("AI is thinking..."):
                response = gen_pipe(prompt, max_new_tokens=MAX_GEN_TOKENS, do_sample=True, temperature=0.7)
                answer = response[0]["generated_text"][len(prompt):].strip()
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "ai", "content": answer})
                st.markdown(f"**AI:** {answer}")
else:
    st.info("Please upload a PDF to start chatting.")

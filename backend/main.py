from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
from typing import List
from pydantic import BaseModel
from PyPDF2 import PdfReader
import faiss
import numpy as np
import openai
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_pdfs"
EMBED_DIR = "pdf_embeddings"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)

# Set your OpenAI API key from environment or fallback to provided key
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable not set. Please add it to your .env file.")
openai.api_key = openai_api_key  # Do NOT hardcode this key! Set it in .env only.

CHUNK_SIZE = 500  # characters
EMBED_DIM = 1536  # OpenAI text-embedding-ada-002

class AskRequest(BaseModel):
    filename: str
    question: str

def chunk_text(text, chunk_size=CHUNK_SIZE):
    # Simple character-based chunking
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def get_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def embed_texts(texts):
    # Call OpenAI API to get embeddings for a list of texts
    resp = openai.Embedding.create(
        input=texts,
        model="text-embedding-ada-002"
    )
    return [np.array(d["embedding"], dtype=np.float32) for d in resp["data"]]

def save_faiss_index(filename, embeddings, chunks):
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(np.vstack(embeddings))
    faiss.write_index(index, os.path.join(EMBED_DIR, f"{filename}.index"))
    with open(os.path.join(EMBED_DIR, f"{filename}_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f)

def load_faiss_index(filename):
    index_path = os.path.join(EMBED_DIR, f"{filename}.index")
    chunks_path = os.path.join(EMBED_DIR, f"{filename}_chunks.json")
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return None, None
    index = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Extract text and create embeddings
    text = get_pdf_text(file_path)
    chunks = chunk_text(text)
    if len(chunks) == 0:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF.")
    embeddings = embed_texts(chunks)
    save_faiss_index(file.filename, embeddings, chunks)
    return {"filename": file.filename, "chunks": len(chunks)}

@app.post("/ask/")
async def ask_question(req: AskRequest):
    filename = req.filename
    question = req.question
    index, chunks = load_faiss_index(filename)
    if index is None or chunks is None:
        raise HTTPException(status_code=404, detail="PDF not processed or not found.")
    # Embed the question
    q_emb = embed_texts([question])[0]
    D, I = index.search(np.array([q_emb]), k=min(5, len(chunks)))
    # Get top matching chunks
    context = "\n\n".join([chunks[i] for i in I[0]])
    # Compose prompt
    prompt = (
        f"You are an assistant for question answering. "
        f"Given the following context from a PDF, answer the user's question. "
        f"Quote relevant sections and summarize as needed.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    # Call OpenAI for answer
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512
    )
    answer = response["choices"][0]["message"]["content"].strip()
    return JSONResponse({"answer": answer})

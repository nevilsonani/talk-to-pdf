# Talk to your PDF

A private, local conversational AI app: upload a PDF, ask questions in natural language, and get quoted, summarized, and context-aware answers using a local language model—no internet or API keys required.

## Features
- Upload any PDF document
- Ask questions in plain English
- Get detailed, quoted answers from your PDF
- Runs fully offline using TinyLlama (or other local LLMs)
- No OpenAI, no cloud, no data leaves your machine

## Requirements
- Python 3.9+
- At least 4–6GB RAM recommended
- OS: Windows, Mac, or Linux

## Setup & Usage

1. **Clone the repository and install requirements:**
   ```sh
   pip install -r backend/requirements.txt
   ```

2. **Run the Streamlit app:**
   ```sh
   streamlit run streamlit_app.py
   ```

   - On first run, the TinyLlama model will be downloaded (~1.1GB).
   - The app runs at [http://localhost:8501](http://localhost:8501)

3. **Upload your PDF and chat!**
   - Drag and drop a PDF.
   - Ask questions about its content.
   - Answers are generated using the local LLM and relevant PDF context.

## Configuration
- Change these at the top of `streamlit_app.py` for speed/quality tradeoff:
  ```python
  GEN_LLM_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  CHUNK_SIZE = 300
  N_CONTEXT_CHUNKS = 3
  MAX_GEN_TOKENS = 64
  ```
- Larger values = longer answers/slower; smaller = faster/shorter.
- For best results, keep total prompt length < 2048 tokens (TinyLlama's limit).

## Notes & Limitations
- All processing is local; no data leaves your device.
- First answer may be slow as the model loads; subsequent answers are faster.
- For much faster or longer answers, run on a machine with a GPU.
- TinyLlama is used by default for efficiency; swap for other HuggingFace LLMs if desired.

## Deployment
- This app is designed for local/private use.
- Cloud deployment is possible on a VM with enough RAM/disk, but not recommended on free/shared hosts due to model size and CPU needs.
- For sharing, consider Dockerizing or using a private server.

## Credits
- [Sentence Transformers](https://www.sbert.net/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

---

**Enjoy private, offline AI-powered PDF Q&A!**
1. Navigate to `backend/`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the server: `uvicorn main:app --reload`

### Frontend
1. Navigate to `frontend/`
2. Install dependencies: `npm install`
3. Run the dev server: `npm run dev`

### Usage
1. Open the frontend in your browser.
2. Upload a PDF.
3. Start chatting!

---

**Note:**
- The AI answering logic is a placeholder. You’ll need to add OpenAI API integration and PDF text retrieval for full functionality.
- Make sure both frontend and backend are running.

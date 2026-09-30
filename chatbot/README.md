# RAG Chatbot (Gemini + Streamlit)

Chat with your own documents (.txt, .md, .pdf).

- **Embeddings:** Gemini API (`gemini-embedding-001`), no heavy local model
- **LLM:** Gemini (`gemini-3.8-flash`)
- **Vector store:** NumPy files in `index/` (built automatically on first start)
- **UI:** Streamlit

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
# open .env and paste your GEMINI_API_KEY
streamlit run app.py
```

## Deploy to Render

1. Push this folder to GitHub (GitHub Desktop -> Publish repository). `.env` is git-ignored, so your key stays private.
2. On https://render.com: **New +** -> **Blueprint** (or **Web Service**) -> pick your repo.
   - If you use a manual Web Service: Build `pip install -r requirements.txt`,
     Start `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
3. Add the environment variable `GEMINI_API_KEY` with your key in the Render dashboard.
4. Deploy. First start builds the index from `docs/`.

Notes:
- Put the documents you want the bot to know in `docs/` and commit them. Files uploaded through the web UI are lost when Render restarts (free disk is temporary).
- Free Render services sleep after inactivity, so the first load can be slow.

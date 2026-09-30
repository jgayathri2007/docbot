"""Core RAG logic: load -> chunk -> embed -> retrieve -> generate."""
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path("docs")
INDEX_DIR = Path("index")
SUPPORTED = {".txt", ".md", ".pdf"}

GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def provider() -> str:
    """LLM_PROVIDER=gemini|claude overrides; otherwise use whichever key is set."""
    p = os.getenv("LLM_PROVIDER", "").lower()
    if p in ("gemini", "claude"):
        return p
    return "gemini" if GEMINI_KEY else "claude"


def embed_backend() -> str:
    """Gemini API embeddings if a Gemini key exists (light, good for deployment),
    otherwise local sentence-transformers (needs requirements-local.txt)."""
    return "gemini" if GEMINI_KEY else "local"


# ---------- embeddings ----------
_local_model = None


def _normalize(v: np.ndarray) -> np.ndarray:
    return (v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)).astype("float32")


def embed(texts, query: bool = False) -> np.ndarray:
    if embed_backend() == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_KEY)
        task = "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT"
        out = []
        for i in range(0, len(texts), 50):  # batch to stay within API limits
            r = client.models.embed_content(
                model=GEMINI_EMBED_MODEL,
                contents=texts[i:i + 50],
                config=types.EmbedContentConfig(task_type=task, output_dimensionality=768),
            )
            out.extend(e.values for e in r.embeddings)
        return _normalize(np.array(out, dtype="float32"))

    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(LOCAL_EMBED_MODEL)
    return _local_model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype("float32")


# ---------- ingestion ----------
def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int = 200, overlap: int = 40):
    """Split text into overlapping windows of `size` words."""
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]


def build_index() -> int:
    """Read everything in docs/, embed it, and save the index to disk."""
    DOCS_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)
    chunks = []
    for f in sorted(DOCS_DIR.rglob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED:
            for i, c in enumerate(chunk_text(read_file(f))):
                chunks.append({"source": f.name, "chunk": i, "text": c})
    if not chunks:
        raise RuntimeError("No documents found in docs/ (supported: .txt, .md, .pdf)")
    vectors = embed([c["text"] for c in chunks])
    np.save(INDEX_DIR / "vectors.npy", vectors)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    (INDEX_DIR / "meta.json").write_text(json.dumps({"backend": embed_backend()}), encoding="utf-8")
    return len(chunks)


def index_exists() -> bool:
    files_ok = all((INDEX_DIR / n).exists() for n in ("vectors.npy", "chunks.json", "meta.json"))
    if not files_ok:
        return False
    meta = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
    return meta.get("backend") == embed_backend()  # rebuild if embedding backend changed


# ---------- retrieval + generation ----------
def retrieve(query: str, k: int = 4):
    vectors = np.load(INDEX_DIR / "vectors.npy")
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    scores = vectors @ embed([query], query=True)[0]  # cosine similarity
    top = np.argsort(-scores)[:k]
    return [{**chunks[i], "score": float(scores[i])} for i in top]


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using the provided context. "
    "If the context doesn't contain the answer, say you don't know based on the documents. "
    "Mention the source file names you relied on."
)


def _generate_gemini(prompt: str, history) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_KEY)
    contents = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part(text=m["content"])],
        )
        for m in (history or [])
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return resp.text


def _generate_claude(prompt: str, history) -> str:
    import anthropic

    messages = list(history or []) + [{"role": "user", "content": prompt}]
    resp = anthropic.Anthropic().messages.create(
        model=CLAUDE_MODEL, max_tokens=1024, system=SYSTEM_PROMPT, messages=messages
    )
    return resp.content[0].text


def answer(query: str, history=None, k: int = 4):
    """Return (answer_text, retrieved_chunks)."""
    hits = retrieve(query, k)
    context = "\n\n".join(f"[{h['source']} #{h['chunk']}]\n{h['text']}" for h in hits)
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    gen = _generate_gemini if provider() == "gemini" else _generate_claude
    return gen(prompt, history), hits

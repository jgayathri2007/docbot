"""Streamlit web UI.  Run with:  streamlit run app.py"""
import streamlit as st

import rag

st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
st.title("💬 Chat with your documents")

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload .txt, .md or .pdf", accept_multiple_files=True,
                               type=["txt", "md", "pdf"])
    if st.button("Add & re-index") and uploads:
        rag.DOCS_DIR.mkdir(exist_ok=True)
        for u in uploads:
            (rag.DOCS_DIR / u.name).write_bytes(u.getbuffer())
        with st.spinner("Indexing..."):
            n = rag.build_index()
        st.success(f"Indexed {n} chunks.")
    if st.button("Rebuild index from docs/"):
        with st.spinner("Indexing..."):
            st.success(f"Indexed {rag.build_index()} chunks.")
    k = st.slider("Chunks to retrieve", 1, 10, 4)

if not rag.index_exists():
    with st.spinner("Building first-time index from docs/ ..."):
        rag.build_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.caption(f"{s['source']} #{s['chunk']} (score {s['score']:.2f})")
                    st.write(s["text"][:400] + "...")

if q := st.chat_input("Ask about your documents"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            text, hits = rag.answer(q, history, k=k)
        st.markdown(text)
    st.session_state.messages.append({"role": "assistant", "content": text, "sources": hits})
    st.rerun()

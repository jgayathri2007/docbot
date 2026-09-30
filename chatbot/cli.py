"""Command-line interface.  Usage:
    python cli.py ingest     # (re)build the index from docs/
    python cli.py chat       # chat in the terminal
"""
import sys

import rag


def chat():
    if not rag.index_exists():
        print("No index found, building one...")
        print(f"Indexed {rag.build_index()} chunks.")
    history = []
    print("Ask a question about your documents (Ctrl+C to quit).")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q:
            continue
        text, hits = rag.answer(q, history)
        print(f"\nBot: {text}")
        print("Sources:", ", ".join(sorted({h['source'] for h in hits})))
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": text}]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "chat"
    if cmd == "ingest":
        print(f"Indexed {rag.build_index()} chunks.")
    elif cmd == "chat":
        chat()
    else:
        print(__doc__)

"""Vector store similarity search with LangChain + Chroma + Ollama.

This script demonstrates the core RAG retrieval loop:

    documents  ->  embeddings (Ollama embed model)  ->  Chroma vector store
                                                            |
                       query text -> embedding ------> similarity search -> top-k docs

Embeddings turn text into vectors. Chroma stores those vectors and, given a
query vector, returns the documents whose vectors are "closest" (most similar).
The number of results returned is the "top-k".

Prereqs:
    - Ollama running locally (`ollama serve`) with an embedding model pulled:
        ollama pull mxbai-embed-large
"""

from dotenv import load_dotenv

load_dotenv()

import tempfile

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

# NOTE on the embedding model:
# gemma4 is a CHAT/generation model and does NOT expose an embeddings endpoint
# in Ollama (it returns "501 this model does not support embeddings"). Embeddings
# require a model trained for it, so we use "mxbai-embed-large" — a dedicated
# embedding model you already have pulled. Pull it (or any embed model) with:
#     ollama pull mxbai-embed-large
EMBEDDING_MODEL = "mxbai-embed-large"

# A small sample corpus. Each Document = page_content (the text that gets
# embedded) + metadata (anything you want to carry alongside it).
SAMPLE_DOCUMENTS = [
    Document(
        page_content="LangChain is a framework for building applications with large language models.",
        metadata={"source": "intro.txt", "topic": "langchain"},
    ),
    Document(
        page_content="Chroma is an open-source vector database used to store and query embeddings.",
        metadata={"source": "chroma.txt", "topic": "vector-store"},
    ),
    Document(
        page_content="Ollama lets you run open models like gemma locally and serve embeddings.",
        metadata={"source": "ollama.txt", "topic": "ollama"},
    ),
    Document(
        page_content="Retrieval-Augmented Generation grounds an LLM's answers in retrieved documents.",
        metadata={"source": "rag.txt", "topic": "rag"},
    ),
    Document(
        page_content="The Eiffel Tower is located in Paris, France, and was completed in 1889.",
        metadata={"source": "paris.txt", "topic": "trivia"},
    ),
]


def build_vector_store(persist_directory: str) -> Chroma:
    """Embed the sample documents and load them into a Chroma vector store."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vector_store = Chroma.from_documents(
        documents=SAMPLE_DOCUMENTS,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    print(f"Indexed {len(SAMPLE_DOCUMENTS)} documents using '{EMBEDDING_MODEL}' embeddings.\n")
    return vector_store


def similarity_search(vector_store: Chroma, query: str, k: int = 3) -> None:
    """Return the top-k documents most similar to the query (no scores)."""
    print(f"similarity_search(query={query!r}, k={k})")
    results = vector_store.similarity_search(query, k=k)
    for rank, doc in enumerate(results, start=1):
        print(f"  {rank}. [{doc.metadata.get('source')}] {doc.page_content}")
    print()


def similarity_search_with_score(vector_store: Chroma, query: str, k: int = 3) -> None:
    """Return the top-k documents along with their distance scores.

    Chroma returns a *distance* (lower = more similar). The closest documents
    appear first.
    """
    print(f"similarity_search_with_score(query={query!r}, k={k})")
    results = vector_store.similarity_search_with_score(query, k=k)
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"  {rank}. distance={score:.4f} [{doc.metadata.get('source')}] {doc.page_content}")
    print()


def main() -> None:
    # A temporary directory keeps each run isolated; point this at a real path
    # (e.g. "./vector_store/chroma_db") if you want the index to persist.
    with tempfile.TemporaryDirectory() as tmpdir:
        vector_store = build_vector_store(tmpdir)

        similarity_search(vector_store, "How do I run models locally?", k=3)
        similarity_search_with_score(vector_store, "What database stores embeddings?", k=3)
        similarity_search_with_score(vector_store, "Where is the Eiffel Tower?", k=2)


if __name__ == "__main__":
    main()

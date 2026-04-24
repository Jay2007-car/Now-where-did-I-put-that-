#!/usr/bin/env python3
"""
RAG chatbot over a local PDF (`notes.pdf`) using:
- LangChain
- OpenAI embeddings
- Chroma vector database
- GPT-4o-mini for generation

Installation:
  1) Create and activate a virtual environment (recommended)
     python -m venv .venv
     source .venv/bin/activate      # On Windows: .venv\\Scripts\\activate

  2) Install dependencies
     pip install -U \
       langchain \
       langchain-openai \
       langchain-community \
       langchain-text-splitters \
       langchain-chroma \
       pypdf \
       chromadb \
       python-dotenv

  3) Configure your OpenAI API key
     export OPENAI_API_KEY="your_api_key_here"   # On Windows PowerShell:
                                                   # $env:OPENAI_API_KEY="your_api_key_here"

Run:
  python rag_app.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------------
# Configuration constants
# -------------------------------
PDF_PATH = Path("notes.pdf")
PERSIST_DIRECTORY = Path(".chroma_notes")
COLLECTION_NAME = "notes_collection"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4


def configure_logging() -> None:
    """Set up basic structured logging for easier debugging and operations."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def validate_environment() -> None:
    """Validate required environment variables and input files before runtime."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Please export it before running this script."
        )

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF file not found: {PDF_PATH.resolve()}\n"
            "Make sure `notes.pdf` exists in the current working directory."
        )


def load_pdf_documents(pdf_path: Path) -> List[Document]:
    """Load raw documents from a PDF file using LangChain's PyPDFLoader."""
    logging.info("Loading PDF from %s", pdf_path)
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    if not documents:
        raise ValueError("No text could be loaded from the PDF. Is it empty or scanned?")

    logging.info("Loaded %d page-level document(s)", len(documents))
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into retrieval-friendly chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    logging.info("Split into %d chunk(s)", len(chunks))
    return chunks


def build_vector_store(chunks: List[Document]) -> Chroma:
    """Embed chunks with OpenAI embeddings and persist them in Chroma."""
    logging.info("Creating embeddings model: %s", EMBEDDING_MODEL)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    logging.info("Building Chroma vector store at %s", PERSIST_DIRECTORY)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIRECTORY),
    )

    return vector_store


def create_rag_chain(vector_store: Chroma):
    """Create a retrieval-augmented generation chain for Q&A."""
    # LLM for answer generation
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    # Retriever fetches the most relevant chunks for each question
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    # Prompt keeps answers grounded in retrieved context
    prompt = ChatPromptTemplate.from_template(
        """
You are a careful assistant answering questions using the provided context only.
If the answer is not present in the context, clearly say you don't know based on the document.

Context:
{context}

Question:
{input}

Answer in a concise, clear way.
""".strip()
    )

    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)
    return rag_chain


def run_chatbot(rag_chain) -> None:
    """Interactive terminal chatbot loop for user questions."""
    print("\nRAG chatbot is ready. Ask questions about notes.pdf")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_question = input("You: ").strip()

        if not user_question:
            print("Please enter a question.\n")
            continue

        if user_question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            result = rag_chain.invoke({"input": user_question})
            answer = result.get("answer", "No answer was produced.")

            print("\n" + "=" * 70)
            print("Assistant:")
            print(answer)
            print("=" * 70 + "\n")
        except Exception as exc:  # Runtime safety for API/network/transient issues
            logging.exception("Failed to answer question: %s", exc)
            print("An error occurred while generating the answer. Please try again.\n")


def main() -> None:
    """Entrypoint to build the RAG pipeline and launch the chatbot."""
    load_dotenv()  # Optional: load OPENAI_API_KEY from a local .env file
    configure_logging()

    validate_environment()
    documents = load_pdf_documents(PDF_PATH)
    chunks = split_documents(documents)
    vector_store = build_vector_store(chunks)
    rag_chain = create_rag_chain(vector_store)
    run_chatbot(rag_chain)


if __name__ == "__main__":
    main()

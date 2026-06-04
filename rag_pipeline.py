import os
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

load_dotenv()
try:
    import streamlit as st
    if st.secrets:
        for key, val in st.secrets.items():
            os.environ.setdefault(key, str(val))
except Exception:
    pass

DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_API_BASE = os.getenv("DEEPINFRA_API_BASE", "https://api.deepinfra.com/v1/openai")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
DOCS_PATH = os.getenv("DOCS_PATH", "./API Documentation Partial.pdf")
FAISS_INDEX_PATH = "./faiss_index"

SYSTEM_PROMPT = """You are a Senior Upwork API Consultant with deep expertise in the Upwork developer platform.
Your ONLY job is to answer questions based on the documentation excerpts provided to you in the user's message.

Rules you must NEVER break:
1. If the answer is clearly present in the provided documentation excerpts, answer accurately and concisely.
2. If the answer is NOT in the provided excerpts, you MUST respond with:
   "I'm sorry, but the provided documentation does not contain that information."
3. Never make up API endpoints, token values, rate limits, or any other technical detail.
4. Always be professional, precise, and developer-friendly.
5. When referencing specifics (endpoints, parameters, token lifetimes), quote them exactly as they appear in the docs.
"""

def load_and_sanity_check(pdf_path):
    print(f"\n📄 Loading document: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    full_text = " ".join(p.page_content for p in pages)
    total_chars = len(full_text)
    print(f"✅ Sanity Check Passed!")
    print(f"   Total pages loaded : {len(pages)}")
    print(f"   Total characters   : {total_chars:,}")
    print(f"   Sample (first 300 chars):\n   '{full_text[:300].strip()}'\n")
    return pages

def chunk_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"🔪 Chunking: {len(pages)} pages → {len(chunks)} chunks (size=500, overlap=50)\n")
    return chunks

def build_or_load_vectorstore(chunks=None):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    if os.path.exists(FAISS_INDEX_PATH):
        print("📦 Loading existing FAISS index...")
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("🔨 Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("✅ FAISS index saved.\n")
    return vectorstore

def retrieve_top_chunks(query, vectorstore, k=3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)

def build_prompt(query, context_docs):
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        context_parts.append(f"[Excerpt {i}]\n{doc.page_content.strip()}")
    context_block = "\n\n".join(context_parts)
    return (
        f"Use ONLY the following documentation excerpts to answer the question.\n\n"
        f"{context_block}\n\n"
        f"Question: {query}"
    )

def call_llm(user_message):
    client = OpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url=DEEPINFRA_API_BASE,
    )
    start = time.time()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=512,
        temperature=0.1,
    )
    latency = time.time() - start
    return response.choices[0].message.content.strip(), latency

_vectorstore_cache = None

def get_vectorstore():
    global _vectorstore_cache
    if _vectorstore_cache is None:
        if os.path.exists(FAISS_INDEX_PATH):
            _vectorstore_cache = build_or_load_vectorstore()
        else:
            pages = load_and_sanity_check(DOCS_PATH)
            chunks = chunk_documents(pages)
            _vectorstore_cache = build_or_load_vectorstore(chunks)
    return _vectorstore_cache

def answer_query(query):
    vectorstore = get_vectorstore()
    source_docs = retrieve_top_chunks(query, vectorstore, k=3)
    user_message = build_prompt(query, source_docs)
    answer, latency = call_llm(user_message)
    return {
        "answer": answer,
        "sources": source_docs,
        "latency": latency,
    }

if __name__ == "__main__":
    test_questions = [
        "How long is an OAuth access token valid for?",
        "Can I use a Client Credentials Grant to access a user's private contract details?",
        "What is the specific request-per-second rate limit for the Upwork API?",
    ]
    for q in test_questions:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        result = answer_query(q)
        print(f"A: {result['answer']}")
        print(f"⏱  Latency: {result['latency']:.2f}s")
        print(f"📎 Sources used: {len(result['sources'])} chunks")

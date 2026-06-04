# Technical Summary – Upwork API RAG Support Bot

## Project Overview
A Retrieval-Augmented Generation (RAG) system that answers developer questions about the Upwork API, grounded strictly in official documentation. Built with Python, LangChain, ChromaDB, and Streamlit; powered by Meta-Llama-3.1-8B-Instruct via the DeepInfra API.

---

## Difficulties Faced

- **Chunking code-heavy documentation:** API documentation contains JSON schemas, curl examples, and endpoint definitions. A plain 500-character split risks slicing a code block mid-example. Solved by using `RecursiveCharacterTextSplitter` with paragraph/newline separators so splits prefer natural boundaries, and a 50-character overlap to repeat the tail of one chunk at the head of the next—ensuring no critical snippet is silently truncated at retrieval time.

- **Hallucination prevention in open-weight LLMs:** Smaller models (8B parameters) are prone to confabulating plausible-sounding but incorrect API details (wrong token lifetimes, non-existent endpoints). Solved via a strict system prompt that explicitly forbids inventing details and mandates a fixed fallback phrase when the answer isn't in the retrieved context.

- **API latency variability:** DeepInfra inference latency can range from 1 s to 10 s+ depending on load and prompt length. A per-request timer is measured and displayed in the UI so the user always sees the real latency rather than a spinner with no feedback.

- **Local embedding model cold-start:** The first run downloads ~90 MB of model weights for `all-MiniLM-L6-v2`. Subsequent runs load from disk cache (Hugging Face `~/.cache`). ChromaDB is persisted to `./chroma_db/` so the embedding step is skipped entirely after the initial build, keeping the app fast.

- **PDF text extraction quality:** `PyPDFLoader` occasionally merges words across column breaks or drops whitespace. This degrades chunk quality and retrieval precision. Mitigated by using `RecursiveCharacterTextSplitter` (which is whitespace-aware) and verifying output via the sanity-check print.

---

## How LLMs Were Used in Development

- **Code scaffolding:** Claude (claude.ai) was used to generate the initial LangChain pipeline structure—loader → splitter → embedder → Chroma → retriever—which was then reviewed, understood, and customised line-by-line.
- **System prompt engineering:** Iterative prompt drafting with Claude to find wording that reliably triggers the hallucination-guard fallback without making the bot refuse legitimate questions.
- **Debugging:** Pasted error tracebacks into Claude to identify ChromaDB version conflicts and resolve the `langchain-huggingface` import path changes in LangChain 0.2.x.

---

## 3 Reasons I Am the Best Fit for the ProAnalyst AI Team

1. **End-to-end RAG implementation experience.** I understand every layer of the pipeline—from PDF parsing and semantic chunking to vector similarity search and LLM prompting—not just how to wire libraries together, but *why* each design choice matters (overlap for code integrity, low temperature for factual tasks, persistent vector stores for speed).

2. **Hallucination-aware system design.** I treat accuracy as a first-class requirement. The system prompt, the retrieval-before-generation flow, and the explicit fallback phrase are deliberate engineering decisions that prevent a support bot from confidently giving wrong API advice—exactly the kind of reliability a developer-facing product demands.

3. **Rapid, explainable delivery.** I can produce a working, documented, production-aware prototype quickly while keeping the code readable and maintainable. Every function in this codebase has a docstring explaining *what* it does and *why*—because code that only the original author understands is a liability, not an asset.

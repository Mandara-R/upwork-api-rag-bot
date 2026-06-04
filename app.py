import streamlit as st
from rag_pipeline import answer_query, get_vectorstore

st.set_page_config(
    page_title="Upwork API Support Bot",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Upwork API Technical Support Bot")
st.markdown(
    "Ask any question about the **Upwork API**. "
    "Answers are grounded strictly in the official documentation."
)
st.divider()

@st.cache_resource(show_spinner="📦 Loading knowledge base…")
def load_kb():
    return get_vectorstore()

vectorstore = load_kb()
st.success("✅ Knowledge base ready!", icon="📚")

st.markdown("#### 💡 Try one of these evaluation questions:")

sample_questions = [
    "How long is an OAuth access token valid for?",
    "Can I use a Client Credentials Grant to access a user's private contract details?",
    "What is the specific request-per-second rate limit for the Upwork API, and is it enforced per Key or per IP?",
]

col1, col2, col3 = st.columns(3)
for col, q in zip([col1, col2, col3], sample_questions):
    if col.button(q, use_container_width=True):
        st.session_state["prefill"] = q

prefill = st.session_state.pop("prefill", "")
query = st.text_input("Your question:", value=prefill)

ask_button = st.button("Ask", type="primary", disabled=not query.strip())

if ask_button and query.strip():
    with st.spinner("🔍 Searching documentation and generating answer…"):
        result = answer_query(query.strip())

    st.markdown("### 💬 Answer")
    st.info(result["answer"])

    st.markdown(f"⏱ **API Latency:** `{result['latency']:.2f}` seconds")

    st.markdown("### 📎 Sources (retrieved documentation chunks)")
    for i, doc in enumerate(result["sources"], 1):
        page_num = doc.metadata.get("page", "N/A")
        with st.expander(f"Excerpt {i}  —  Page {page_num}"):
            st.code(doc.page_content, language=None)

st.divider()
st.caption("Built with LangChain · FAISS · sentence-transformers · DeepInfra (Meta-Llama-3.1-8B) · Streamlit")
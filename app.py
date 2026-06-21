import streamlit as st
import json
import os
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────
FAISS_DIR    = "faiss_index"
MODEL_NAME   = "all-MiniLM-L6-v2"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
GROQ_MODEL   = "llama-3.1-8b-instant"
TOP_K        = 4

# ── Detect environment ────────────────────────────────────────────────────────
# If GROQ_API_KEY exists in Streamlit secrets or env → use Groq (cloud)
# Otherwise → use Ollama (local)
# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "NSU ECE Chatbot",
    page_icon  = "🤖",
    layout     = "centered"
)
# 👇 CSS injection (PUT HERE)
st.markdown("""
            
<style>
             /* Title */
h1 {
    text-align: center;
}
           

.subtitle {
    text-align: center;
    font-size: 18px;
    opacity: 0.8;
    margin-top: -10px;
    margin-bottom: 20px;
}
            
   /* Sidebar */
    [data-testid="stSidebar"] {
     /* background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); */
    }
    
    [data-testid="stSidebar"] h2 {
        color: #3545ab !important;
    }
            [data-testid="stSidebar"] h1 {
        color: rgb(122, 122, 243) !important;
    }
      
            

      /* Dividers */
     hr {
        border-color: rgba(0, 217, 255, 0.2) !important;
    }
            
            
     /* Links */
    a {
       /* color: var(--accent) !important; */
        text-decoration: none;
    }
    
</style>
""", unsafe_allow_html=True)

def get_mode():
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key:
            return "groq", key
    except Exception:
        pass
    env_key = os.environ.get("GROQ_API_KEY", "")
    if env_key:
        return "groq", env_key
    return "ollama", None

# ── Page Setup ────────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title = "NSU ECE Chatbot",
#     page_icon  = "🤖",
#     layout     = "centered"
# )

# ── Load FAISS + Embedding Model (cached) ─────────────────────────────────────
@st.cache_resource
def load_resources():
    faiss_path = os.path.join(FAISS_DIR, "index.faiss")
    meta_path  = os.path.join(FAISS_DIR, "metadata.json")

    index = faiss.read_index(faiss_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = SentenceTransformer(MODEL_NAME)

    return index, metadata, model

# ── Retrieve relevant chunks ──────────────────────────────────────────────────
def retrieve(query, index, metadata, model, top_k=TOP_K):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = metadata[idx]
        results.append({
            "score" : float(score),
            "url"   : chunk["url"],
            "title" : chunk["title"],
            "text"  : chunk["text"]
        })
    return results

# ── Build prompt ──────────────────────────────────────────────────────────────
def build_prompt(query, chunks):
    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"[Source {i}] {chunk['title']}\n"
        context += f"URL: {chunk['url']}\n"
        context += f"{chunk['text']}\n\n"

    prompt = f"""You are a helpful assistant for the ECE department at North South University (NSU).
Answer the user's question based ONLY on the context provided below.
If the answer is not found in the context, say "I don't have enough information about that."
Do NOT make up information. Be clear and concise.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

User Question: {query}

Answer:"""
    return prompt

# ── Call Ollama (local) ───────────────────────────────────────────────────────
def ask_ollama(prompt):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model"  : OLLAMA_MODEL,
                "prompt" : prompt,
                "stream" : False
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"].strip(), None

    except requests.exceptions.ConnectionError:
        return None, "❌ Ollama is not running. Please start it with `ollama serve`."
    except requests.exceptions.Timeout:
        return None, "❌ Ollama took too long. Please try again."
    except Exception as e:
        return None, f"❌ Ollama error: {e}"

# ── Call Groq (cloud) ─────────────────────────────────────────────────────────
def ask_groq(prompt, api_key):
    try:
        client   = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model    = GROQ_MODEL,
            messages = [{"role": "user", "content": prompt}],
            timeout  = 60
        )
        return response.choices[0].message.content.strip(), None

    except Exception as e:
        return None, f"❌ Groq error: {e}"

# ── UI ────────────────────────────────────────────────────────────────────────
def main():
    mode, api_key = get_mode()

    # Header
    st.markdown("""
    <div class="hero">

    <h1>🤖 NSU ECE Chatbot</h1>

    <div class="subtitle">
    Get instant answers about the ECE Department at North South University
    </div>

    </div>
    """, unsafe_allow_html=True)

    col1,col2,col3=st.columns([1,1,1])

    with col2:
        if mode=="groq":
            st.success("☁️ Cloud Mode")
        else:
            st.info("💻 Local Mode")

    st.divider()

    # Load resources
    with st.spinner("Loading models and index..."):
        try:
            index, metadata, model = load_resources()
        except Exception as e:
            st.error(f"❌ Failed to load resources: {e}")
            st.info("Make sure you have run embedder.py and the faiss_index/ folder exists.")
            st.stop()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role"    : "assistant",
                "content" : "Hi! 👋 I'm the NSU ECE Department assistant. Ask me about programs, faculty, research, admissions, or anything else!"
            }
        ]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 View Sources"):
                    for source in message["sources"]:
                        st.markdown(f"**{source['title']}**")
                        st.markdown(f"🔗 [{source['url']}]({source['url']})")
                        st.divider()

    # Chat input
    if query := st.chat_input("Ask a question about NSU ECE..."):

        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                chunks = retrieve(query, index, metadata, model)
                prompt = build_prompt(query, chunks)

                # Use Groq or Ollama depending on environment
                if mode == "groq":
                    answer, error = ask_groq(prompt, api_key)
                else:
                    answer, error = ask_ollama(prompt)

                if error:
                    st.error(error)
                    st.session_state.messages.append({
                        "role"   : "assistant",
                        "content": error
                    })
                else:
                    st.markdown(answer)

                    seen    = set()
                    sources = []
                    for chunk in chunks:
                        if chunk["url"] not in seen:
                            sources.append({"title": chunk["title"], "url": chunk["url"]})
                            seen.add(chunk["url"])

                    with st.expander("📚 View Sources"):
                        for source in sources:
                            st.markdown(f"**{source['title']}**")
                            st.markdown(f"🔗 [{source['url']}]({source['url']})")
                            st.divider()

                    st.session_state.messages.append({
                        "role"    : "assistant",
                        "content" : answer,
                        "sources" : sources
                    })

    # Sidebar
    with st.sidebar:
        st.markdown("## ℹ️ About This Bot")
        st.markdown("""
        This chatbot uses **RAG** (Retrieval-Augmented Generation) to answer questions about the ECE Department at North South University.
        
        **How it works:**
        1. Your question is converted to a semantic vector
        2. FAISS finds the most relevant content from our knowledge base
        3. An AI model generates a natural answer
        4. Sources are provided for verification
        """)

        st.divider()

        st.markdown("## ⚙️ System Info")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🧠 Embeddings", "all-MiniLM")
            st.metric("📦 Chunks", f"{len(metadata)}")
        with col2:
            llm_name = "Groq" if mode == "groq" else "Ollama"
            st.metric("🤖 LLM", llm_name)
            st.metric("🔎 Top Results", f"{TOP_K}")

        st.divider()

        st.markdown("## 🛠️ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = [
                    {
                        "role"    : "assistant",
                        "content" : "👋 Chat cleared! Ready for new questions."
                    }
                ]
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

        st.divider()

        st.markdown("## 📱 About NSU ECE")
        st.markdown("""
        **North South University**
        
        Department of Electrical and Computer Engineering
        
        🌐 [ece.northsouth.edu](https://ece.northsouth.edu)
        📍 Bashundhara, Dhaka, Bangladesh
        """)

        st.divider()
        st.caption("Made with ❤️ using RAG + Streamlit")

if __name__ == "__main__":
    main()

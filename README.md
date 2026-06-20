# 🤖 NSU ECE Department Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about the ECE Department at North South University using content scraped directly from their website.

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Web Scraping | `requests` + `BeautifulSoup` |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Database | `FAISS` |
| LLM (local) | `Ollama` + `llama3.2` |
| LLM (cloud) | `Groq` + `llama3-8b-8192` |
| Web App | `Streamlit` |

## 📁 Project Structure

```
project/
├── page_name.txt        # list of URLs to scrape
├── scraper.py           # Step 1: scrape website pages
├── chunker.py           # Step 2: split content into chunks
├── cleaner.py           # Step 3: clean text
├── embedder.py          # Step 4: embed chunks + build FAISS index
├── app.py               # Step 5: Streamlit chatbot app
└── requirements.txt
```

## 🚀 Setup & Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/nsu-ece-chatbot.git
cd nsu-ece-chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Build the knowledge base (run once)
```bash
python scraper.py
python chunker.py
python cleaner.py
python embedder.py
```

### 5. Run the chatbot locally (uses Ollama)
```bash
# Make sure Ollama is running
ollama serve

# In another terminal
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and connect your repo
3. Add your Groq API key in **Streamlit Secrets**:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```
4. The app auto-detects Groq and uses it instead of Ollama ✅

> Get a free Groq API key at: https://console.groq.com

## 💡 How It Works

```
User Question
     ↓
Embedded with all-MiniLM-L6-v2
     ↓
FAISS finds top 4 most relevant chunks
     ↓
Chunks + Question sent to LLM as prompt
     ↓
LLM generates answer based only on retrieved content
     ↓
Answer + Sources shown in chat UI
```

## 👤 Author
Built with ❤️ using Python, FAISS, and Streamlit.
```
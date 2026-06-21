# NSU ECE Chatbot - Complete Project Guide & Learnings

## Project Overview

**What we built:** A RAG (Retrieval-Augmented Generation) chatbot that answers questions about a specific website using:
- Web scraping to collect data
- Vector embeddings for semantic search
- FAISS for fast similarity search
- Groq/Ollama LLM for intelligent responses
- Streamlit for the web interface

**Why this architecture?** RAG is better than fine-tuning because:
- ✅ No training required (instant deployment)
- ✅ Uses latest real data from the website
- ✅ Sources are traceable (we show where answers come from)
- ✅ Works completely offline (except Groq cloud)
- ✅ Much cheaper than API-only solutions

---

## Step-by-Step Breakdown

### **Step 1: Web Scraping (`scraper.py`)**

**What we did:**
- Read URLs from a text file (one URL per line)
- Visited each page with `requests` library
- Extracted text using `BeautifulSoup`
- Removed navigation menus, footers, scripts automatically
- Saved all content to JSON and TXT files

**Code pattern:**
```python
# Read URLs → Visit pages → Extract text → Save to JSON
```

**Mistakes we avoided:**
```
❌ No delay between requests → Gets website blocked
✅ Added 1.5 second delay → Website stays happy

❌ Scraping all HTML noise → Low quality data
✅ Removed nav/footer/script tags → Clean content only

❌ Crashing on one failed page → Entire pipeline fails
✅ Skip failed pages, continue → Robust pipeline
```

**Key lesson for next projects:** Always add delays between requests and handle errors gracefully.

---

### **Step 2: Chunking (`chunker.py`)**

**What we did:**
- Split long pages into smaller pieces (200 words each)
- Added 20-word overlap between chunks (so no info is lost at boundaries)
- Kept metadata (URL, title) with each chunk

**Why chunking matters:**
- Embedding models have token limits (~512 tokens for us)
- Smaller chunks = more precise search results
- Without overlap, information at chunk boundaries gets lost

**Example:**
```
Page: "The ECE department offers BS, MS, PhD programs. 
The BS takes 4 years..."

Chunk 1: [0-200 words]: "The ECE department offers BS, MS, PhD programs..."
Chunk 2: [180-380 words]: "...programs. The BS takes 4 years..." (20-word overlap)
```

**Mistakes we avoided:**
```
❌ No overlap → Info loss at boundaries
✅ 20-word overlap → No gaps

❌ Chunks too big (500 words) → Less precise search
✅ 200-word chunks → Perfect balance
```

**Key lesson for next projects:** Experiment with chunk size and overlap based on your content. For dense academic content, smaller chunks work better.

---

### **Step 3: Text Cleaning (`cleaner.py`)**

**What we did:**
- Removed URLs, emails, phone numbers
- Cleaned HTML entities (`&nbsp;` → space)
- Fixed repeated punctuation (`!!!` → `!`)
- Removed breadcrumbs, copyright lines, special symbols
- Dropped chunks that became too short after cleaning

**12 cleaning rules applied:**
1. HTML entities decoding
2. URL removal
3. Email removal
4. Phone number removal
5. Breadcrumb removal
6. Copyright line removal
7. Special symbol removal
8. Social media noise removal
9. Repeated punctuation fixing
10. Standalone page numbers removal
11. Bracket content removal (like `[READ MORE]`)
12. Excessive whitespace fixing

**Mistakes we avoided:**
```
❌ No cleaning → Garbage data in embeddings
✅ 12-step cleaning → High quality embeddings

❌ Removing too much → Loss of important info
✅ Careful regex patterns → Keep important content
```

**Key lesson for next projects:** Create a `clean_text()` function with modular steps. You can reuse and modify it for different websites.

---

### **Step 4: Embedding & FAISS (`embedder.py`)**

**What we did:**
- Used `sentence-transformers` model (all-MiniLM-L6-v2) to convert text to vectors
- Each chunk becomes a 384-dimensional vector
- Built FAISS index (fast similarity search database)
- Saved index + metadata for later use

**Why these choices?**
```
all-MiniLM-L6-v2:
- ✅ Fast (runs on CPU)
- ✅ Free (no API key)
- ✅ Good quality (90MB model size)
- ✅ Works offline

FAISS:
- ✅ Extremely fast search
- ✅ Can handle thousands of chunks
- ✅ Memory efficient
```

**Mistakes we avoided:**
```
❌ Using expensive OpenAI embeddings
✅ Local free model works just as well

❌ Building index without saving
✅ Save to disk so app can load quickly

❌ Forgotten about normalization
✅ Normalized embeddings for cosine similarity
```

**Key lesson for next projects:** Always test embedding quality early. Try your model on a few test queries before committing all data.

---

### **Step 5: Streamlit App (`app.py`)**

**What we did:**
- Built a chat interface using Streamlit
- Integrated FAISS retrieval (find relevant chunks)
- Connected to Groq LLM (generate answers)
- Auto-detects which LLM to use (Groq or Ollama)
- Shows sources for every answer

**Key features:**
```
User Question
     ↓
Embed question (same model as chunks)
     ↓
FAISS finds top 4 most relevant chunks
     ↓
Build prompt: "Here's context: [...]. Answer: "
     ↓
Send to Groq/Ollama
     ↓
Show answer + clickable sources
```

**Mistakes we avoided:**
```
❌ Hard-coded LLM choice
✅ Auto-detect: Groq if key exists, else Ollama

❌ Loading models every query
✅ Cache with @st.cache_resource (load once)

❌ No error handling
✅ Try/catch for Ollama and Groq failures
```

**Key lesson for next projects:** Use caching for expensive operations. Chat history in `st.session_state` keeps conversation persistent.

---

## Deployment Mistakes & Solutions

### **Mistake 1: `.gitignore` created after pushing files**
```
❌ Created secrets.toml before .gitignore → almost exposed API key
✅ Always create .gitignore FIRST, then other files
```

### **Mistake 2: FAISS index not pushed to GitHub**
```
❌ Forgot faiss_index/ → app crashed on cloud with "file not found"
✅ Temporarily commented faiss_index/ in .gitignore for deployment
```

### **Mistake 3: Wrong package versions**
```
❌ numpy==1.26.0 doesn't exist
❌ faiss-cpu==1.8.0 not compatible with Python 3.14
❌ groq==0.9.0 missing proxies argument
❌ llama3-8b-8192 model decommissioned

✅ Always use available versions (check PyPI)
✅ Use >= for dependencies, pin only stable versions
✅ Test on cloud early to catch version issues
```

### **Mistake 4: Merge conflicts with GitHub**
```
❌ Edited .gitignore on GitHub website → local repo fell behind
❌ Then tried to push without pulling → merge conflict

✅ Always git pull before pushing
✅ Avoid editing files in multiple places
```

**How to avoid:** Work only locally, push when done. Don't edit on GitHub website.

---

## File Structure to Replicate

```
project/
│
├── .gitignore              ← created FIRST (security critical!)
├── .streamlit/
│   └── secrets.toml        ← never pushed to GitHub
│
├── page_name.txt           ← input: list of URLs
├── scraper.py              ← Step 1
├── chunker.py              ← Step 2
├── cleaner.py              ← Step 3
├── embedder.py             ← Step 4
├── app.py                  ← Step 5 (final app)
│
├── requirements.txt        ← all dependencies
├── README.md               ← documentation
│
└── faiss_index/
    ├── index.faiss         ← the vector database
    └── metadata.json       ← chunk metadata
```

---

## Environment Setup Checklist

### Local Development
```bash
✅ python -m venv venv
✅ venv\Scripts\activate
✅ pip install -r requirements.txt
✅ ollama pull llama3.2
✅ ollama serve (in background)
✅ streamlit run app.py
```

### Cloud Deployment (Streamlit)
```bash
✅ Create GitHub repo (public)
✅ Push all files except secrets.toml and venv
✅ Create .streamlit/secrets.toml with GROQ_API_KEY
✅ Connect to Streamlit Cloud
✅ Add GROQ_API_KEY in Streamlit Secrets UI
✅ Deploy
```

---

## Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: groq` | Missing package | `pip install -r requirements.txt` |
| `FAISS index not found` | Index not pushed to GitHub | Comment faiss_index in .gitignore, push again |
| `KeyError: '__version__'` | Old Pillow version | Remove Pillow, use newer package versions |
| `Groq error: proxies` | Old groq library | Update: `groq>=0.12.0` |
| `Model decommissioned` | Old Groq model | Use: `llama-3.1-8b-instant` |
| `numpy==1.26.0 not found` | Version typo | Check PyPI for available versions |
| `secrets.toml in git status` | File not in .gitignore | Add `.streamlit/` to .gitignore |

---

## Best Practices for Next Projects

### **Before You Code**
1. ✅ Plan the architecture (scrape → chunk → embed → retrieve → generate)
2. ✅ Choose models/tools that are free and fast
3. ✅ Create `.gitignore` FIRST (security)
4. ✅ Create virtual environment FIRST (isolation)

### **While Coding**
1. ✅ Build incrementally (test each step before moving forward)
2. ✅ Save intermediate outputs (JSON files) for debugging
3. ✅ Add progress messages (print what's happening)
4. ✅ Handle errors gracefully (don't crash on one bad input)

### **Before Deploying**
1. ✅ Test locally first (both Ollama and Groq modes)
2. ✅ Check `.gitignore` (make sure secrets are hidden)
3. ✅ Use correct package versions (test on requirements.txt)
4. ✅ Read error logs carefully (they tell you exactly what's wrong)

### **For Configuration**
```python
# Good pattern: put all config at the top
FAISS_DIR    = "faiss_index"
MODEL_NAME   = "all-MiniLM-L6-v2"
GROQ_MODEL   = "llama-3.1-8b-instant"
CHUNK_SIZE   = 200
OVERLAP      = 20

# Bad pattern: hardcode values throughout code
```

### **For Secrets**
```python
# NEVER do this:
GROQ_API_KEY = "gsk_xxxxx"  # ❌ Exposes in GitHub!

# Always do this:
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")  # ✅ Safe
```

---

## What Worked Really Well

1. **RAG Architecture** — Much simpler than fine-tuning, instant deployment
2. **FAISS for retrieval** — Blazingly fast, even with 1000+ chunks
3. **Groq for cloud** — Free, fast, no credit card needed
4. **Streamlit for UI** — Built beautiful chat interface in minutes
5. **Incremental pipeline** — Each script (scraper → chunker → ...) was independent, easy to debug

---

## What We'd Do Differently Next Time

1. **Don't use `==` for all versions** → Use `>=` to avoid version conflicts
2. **Test cloud deployment earlier** → Don't wait until the end
3. **Check model availability** → Verify Groq models don't get deprecated
4. **Use a config file** → Instead of hardcoding FAISS_DIR, MODEL_NAME, etc.
5. **Add logging** → Track what the pipeline is doing (especially for large datasets)

---

## For Your Next RAG Project

**Copy this template:**
1. Create `.gitignore` and `.streamlit/secrets.toml`
2. Create venv and requirements.txt
3. Build scraper.py (adapt URL reading for your source)
4. Build chunker.py (adjust chunk size for your content)
5. Build cleaner.py (customize cleaning rules for your domain)
6. Build embedder.py (same code, works for any content)
7. Build app.py (same code, works for any FAISS index)
8. Test locally, push to GitHub, deploy to Streamlit

**The beauty:** Steps 6-7 don't need changes. Only 1-5 are project-specific.

---

## Key Takeaways

✅ **RAG > Fine-tuning** for knowledge bases  
✅ **Free tools** (Ollama + FAISS) work as well as paid APIs  
✅ **Local testing first** → cloud deployment later  
✅ **Incremental development** → test each step  
✅ **Proper `.gitignore`** → never expose secrets  
✅ **Version management** → read error logs carefully  
✅ **Documentation** → helps future you and others  

---

## Quick Reference Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run pipeline
python scraper.py
python chunker.py
python cleaner.py
python embedder.py

# Local Ollama
ollama pull llama3.2
ollama serve

# Streamlit app
streamlit run app.py

# Git workflow
git init
git add .
git commit -m "message"
git remote add origin URL
git push -u origin main
git pull origin main
```

---

**Congratulations on completing this project!** 🎉

You now understand:
- Web scraping & data cleaning
- Vector embeddings & similarity search
- LLM integration (both local & cloud)
- Web app development with Streamlit
- Git & GitHub workflows
- Cloud deployment

These skills apply to many other projects. Keep building! 🚀
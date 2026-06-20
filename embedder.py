import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────────────────────
JSON_IN      = "cleaned_chunks.json"   # output from cleaner.py
FAISS_DIR    = "faiss_index"           # folder to save FAISS index + metadata
MODEL_NAME   = "all-MiniLM-L6-v2"     # free, fast, works offline after 1st download
BATCH_SIZE   = 32                      # chunks to embed at once


def load_chunks(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} cleaned chunks from '{filepath}'")
    return data


def embed_chunks(chunks, model):
    """Generate embeddings for all chunks in batches."""
    texts      = [chunk["text"] for chunk in chunks]
    total      = len(texts)

    print(f"\n🔢 Embedding {total} chunks in batches of {BATCH_SIZE}...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True   # normalize for cosine similarity
    )
    print(f"✅ Embeddings shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings):
    """Build a FAISS index from embeddings."""
    dim   = embeddings.shape[1]           # embedding dimension (384 for MiniLM)
    index = faiss.IndexFlatIP(dim)        # Inner Product = cosine similarity (since normalized)
    index.add(embeddings.astype(np.float32))
    print(f"✅ FAISS index built — {index.ntotal} vectors, dimension {dim}")
    return index


def save_index(index, chunks, output_dir):
    """Save FAISS index and metadata to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save FAISS index
    faiss_path = os.path.join(output_dir, "index.faiss")
    faiss.write_index(index, faiss_path)
    print(f"💾 FAISS index saved → '{faiss_path}'")

    # Save metadata (url, title, text for each chunk)
    metadata = [
        {
            "chunk_id"    : chunk["chunk_id"],
            "url"         : chunk["url"],
            "title"       : chunk["title"],
            "chunk_index" : chunk["chunk_index"],
            "text"        : chunk["text"]
        }
        for chunk in chunks
    ]
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"💾 Metadata saved    → '{meta_path}'")


def test_search(index, chunks, model, query="What programs does ECE offer?", top_k=3):
    """Do a quick test search to verify everything works."""
    print(f"\n🔍 Test search: '{query}'")

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = index.search(query_embedding, top_k)

    print(f"\n📌 Top {top_k} results:")
    print("-" * 60)
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        chunk = chunks[idx]
        print(f"Rank  : {rank}")
        print(f"Score : {score:.4f}")
        print(f"Title : {chunk['title']}")
        print(f"URL   : {chunk['url']}")
        print(f"Text  : {chunk['text'][:200]}...")
        print("-" * 60)


def main():
    print("🧠 Step 3: Embedding + Building FAISS Index...\n")

    # Load cleaned chunks
    chunks = load_chunks(JSON_IN)

    # Load embedding model (downloads once, ~90MB)
    print(f"\n📦 Loading embedding model '{MODEL_NAME}'...")
    print("   (First run downloads ~90MB — please wait)")
    model = SentenceTransformer(MODEL_NAME)
    print(f"✅ Model loaded!")

    # Generate embeddings
    embeddings = embed_chunks(chunks, model)

    # Build FAISS index
    print(f"\n🗂️  Building FAISS index...")
    index = build_faiss_index(embeddings)

    # Save everything
    print(f"\n💾 Saving index and metadata...")
    save_index(index, chunks, FAISS_DIR)

    # Quick test
    test_search(index, chunks, model)

    print(f"\n🎉 Done! FAISS index is ready in '{FAISS_DIR}/' folder.")
    print(f"   Next step → build app.py (the chatbot)!")


if __name__ == "__main__":
    main()
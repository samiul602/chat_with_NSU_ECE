import json

# ── Config ───────────────────────────────────────────────────────────────────
JSON_IN    = "scraped_data.json"   # output from scraper.py
JSON_OUT   = "chunked_data.json"   # output of this script
CHUNK_SIZE = 200                   # words per chunk
OVERLAP    = 20                    # overlapping words between chunks


def load_scraped_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} pages from '{filepath}'")
    return data


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping word-based chunks."""
    words  = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end   = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # move forward with overlap

    return chunks


def chunk_all_pages(data):
    """Process all pages and return list of chunks with metadata."""
    all_chunks = []
    total_chunks = 0

    for page in data:
        url     = page["url"]
        title   = page["title"]
        content = page["content"]

        chunks = split_into_chunks(content)
        total_chunks += len(chunks)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id" : f"{url}__chunk_{i}",   # unique ID
                "url"      : url,
                "title"    : title,
                "chunk_index" : i,
                "total_chunks": len(chunks),
                "text"     : chunk
            })

        print(f"  📄 '{title[:50]}' → {len(chunks)} chunks")

    print(f"\n✅ Total chunks created: {total_chunks}")
    return all_chunks


def save_chunks(chunks, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"💾 Chunks saved → '{filepath}'")


def main():
    print("🔪 Step 2: Chunking scraped content...\n")

    data   = load_scraped_data(JSON_IN)
    chunks = chunk_all_pages(data)
    save_chunks(chunks, JSON_OUT)

    print(f"\n🎉 Done! '{JSON_OUT}' is ready for Step 3 (embedding).")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Chunk size  : {CHUNK_SIZE} words (overlap: {OVERLAP} words)")


if __name__ == "__main__":
    main()
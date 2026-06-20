import json
import re

# ── Config ───────────────────────────────────────────────────────────────────
JSON_IN      = "chunked_data.json"    # output from chunker.py
JSON_OUT     = "cleaned_chunks.json"  # output of this script
MIN_WORDS    = 20                     # drop chunks shorter than this


def load_chunks(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} chunks from '{filepath}'")
    return data


def clean_text(text):
    """Apply all cleaning steps to a chunk of text."""

    # 1. Decode common HTML entities
    html_entities = {
        "&amp;"  : "&",
        "&nbsp;" : " ",
        "&lt;"   : "<",
        "&gt;"   : ">",
        "&quot;" : '"',
        "&#39;"  : "'",
        "&mdash;": "—",
        "&ndash;": "–",
        "&copy;" : "",
        "&reg;"  : "",
        "&trade;": "",
    }
    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)

    # 2. Remove URLs (http, https, www)
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # 3. Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # 4. Remove phone numbers (various formats)
    text = re.sub(r'[\+\(]?[0-9][0-9\s\-\(\)]{7,}[0-9]', '', text)

    # 5. Remove breadcrumb navigation (Home > About > Page)
    text = re.sub(r'\b\w+(\s*>\s*\w+)+', '', text)

    # 6. Remove copyright lines
    text = re.sub(r'©.*?(reserved|university|nsu).*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'copyright.*?\d{4}.*', '', text, flags=re.IGNORECASE)

    # 7. Remove special/unicode symbols but keep basic punctuation
    text = re.sub(r'[©®™•·→←↑↓★☆♦◆▪▸●]', '', text)

    # 8. Remove social media noise
    text = re.sub(r'(follow us|share this|tweet|retweet|like us|subscribe).*', 
                  '', text, flags=re.IGNORECASE)

    # 9. Remove repeated punctuation (!!!, ..., ---)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[-]{2,}', '-', text)
    text = re.sub(r'[|]{1,}', ' ', text)

    # 10. Remove standalone numbers/codes on their own line (like page numbers)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # 11. Remove bracket content like [READ MORE], [DOWNLOAD], [Click here]
    text = re.sub(r'\[.*?\]', '', text)

    # 12. Remove excessive whitespace and blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)       # max 2 newlines
    text = re.sub(r'[ \t]{2,}', ' ', text)        # multiple spaces → one
    text = re.sub(r' +\n', '\n', text)             # trailing spaces
    text = text.strip()

    return text


def process_chunks(chunks):
    """Clean all chunks and filter out short/empty ones."""
    cleaned   = []
    dropped   = 0

    for chunk in chunks:
        original_text = chunk["text"]
        cleaned_text  = clean_text(original_text)

        word_count = len(cleaned_text.split())

        # Drop chunks that are too short after cleaning
        if word_count < MIN_WORDS:
            dropped += 1
            continue

        chunk["text"]       = cleaned_text
        chunk["word_count"] = word_count
        cleaned.append(chunk)

    print(f"  ✅ Cleaned  : {len(cleaned)} chunks kept")
    print(f"  🗑️  Dropped  : {dropped} chunks (too short after cleaning)")
    return cleaned


def save_cleaned(chunks, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"💾 Cleaned chunks saved → '{filepath}'")


def preview_sample(chunks, n=2):
    """Show a sample of cleaned chunks for verification."""
    print(f"\n👀 Preview of {n} cleaned chunks:")
    print("-" * 60)
    for chunk in chunks[:n]:
        print(f"URL   : {chunk['url']}")
        print(f"Title : {chunk['title']}")
        print(f"Words : {chunk['word_count']}")
        print(f"Text  : {chunk['text'][:300]}...")
        print("-" * 60)


def main():
    print("🧹 Cleaning chunked text...\n")

    chunks  = load_chunks(JSON_IN)

    print(f"\n🔧 Applying cleaning rules...")
    cleaned = process_chunks(chunks)

    save_cleaned(cleaned, JSON_OUT)
    preview_sample(cleaned)

    print(f"\n🎉 Done! '{JSON_OUT}' is ready for Step 3 (embedding).")
    print(f"   Chunks before cleaning : {len(chunks)}")
    print(f"   Chunks after  cleaning : {len(cleaned)}")


if __name__ == "__main__":
    main()
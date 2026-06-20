import requests
from bs4 import BeautifulSoup
import json
import time
import os

# ── Config ──────────────────────────────────────────────────────────────────
URL_FILE   = "page_name.txt"   # your URL list
JSON_OUT   = "scraped_data.json"
TXT_OUT    = "scraped_data.txt"
DELAY      = 1.5               # seconds between requests (be polite!)
HEADERS    = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Tags to remove (noise: nav, footer, scripts, etc.)
NOISE_TAGS = ["nav", "footer", "header", "script", "style",
              "aside", "form", "button", "noscript", "iframe"]


def load_urls(filepath):
    """Read URLs from text file (one per line)."""
    with open(filepath, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    print(f"✅ Loaded {len(urls)} URLs from '{filepath}'")
    return urls


def clean_text(text):
    """Remove excessive whitespace from extracted text."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]          # drop empty lines
    return "\n".join(lines)


def scrape_page(url):
    """Scrape a single URL and return title + clean content."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  ⚠️  Failed: {url} → {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove noise tags
    for tag in NOISE_TAGS:
        for element in soup.find_all(tag):
            element.decompose()

    # Get page title
    title = soup.title.string.strip() if soup.title else "No Title"

    # Extract main content — try <main> first, then <body>
    main = soup.find("main") or soup.find("div", {"id": "main"}) \
                              or soup.find("div", {"class": "content"}) \
                              or soup.body

    content = clean_text(main.get_text(separator="\n")) if main else ""

    if not content:
        print(f"  ⚠️  Empty content: {url}")
        return None

    print(f"  ✅ Scraped: {title} ({len(content)} chars)")
    return {"url": url, "title": title, "content": content}


def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 JSON saved → '{filepath}'")


def save_txt(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write("=" * 70 + "\n")
            f.write(f"URL   : {item['url']}\n")
            f.write(f"TITLE : {item['title']}\n")
            f.write("=" * 70 + "\n")
            f.write(item["content"] + "\n\n")
    print(f"📄 TXT  saved → '{filepath}'")


def main():
    urls    = load_urls(URL_FILE)
    results = []

    print(f"\n🔍 Starting scrape of {len(urls)} pages...\n")

    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url}")
        data = scrape_page(url)
        if data:
            results.append(data)
        time.sleep(DELAY)   # polite delay between requests

    print(f"\n✅ Successfully scraped {len(results)}/{len(urls)} pages.")

    save_json(results, JSON_OUT)
    save_txt(results, TXT_OUT)

    print("\n🎉 Done! Files ready for Step 2 (chunking + embedding).")


if __name__ == "__main__":
    main()
"""
build_corpus_v2.py — Build the expanded v2 RAG corpus (fix/v2 W3b).

Corpus = v1 Tier-1 (4 guideline PDFs) + Tier-1 v2 (2 wearable-relevant
guidelines: EHRA 2022 digital-devices guide, 2023 ACC/AHA AF guideline)
+ v1 Tier-2 (200 OA articles). Chunking identical to v1 (500 words, 50 overlap).

Writes chroma_db_v2 (collection medical_corpus_v2) + outputs_v2/corpus_stats_v2.json.
The v1 chroma_db is left untouched.
"""

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIER1_DIR = ROOT / "Dataset" / "RAG corpus (medical literatureguidelines)"
TIER1_V2_DIR = ROOT / "Dataset" / "Tier1_v2"
TIER2_DIR = ROOT / "Dataset" / "Tier2_literature"
CHROMA_V2 = ROOT / "chroma_db_v2"
OUT = ROOT / "outputs_v2"

CHUNK_WORDS = 500
OVERLAP_WORDS = 50


def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    if len(words) <= chunk_words:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_words]))
        start += chunk_words - overlap
    return chunks


def load_docs():
    import fitz
    docs = []
    for pdf_path in sorted(TIER1_DIR.glob("*.pdf")):
        d = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in d)
        d.close()
        docs.append({"source": pdf_path.stem, "tier": "tier1", "tier1_v": "v1", "text": text})
    for txt_path in sorted(TIER1_V2_DIR.glob("*.txt")):
        docs.append({"source": txt_path.stem, "tier": "tier1", "tier1_v": "v2",
                     "text": txt_path.read_text(encoding="utf-8")})
    for bucket_dir in sorted(TIER2_DIR.iterdir()):
        if not bucket_dir.is_dir():
            continue
        for md_path in sorted(bucket_dir.glob("*.md")):
            docs.append({"source": md_path.stem, "tier": "tier2", "bucket": bucket_dir.name,
                         "text": md_path.read_text(encoding="utf-8")})
    return docs


def main():
    import chromadb
    from sentence_transformers import SentenceTransformer

    OUT.mkdir(exist_ok=True)
    docs = load_docs()
    n_t1v1 = sum(1 for d in docs if d.get("tier1_v") == "v1")
    n_t1v2 = sum(1 for d in docs if d.get("tier1_v") == "v2")
    n_t2 = sum(1 for d in docs if d["tier"] == "tier2")
    print(f"docs: tier1_v1={n_t1v1} tier1_v2={n_t1v2} tier2={n_t2}", flush=True)

    all_chunks = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            all_chunks.append({"text": chunk, "source": doc["source"], "tier": doc["tier"],
                               "tier1_v": doc.get("tier1_v", ""), "bucket": doc.get("bucket", ""),
                               "chunk_idx": i})
    print(f"chunks: {len(all_chunks):,} "
          f"(mean {sum(len(c['text'].split()) for c in all_chunks)/len(all_chunks):.0f} words)", flush=True)

    client = chromadb.PersistentClient(path=str(CHROMA_V2))
    col = client.get_or_create_collection("medical_corpus_v2")
    if col.count() == 0:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        t0 = time.time()
        texts = [c["text"] for c in all_chunks]
        metas = [{k: v for k, v in c.items() if k != "text"} for c in all_chunks]
        embs = []
        B = 64
        for i in range(0, len(texts), B):
            embs.extend(embedder.encode(texts[i:i + B], show_progress_bar=False).tolist())
            if (i // B) % 20 == 0:
                print(f"  embedded {i + B}/{len(texts)}", flush=True)
        col.add(embeddings=embs, documents=texts, metadatas=metas,
                ids=[f"chunk_{i}" for i in range(len(texts))])
        print(f"embedded {len(texts):,} chunks in {time.time()-t0:.0f}s", flush=True)
    print(f"collection size: {col.count()}", flush=True)

    stats = {
        "docs_total": len(docs),
        "tier1_v1_docs": n_t1v1, "tier1_v2_docs": n_t1v2, "tier2_docs": n_t2,
        "chunks_total": len(all_chunks),
        "chunks_tier1_v1": sum(1 for c in all_chunks if c.get("tier1_v") == "v1"),
        "chunks_tier1_v2": sum(1 for c in all_chunks if c.get("tier1_v") == "v2"),
        "chunks_tier2": sum(1 for c in all_chunks if c["tier"] == "tier2"),
        "mean_chunk_words": round(sum(len(c["text"].split()) for c in all_chunks) / len(all_chunks), 1),
        "tier1_v2_documents": [
            {"source": d["source"], "words": len(d["text"].split())}
            for d in docs if d.get("tier1_v") == "v2"],
    }
    import csv as _csv
    with open(TIER1_V2_DIR / "manifest.csv", encoding="utf-8") as f:
        stats["tier1_v2_manifest"] = list(_csv.DictReader(f))
    (OUT / "corpus_stats_v2.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()

import os
import re
import json
import hashlib
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# === UTILS ===
def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


# === MAIN FUNCTION ===
def create_faiss_store(
    md_dir: str = "./personal_data",
    chunk_size: int = 1000,
    chunk_overlap: int = 250,
    persist_dir: str = "./backend/data/faiss_store",
    chunk_save_path: str = "./backend/data/all_chunks.json",
    min_chunk_chars: int = 50,
):
    """
    Reads all .md files in md_dir, splits into chunks, saves chunks to JSON,
    and builds a FAISS index with HuggingFace embeddings.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n# ", "\n## ", "\n### ", "\n#### ", "\n\n", "\n- ", "\n", ". ", " "],
        keep_separator=True,
        length_function=len,  # consider tokenizer-based later
        is_separator_regex=False,
    )

    docs, all_chunks, failed_chunks = [], [], []

    # Gather markdown files
    md_files = list(Path(md_dir).glob("*.md"))
    if not md_files:
        print(f"⚠️ No markdown files found in: {md_dir}")
    for md_file in md_files:
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except Exception as e:
            print(f"❌ Failed to read {md_file}: {e}")
            continue

        if not content:
            continue

        # NON-DESTRUCTIVE: only insert a space after hashes when missing
        # Keeps heading level (##, ###, etc.) and full text
        content = re.sub(r'\n(#+)(\S)', r'\n\1 \2', content)

        docs.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "header": content.split("\n")[0] if "\n" in content else content,
                },
            }
        )

    # Split into chunks and keep them (no LLM enrichment)
    for doc in docs:
        try:
            chunks = splitter.split_text(doc["content"])
        except Exception as e:
            print(f"❌ Error splitting {doc['metadata']['source']}: {e}")
            continue

        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk) < min_chunk_chars:
                continue

            chunk_id = f"{doc['metadata']['source']}_#{i}_{hash_text(chunk)}"
            metadata = {
                **doc["metadata"],
                "chunk_id": chunk_id,
                "has_header": chunk.startswith("#"),
                "word_count": len(chunk.split()),
            }
            header = doc["metadata"]["header"]
            chunk = f"[HEADER] {header}\n\n{chunk}" 
            # Keep raw chunk (no summaries / questions)
            all_chunks.append({"text": chunk, "metadata": metadata})

    print(f"✅ Markdown files processed: {len(docs)}")
    print(f"✅ Chunks created: {len(all_chunks)} | ⚠️ Failed: {len(failed_chunks)}")

    # Ensure output dir exists and save raw chunks JSON
    os.makedirs(os.path.dirname(chunk_save_path), exist_ok=True)
    with open(chunk_save_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"📁 Saved chunks → {chunk_save_path}")

    # If nothing to index, stop here
    if not all_chunks:
        print("⚠️ No chunks to index. Skipping FAISS build.")
        return

    # Prepare FAISS save path
    os.makedirs(persist_dir, exist_ok=True)
    version_tag = f"v{len(all_chunks)}_{chunk_size}-{chunk_overlap}"
    save_path = os.path.join(persist_dir, version_tag)
    os.makedirs(save_path, exist_ok=True)

    # Embeddings + FAISS
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = FAISS.from_texts(
        texts=[c["text"] for c in all_chunks],
        embedding=embeddings,
        metadatas=[c["metadata"] for c in all_chunks],
    )
    vector_store.save_local(save_path)

    print(f"✅ FAISS index saved at: {save_path}")
    avg_len = sum(len(c["text"]) for c in all_chunks) / len(all_chunks)
    print(f"📊 Stats → Chunks: {len(all_chunks)} | Avg length: {avg_len:.1f} characters")

    if failed_chunks:
        with open("failed_chunks.txt", "w", encoding="utf-8") as f:
            for line in failed_chunks:
                f.write(line + "\n")
        print("📝 Failed chunk IDs saved to failed_chunks.txt")


if __name__ == "__main__":
    create_faiss_store(
        md_dir="./personal_data",
        chunk_size=1000,
        chunk_overlap=250,
        persist_dir="./backend/data/faiss_store",
        chunk_save_path="./backend/data/all_chunks.json",
    )
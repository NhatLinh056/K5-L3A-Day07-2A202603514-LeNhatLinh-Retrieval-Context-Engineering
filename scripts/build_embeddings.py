"""Build and cache Day 6 dense embeddings for the 971 chunks.
Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384-dim)
Output: data/chunks_embeddings.npz
"""
import json
import time
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    chunks_path = data_dir / "chunks.json"
    output_path = data_dir / "chunks_embeddings.npz"
    
    print(f"Loading chunks from: {chunks_path}")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"Total chunks: {len(chunks)}")
    
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"Embedding dimension: {model.get_embedding_dimension()}")
    
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["id"] for c in chunks]
    
    print("Encoding 971 chunks...")
    start_time = time.perf_counter()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    elapsed = time.perf_counter() - start_time
    print(f"Encoding completed in {elapsed:.2f} seconds. Shape: {embeddings.shape}")
    
    np.savez_compressed(
        output_path,
        chunk_ids=np.array(chunk_ids),
        embeddings=embeddings.astype(np.float32)
    )
    print(f"Saved compressed embeddings to: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

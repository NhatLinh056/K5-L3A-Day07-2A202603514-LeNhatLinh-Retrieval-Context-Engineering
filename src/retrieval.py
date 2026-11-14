"""Implementation of the 3 Retrieval Strategies for Day 7:
1. Lexical Retrieval (TF-IDF)
2. Dense Retrieval (Day 6 sentence-transformers embeddings + Cosine Similarity + Metadata filter)
3. Hybrid Retrieval (Reciprocal Rank Fusion - RRF combining Dense and Lexical)
"""
import json
import math
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np


def tokenize(text: str) -> list[str]:
    """Tokenize normalized text into words."""
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


def hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    """Check if chunk matches any gold evidence in source_id, page, and evidence substring."""
    return any(
        chunk['source_id'] == g['source_id']
        and chunk['page'] == g['page']
        and g['evidence'] in chunk['text']
        for g in gold
    )


def pack_context(results: list[dict[str, Any]], budget: int = 800) -> tuple[str, list[str]]:
    """Pack candidate chunks into context string within character budget."""
    blocks, ids, seen = [], [], set()
    for c in results:
        if c['id'] in seen:
            continue
        seen.add(c['id'])
        block = f"[{c['id']} | {c['source']} | page {c['page']}]\n{c['text']}"
        if len('\n\n'.join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(c['id'])
    return '\n\n'.join(blocks), ids


class TfidfRetriever:
    """Method 1: Lexical retrieval using TF-IDF."""
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        counts = [Counter(tokenize(c['text'])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {term: math.log((1 + len(chunks)) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts: Counter) -> dict[str, float]:
        vector = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        return {term: v / norm for term, v in vector.items()}

    def search(self, query: str, top_k: int = 3, filters: dict[str, Any] | None = None,
               query_vector: Any = None, **kwargs) -> list[dict[str, Any]]:
        query_vec = self._vector(Counter(tokenize(query)))
        results = []
        for chunk, doc_vec in zip(self.chunks, self.vectors):
            if filters and filters.get('chapter') and chunk['page'] != filters['chapter']:
                continue
            score = sum(v * doc_vec.get(term, 0.0) for term, v in query_vec.items())
            if score > 0:
                results.append({**chunk, 'score': float(score), 'strategy': 'tfidf'})
        return sorted(results, key=lambda c: (-c['score'], c['id']))[:top_k]


class DenseRetriever:
    """Method 2: Dense retrieval using Day 6 precomputed embeddings and query encoder."""
    def __init__(self, chunks: list[dict[str, Any]], embeddings_path: Path, model_name: str) -> None:
        self.chunks = {c['id']: c for c in chunks}
        self.chunk_list = chunks
        data = np.load(embeddings_path)
        self.chunk_ids = data['chunk_ids'].tolist()
        self.embeddings = data['embeddings']  # shape: (N, 384), pre-normalized
        self.id_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode(query, normalize_embeddings=True)
        return vec.astype(np.float32)

    def search(self, query: str, top_k: int = 3, filters: dict[str, Any] | None = None,
               query_vector: np.ndarray | None = None) -> list[dict[str, Any]]:
        if query_vector is None:
            query_vector = self.embed_query(query)
            
        # Cosine similarity is inner product of normalized vectors
        scores = np.dot(self.embeddings, query_vector)
        
        # Apply metadata filter if specified
        candidates = []
        for idx, cid in enumerate(self.chunk_ids):
            chunk = self.chunks[cid]
            if filters and filters.get('chapter') and chunk['page'] != filters['chapter']:
                continue
            candidates.append((cid, float(scores[idx])))
            
        # Sort descending by score
        candidates.sort(key=lambda x: (-x[1], x[0]))
        top_candidates = candidates[:top_k]
        
        return [{**self.chunks[cid], 'score': score, 'strategy': 'dense'} for cid, score in top_candidates]


class HybridRrfRetriever:
    """Method 3: Hybrid retrieval combining Dense and TF-IDF with Reciprocal Rank Fusion (RRF)."""
    def __init__(self, dense_retriever: DenseRetriever, tfidf_retriever: TfidfRetriever, rrf_k: int = 60) -> None:
        self.dense = dense_retriever
        self.tfidf = tfidf_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 3, filters: dict[str, Any] | None = None,
               query_vector: np.ndarray | None = None) -> list[dict[str, Any]]:
        candidate_k = max(20, top_k * 3)
        dense_results = self.dense.search(query, top_k=candidate_k, filters=filters, query_vector=query_vector)
        tfidf_results = self.tfidf.search(query, top_k=candidate_k, filters=filters)
        
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}
        
        for rank, c in enumerate(dense_results, 1):
            cid = c['id']
            chunk_map[cid] = c
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            
        for rank, c in enumerate(tfidf_results, 1):
            cid = c['id']
            chunk_map[cid] = c
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: (-rrf_scores[cid], cid))
        results = []
        for cid in sorted_cids[:top_k]:
            c = chunk_map[cid]
            results.append({
                'id': c['id'],
                'source_id': c['source_id'],
                'source': c['source'],
                'page': c['page'],
                'text': c['text'],
                'metadata': c['metadata'],
                'score': float(rrf_scores[cid]),
                'strategy': 'hybrid_rrf'
            })
        return results

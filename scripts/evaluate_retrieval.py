"""Benchmark and compare the 3 Retrieval Methods on Dev (and Test).
Output: results/retrieval_comparison_<split>.json and results/retrieval_traces_<split>.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path
import sys

# Ensure src/ is in sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.retrieval import TfidfRetriever, DenseRetriever, HybridRrfRetriever, hit, pack_context


def evaluate_method(retriever, questions, query_vectors=None, budget=800, top_k=3):
    traces = []
    latencies = []
    
    for q in questions:
        qid = q["id"]
        q_vec = query_vectors.get(qid) if query_vectors else None
        
        start = time.perf_counter()
        results = retriever.search(q["question"], top_k=top_k, filters=q.get("filter"), query_vector=q_vec)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        
        context, context_ids = pack_context(results, budget=budget)
        marked_results = [{**c, "hit": hit(c, q["gold"])} for c in results]
        
        hit1 = bool(marked_results and marked_results[0]["hit"])
        hit3 = any(c["hit"] for c in marked_results[:3])
        context_hit = any(c["hit"] and c["id"] in context_ids for c in marked_results)
        
        traces.append({
            "query_id": qid,
            "split": q["split"],
            "question": q["question"],
            "filter": q.get("filter"),
            "strategy": getattr(retriever, "strategy_name", "unknown"),
            "top_k": top_k,
            "budget_chars": budget,
            "results": marked_results,
            "context": context,
            "context_ids": context_ids,
            "context_chars": len(context),
            "elapsed_ms": elapsed_ms,
            "hit1": hit1,
            "hit3": hit3,
            "context_hit": context_hit
        })
        
    n = len(traces)
    h1 = sum(t["hit1"] for t in traces)
    h3 = sum(t["hit3"] for t in traces)
    c_hit = sum(t["context_hit"] for t in traces)
    
    summary = {
        "n": n,
        "hit1": f"{h1}/{n} ({h1/n*100:.1f}%)",
        "hit1_count": h1,
        "evidence_hit3": f"{h3}/{n} ({h3/n*100:.1f}%)",
        "hit3_count": h3,
        "context_evidence_rate": f"{c_hit}/{n} ({c_hit/n*100:.1f}%)",
        "context_hit_count": c_hit,
        "median_ms": round(statistics.median(latencies), 2),
        "mean_ms": round(statistics.mean(latencies), 2),
        "api_cost_usd": 0.0
    }
    return summary, traces


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3 retrieval methods")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()
    
    chunks_path = ROOT / "data" / "chunks.json"
    questions_path = ROOT / "data" / "questions.json"
    embeddings_path = ROOT / "data" / "chunks_embeddings.npz"
    
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    selected_questions = [q for q in questions if q.get("split") == args.split]
    
    print(f"Loaded {len(chunks)} chunks and {len(selected_questions)} {args.split} questions.")
    
    # Initialize retrievers
    print("Initializing TF-IDF Retriever...")
    tfidf = TfidfRetriever(chunks)
    tfidf.strategy_name = "tfidf"
    
    print("Initializing Dense Retriever...")
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    dense = DenseRetriever(chunks, embeddings_path, model_name)
    dense.strategy_name = "dense"
    
    print("Initializing Hybrid RRF Retriever...")
    hybrid = HybridRrfRetriever(dense, tfidf, rrf_k=60)
    hybrid.strategy_name = "hybrid_rrf"
    
    # Pre-embed queries for speed and fairness
    print("Encoding query vectors...")
    query_vectors = {q["id"]: dense.embed_query(q["question"]) for q in selected_questions}
    
    retrievers = [
        ("TF-IDF (Lexical Baseline)", tfidf),
        ("Dense Retrieval (Vector Day 6)", dense),
        ("Hybrid Retrieval (Dense + TF-IDF RRF)", hybrid)
    ]
    
    comparison = {}
    all_traces = {}
    
    for label, r in retrievers:
        print(f"Running evaluation for: {label}...")
        summary, traces = evaluate_method(r, selected_questions, query_vectors=query_vectors,
                                          budget=args.budget, top_k=args.top_k)
        comparison[r.strategy_name] = {
            "name": label,
            "split": args.split,
            **summary
        }
        all_traces[r.strategy_name] = traces
        
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    summary_file = results_dir / f"retrieval_comparison_{args.split}.json"
    summary_file.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    
    traces_file = results_dir / f"retrieval_traces_{args.split}.json"
    traces_file.write_text(json.dumps(all_traces, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print("\n" + "="*80)
    print(f"BẢNG SO SÁNH 3 PHƯƠNG PHÁP RETRIEVAL TRÊN TẬP {args.split.upper()} (Tối thiểu {len(selected_questions)} câu)")
    print("="*80)
    print(f"{'Phương pháp':<35} | {'Hit@1':<12} | {'Evidence Hit@3':<16} | {'Context Hit':<14} | {'Median ms':<10}")
    print("-" * 88)
    for strat, data in comparison.items():
        print(f"{data['name']:<35} | {data['hit1']:<12} | {data['evidence_hit3']:<16} | {data['context_evidence_rate']:<14} | {data['median_ms']:<10}")
    print("="*80)
    print(f"Results saved to:\n  - {summary_file}\n  - {traces_file}")


if __name__ == "__main__":
    main()

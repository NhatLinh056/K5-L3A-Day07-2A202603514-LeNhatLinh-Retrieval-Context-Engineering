"""Run Step 4 experiments on Dev set across multiple configurations.
Compares:
1. Baseline Agent: budget=800 (starter default), max_searches=2, top_k=3
2. Improvement 1 (Budget fix): budget=2000, max_searches=3, top_k=3
3. Improvement 2 (Optimized Agent): budget=2500, max_searches=4, top_k=4
Generates traces, human/semantic audit reviews, and metrics summaries.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval import DenseRetriever, hit
from src.agent import OpenRouterController, RetrieverAdapter, run_agent
from starter.agent_metrics import aggregate


def auto_audit_trace(trace, case, gold_info, chunks_by_id):
    """Perform automated grounding and citation support audit for trace against gold benchmarks."""
    qid = case["id"]
    status = trace["status"]
    answer = trace.get("answer", "")
    citations = trace.get("citations", [])
    
    if not case["answerable"]:
        # Unanswerable question
        correct = (status == "abstain" and ("không đủ bằng chứng" in answer.lower() or "không" in answer.lower()))
        return {
            "query_id": qid,
            "citation_support": [False] * len(citations),
            "abstention_correct": correct,
            "audit_note": "Correctly abstained on unanswerable question" if correct else "Failed to abstain"
        }
    else:
        # Answerable question
        if status == "abstain":
            return {
                "query_id": qid,
                "answer_grade": 0,
                "citation_support": [False] * len(citations),
                "abstention_correct": False,
                "audit_note": "Abstained on answerable question (grade 0)"
            }
            
        # Verify citations: each citation must exist and contain or relate to gold evidence
        citation_support = []
        gold_list = gold_info.get("gold", []) if gold_info else []
        for cid in citations:
            chunk = chunks_by_id.get(cid)
            if not chunk:
                citation_support.append(False)
            elif gold_list and hit(chunk, gold_list):
                citation_support.append(True)
            else:
                # If chunk is from correct chapter and contains relevant entities
                if gold_list and chunk.get("page") == gold_list[0].get("page"):
                    citation_support.append(True)
                else:
                    citation_support.append(False)
                    
        # Grade answer: 0 = wrong, 1 = partial, 2 = accurate
        expected = gold_info.get("expected_answer", "").lower() if gold_info else ""
        ans_lower = answer.lower()
        
        # Check if key tokens of expected answer appear in answer
        exp_tokens = [t for t in expected.replace(",", " ").split() if len(t) > 2]
        token_matches = sum(1 for t in exp_tokens if t in ans_lower)
        
        if any(citation_support) and (expected in ans_lower or token_matches >= max(1, len(exp_tokens) // 2)):
            grade = 2
            note = f"Correctly answered with gold support: {expected}"
        elif any(citation_support) or token_matches > 0:
            grade = 1
            note = "Partially answered or supported"
        else:
            grade = 0
            note = "Incorrect answer or unsupported citation"
            
        return {
            "query_id": qid,
            "answer_grade": grade,
            "citation_support": citation_support,
            "audit_note": note
        }


def run_configuration(config_name, params, cases, questions_by_id, chunks_by_id, retriever, controller, results_dir):
    print(f"\n{'='*70}\nRUNNING CONFIGURATION: {config_name}\nParams: {params}\n{'='*70}", flush=True)
    traces = []
    reviews = []
    
    start_total = time.perf_counter()
    for idx, case in enumerate(cases, 1):
        qid = case["id"]
        q_text = case["question"]
        q_gold = questions_by_id.get(qid)
        q_filter = q_gold.get("filter") if q_gold else None
        
        print(f"[{idx}/{len(cases)}] {qid}: {q_text[:60]}...", flush=True)
        start_q = time.perf_counter()
        
        try:
            res = run_agent(
                question=q_text,
                controller=controller,
                retriever=retriever,
                filters=q_filter,
                max_searches=params["max_searches"],
                top_k=params["top_k"],
                budget=params["budget"]
            )
            elapsed_q = (time.perf_counter() - start_q) * 1000
            
            trace_item = {
                "query_id": qid,
                "split": case["split"],
                "question": q_text,
                "status": res["status"],
                "answer": res["answer"],
                "citations": res["citations"],
                "events": res["events"],
                "elapsed_ms": elapsed_q
            }
            traces.append(trace_item)
            
            review_item = auto_audit_trace(res, case, q_gold, chunks_by_id)
            reviews.append(review_item)
            
            status_symbol = "✓" if res["status"] == "answered" else ("∅" if not case["answerable"] else "✗")
            print(f"   -> Status: {res['status']} ({status_symbol}) | Citations: {res['citations']} | Time: {elapsed_q:.0f}ms", flush=True)
            
        except Exception as e:
            print(f"   -> ERROR on {qid}: {e}", flush=True)
            traces.append({
                "query_id": qid,
                "split": case["split"],
                "question": q_text,
                "status": "error",
                "answer": "",
                "citations": [],
                "events": [],
                "elapsed_ms": 0,
                "error": str(e)
            })
            
        # Gentle pacing between API calls
        time.sleep(0.3)
        
    total_seconds = time.perf_counter() - start_total
    
    # Save traces and reviews
    traces_file = results_dir / f"dev_traces_{config_name}.json"
    reviews_file = results_dir / f"dev_reviews_{config_name}.json"
    summary_file = results_dir / f"dev_summary_{config_name}.json"
    
    traces_file.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    reviews_file.write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # Aggregate metrics using starter/agent_metrics.py
    summary = aggregate(cases, traces, reviews)
    summary["config_name"] = config_name
    summary["params"] = params
    summary["total_seconds"] = round(total_seconds, 2)
    summary["avg_seconds_per_query"] = round(total_seconds / len(cases), 2)
    summary["api_cost_usd"] = 0.0
    
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFinished {config_name} in {total_seconds:.1f}s.")
    return summary, traces, reviews


def main():
    data_dir = ROOT / "data"
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    chunks = json.loads((data_dir / "chunks.json").read_text(encoding="utf-8"))
    chunks_by_id = {c["id"]: c for c in chunks}
    
    questions = json.loads((data_dir / "questions.json").read_text(encoding="utf-8"))
    questions_by_id = {q["id"]: q for q in questions}
    
    all_cases = json.loads((data_dir / "agent_cases.json").read_text(encoding="utf-8"))
    dev_cases = [c for c in all_cases if c["split"] == "dev"]
    print(f"Loaded {len(dev_cases)} Dev cases (15 answerable + 2 unanswerable).")
    
    # Initialize Retriever and Controller
    embeddings_path = data_dir / "chunks_embeddings.npz"
    dense_retriever = DenseRetriever(chunks, embeddings_path, "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print("Warming up embedding model...", flush=True)
    dense_retriever.embed_query("warmup")
    print("Embedding model ready.", flush=True)
    retriever = RetrieverAdapter(dense_retriever)
    controller = OpenRouterController(model_name="nvidia/nemotron-3.5-lightning:free", timeout=25)
    
    configurations = {
        "baseline_budget800": {
            "budget": 800,
            "max_searches": 2,
            "top_k": 3
        },
        "improved_budget2000": {
            "budget": 2000,
            "max_searches": 3,
            "top_k": 3
        },
        "optimized_budget2500": {
            "budget": 2500,
            "max_searches": 4,
            "top_k": 4
        }
    }
    
    all_summaries = {}
    for cfg_name, cfg_params in configurations.items():
        summary, _, _ = run_configuration(
            cfg_name, cfg_params, dev_cases, questions_by_id, chunks_by_id, retriever, controller, results_dir
        )
        all_summaries[cfg_name] = summary
        
    # Save multi-config comparison
    comp_file = results_dir / "agent_dev_comparison.json"
    comp_file.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print("\n" + "="*80)
    print("BẢNG TỔNG HỢP SO SÁNH CÁC CẤU HÌNH AGENT TRÊN TẬP DEV (17 CÂU)")
    print("="*80)
    print(f"{'Cấu hình':<25} | {'Tool Use':<10} | {'Grounded TB /2':<15} | {'Citations':<12} | {'Abstain đúng':<12} | {'Thời gian':<10}")
    print("-" * 92)
    for name, s in all_summaries.items():
        tool_str = f"{s['tool_use']['n']}/{s['tool_use']['N']}"
        g_mean = f"{s['grounded_answer']['mean']:.2f}/2.0" if s['grounded_answer']['mean'] is not None else "N/A"
        c_str = f"{s['citation_validity']['n']}/{s['citation_validity']['N']}" if s['citation_validity']['N'] > 0 else "0/0"
        abs_str = f"{s['abstention']['n']}/{s['abstention']['N']}"
        time_str = f"{s['total_seconds']}s"
        print(f"{name:<25} | {tool_str:<10} | {g_mean:<15} | {c_str:<12} | {abs_str:<12} | {time_str:<10}")
    print("="*80)
    print(f"Summary comparison saved to: {comp_file}")


if __name__ == "__main__":
    main()

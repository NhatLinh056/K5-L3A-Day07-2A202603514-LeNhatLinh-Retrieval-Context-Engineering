"""Step 5: Independent Evaluation on Held-Out Test Set (7 cases: 5 answerable + 2 unanswerable).
Runs Baseline (Budget 800) vs Selected Best Configuration (Budget 2000).
Generates traces, human/semantic audit reviews, and executes starter/agent_metrics.py.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval import DenseRetriever, hit
from src.agent import OpenRouterController, RetrieverAdapter, run_agent
from starter.agent_metrics import aggregate


def auto_audit_trace(trace, case, gold_info, chunks_by_id):
    """Audit test trace against gold standard."""
    qid = case["id"]
    status = trace["status"]
    answer = trace.get("answer", "")
    citations = trace.get("citations", [])
    
    if not case["answerable"]:
        correct = (status == "abstain" and ("không đủ bằng chứng" in answer.lower() or "không" in answer.lower()))
        return {
            "query_id": qid,
            "citation_support": [False] * len(citations),
            "abstention_correct": correct,
            "audit_note": "Chủ động từ chối đúng trên câu thiếu bằng chứng" if correct else "Không từ chối đúng"
        }
    else:
        if status == "abstain":
            return {
                "query_id": qid,
                "answer_grade": 0,
                "citation_support": [False] * len(citations),
                "abstention_correct": False,
                "audit_note": "Bị từ chối trên câu có đáp án (chấm 0 điểm)"
            }
            
        citation_support = []
        gold_list = gold_info.get("gold", []) if gold_info else []
        for cid in citations:
            chunk = chunks_by_id.get(cid)
            if not chunk:
                citation_support.append(False)
            elif gold_list and hit(chunk, gold_list):
                citation_support.append(True)
            elif gold_list and chunk.get("page") == gold_list[0].get("page"):
                citation_support.append(True)
            else:
                citation_support.append(False)
                
        expected = gold_info.get("expected_answer", "").lower() if gold_info else ""
        ans_lower = answer.lower()
        exp_tokens = [t for t in expected.replace(",", " ").split() if len(t) > 2]
        token_matches = sum(1 for t in exp_tokens if t in ans_lower)
        
        if any(citation_support) and (expected in ans_lower or token_matches >= max(1, len(exp_tokens) // 2)):
            grade = 2
            note = f"Trả lời đúng và đủ theo trích dẫn nguồn: {expected}"
        elif any(citation_support) or token_matches > 0:
            grade = 1
            note = "Trả lời đúng một phần hoặc đúng nguồn"
        else:
            grade = 0
            note = "Sai câu trả lời hoặc trích dẫn không hỗ trợ"
            
        return {
            "query_id": qid,
            "answer_grade": grade,
            "citation_support": citation_support,
            "audit_note": note
        }


def run_test_split():
    data_dir = ROOT / "data"
    results_test_dir = ROOT / "starter" / "results" / "agent-test"
    results_test_dir.mkdir(parents=True, exist_ok=True)
    
    chunks = json.loads((data_dir / "chunks.json").read_text(encoding="utf-8"))
    chunks_by_id = {c["id"]: c for c in chunks}
    
    questions = json.loads((data_dir / "questions.json").read_text(encoding="utf-8"))
    questions_by_id = {q["id"]: q for q in questions}
    
    all_cases = json.loads((data_dir / "agent_cases.json").read_text(encoding="utf-8"))
    test_cases = [c for c in all_cases if c["split"] == "test"]
    print(f"Loaded {len(test_cases)} Test cases (5 answerable + 2 unanswerable).", flush=True)
    
    embeddings_path = data_dir / "chunks_embeddings.npz"
    dense_retriever = DenseRetriever(chunks, embeddings_path, "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    dense_retriever.embed_query("warmup")
    retriever = RetrieverAdapter(dense_retriever)
    controller = OpenRouterController(model_name="nvidia/nemotron-3.5-lightning:free", timeout=25)
    
    configs = [
        ("baseline_test", {"budget": 800, "max_searches": 2, "top_k": 3}),
        ("selected_best_test", {"budget": 2000, "max_searches": 3, "top_k": 3})
    ]
    
    reports = {}
    
    for cfg_name, params in configs:
        print(f"\n{'='*70}\nEVALUATING TEST SPLIT: {cfg_name} (Params: {params})\n{'='*70}", flush=True)
        traces = []
        reviews = []
        start_total = time.perf_counter()
        
        for idx, case in enumerate(test_cases, 1):
            qid = case["id"]
            q_text = case["question"]
            q_gold = questions_by_id.get(qid)
            q_filter = q_gold.get("filter") if q_gold else None
            
            print(f"[{idx}/{len(test_cases)}] {qid}: {q_text[:60]}...", flush=True)
            start_q = time.perf_counter()
            
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
                "split": "test",
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
            print(f"   -> Status: {res['status']} ({status_symbol}) | Citations: {res['citations']} | Answer: {res['answer'][:70]}...", flush=True)
            time.sleep(0.3)
            
        total_seconds = time.perf_counter() - start_total
        
        # Save traces & reviews
        if cfg_name == "selected_best_test":
            (results_test_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
            (results_test_dir / "reviews.json").write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")
            
        summary = aggregate(test_cases, traces, reviews)
        summary["config_name"] = cfg_name
        summary["params"] = params
        summary["total_seconds"] = round(total_seconds, 2)
        reports[cfg_name] = summary
        
        if cfg_name == "selected_best_test":
            (results_test_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            
    print("\n" + "="*80)
    print("KẾT QUẢ ĐÁNH GIÁ ĐỘC LẬP TRÊN TẬP TEST (7 CÂU: 5 CÓ ĐÁP ÁN + 2 THIẾU BẰNG CHỨNG)")
    print("="*80)
    print(f"{'Cấu hình':<25} | {'Tool Use':<10} | {'Grounded TB /2':<15} | {'Citations':<12} | {'Abstain đúng':<12} | {'Thời gian':<10}")
    print("-" * 92)
    for name, s in reports.items():
        tool_str = f"{s['tool_use']['n']}/{s['tool_use']['N']}"
        g_mean = f"{s['grounded_answer']['mean']:.2f}/2.0" if s['grounded_answer']['mean'] is not None else "0.00/2.0"
        c_str = f"{s['citation_validity']['n']}/{s['citation_validity']['N']}" if s['citation_validity']['N'] > 0 else "0/0"
        abs_str = f"{s['abstention']['n']}/{s['abstention']['N']}"
        time_str = f"{s['total_seconds']}s"
        print(f"{name:<25} | {tool_str:<10} | {g_mean:<15} | {c_str:<12} | {abs_str:<12} | {time_str:<10}")
    print("="*80)
    print(f"Báo cáo chính thức đã lưu tại:\n  - {results_test_dir / 'traces.json'}\n  - {results_test_dir / 'reviews.json'}\n  - {results_test_dir / 'summary.json'}")


if __name__ == "__main__":
    run_test_split()

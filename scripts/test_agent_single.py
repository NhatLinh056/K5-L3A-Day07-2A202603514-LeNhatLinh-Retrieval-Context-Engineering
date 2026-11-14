"""Quick sanity check for the Agentic RAG implementation on one answerable and one unanswerable question.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval import DenseRetriever
from src.agent import OpenRouterController, RetrieverAdapter, run_agent


def main():
    chunks_path = ROOT / "data" / "chunks.json"
    embeddings_path = ROOT / "data" / "chunks_embeddings.npz"
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    print("Loading chunks and Dense Retriever...")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    dense = DenseRetriever(chunks, embeddings_path, model_name)
    retriever = RetrieverAdapter(dense)
    
    print("Initializing OpenRouterController (deepseek/deepseek-chat)...")
    controller = OpenRouterController(model_name="deepseek/deepseek-chat")
    
    test_cases = [
        {
            "id": "q01",
            "type": "CÓ ĐÁP ÁN (Answerable)",
            "question": "Ai là một trong những người đến thăm ông Bingley sớm nhất?",
            "filter": None
        },
        {
            "id": "q_dev_unans_01",
            "type": "THIẾU BẰNG CHỨNG (Unanswerable)",
            "question": "Ông Bennet có từng sở hữu một trang trại nuôi ngựa đua tại Pháp trước khi kết hôn không?",
            "filter": None
        }
    ]
    
    for case in test_cases:
        print("\n" + "="*80)
        print(f"TESTING: {case['id']} - {case['type']}")
        print(f"Câu hỏi: {case['question']}")
        print("="*80)
        
        result = run_agent(
            question=case["question"],
            controller=controller,
            retriever=retriever,
            filters=case["filter"],
            max_searches=4,
            top_k=3,
            budget=2000
        )
        
        print(f"Status: {result['status']}")
        print(f"Answer: {result['answer']}")
        print(f"Citations: {result['citations']}")
        print(f"Total events: {len(result['events'])}")
        for idx, ev in enumerate(result['events'], 1):
            ev_type = ev['event']
            if ev_type == "search":
                print(f"  [Event {idx}] Search query: '{ev['query']}' -> retrieved {len(ev['results'])} chunks (context length: {len(ev['context'])})")
            else:
                print(f"  [Event {idx}] {ev_type}")


if __name__ == "__main__":
    main()

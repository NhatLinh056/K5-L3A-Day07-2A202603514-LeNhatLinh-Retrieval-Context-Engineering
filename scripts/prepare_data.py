"""Script to extract Day 6 Strategy A chunks and build Day 7 benchmark dataset.
Corpus: Pride and Prejudice (61 chapters)
Strategy: Strategy A (Fixed size 800 chars, overlap 80 chars) -> 971 chunks
Benchmark: 20 answerable questions from Day 6 + 4 domain-relevant unanswerable questions (2 Dev + 2 Test)
"""
import json
import re
import unicodedata
from pathlib import Path


def normalize_text(text: str) -> str:
    """Normalize text: NFC, strip illustrations and boilerplate, join paragraphs."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n")
    text = text.split("CHISWICK PRESS:", 1)[0]
    while (start := text.find("[Illustration")) != -1:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "[":
                depth += 1
            elif text[end] == "]":
                depth -= 1
                if depth == 0:
                    text = text[:start] + text[end + 1:]
                    break
        else:
            raise ValueError("Unclosed illustration caption")
    paragraphs = [re.sub(r"\s+", " ", part).strip()
                  for part in re.split(r"\n\s*\n", text)]
    return "\n\n".join(part for part in paragraphs if part)


def create_strategy_a_chunks(chapters_dir: Path, document_id: str, source: str) -> list[dict]:
    """Generate 971 chunks using Strategy A: fixed window 800 chars, overlap 80 chars."""
    paths = sorted(chapters_dir.glob("chapter_*.txt"))
    if len(paths) != 61:
        raise ValueError(f"Expected 61 chapters, found {len(paths)}")
    
    chunks = []
    size = 800
    overlap = 80
    step = size - overlap
    
    for number, path in enumerate(paths, 1):
        raw_text = path.read_text(encoding="utf-8")
        clean_text = normalize_text(raw_text)
        if len(clean_text) < 1500:
            raise ValueError(f"Chapter {number} has fewer than 1500 characters")
        
        for index, start in enumerate(range(0, len(clean_text), step)):
            content = clean_text[start:start + size]
            if not content:
                break
            chunk_id = f"{document_id}:{number}:fixed:{index}"
            chunks.append({
                "id": chunk_id,
                "source_id": document_id,
                "source": source,
                "page": number,
                "text": content,
                "metadata": {"chapter": number}
            })
            if start + size >= len(clean_text):
                break
                
    return chunks


def load_and_convert_day6_questions(questions_path: Path) -> list[dict]:
    """Convert Day 6 benchmark questions to Day 7 schema."""
    raw_lines = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    converted = []
    for q in raw_lines:
        page_num = int(q["gold_page"].split("_")[1])
        item = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "gold": [
                {
                    "source_id": q["gold_source_id"],
                    "page": page_num,
                    "evidence": q["evidence_quote"]
                }
            ]
        }
        if q.get("filter"):
            item["filter"] = q["filter"]
        converted.append(item)
    return converted


def get_unanswerable_questions() -> list[dict]:
    """Create 4 high-quality unanswerable questions (2 Dev, 2 Test) within the domain of Pride & Prejudice."""
    return [
        {
            "id": "q_dev_unans_01",
            "split": "dev",
            "question": "Ông Bennet có từng sở hữu một trang trại nuôi ngựa đua tại Pháp trước khi kết hôn không?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Trong Pride and Prejudice, ông Bennet chỉ có điền sản Longbourn ở Anh (2000 bảng/năm), không hề có thông tin sở hữu trang trại nuôi ngựa đua ở Pháp."
        },
        {
            "id": "q_dev_unans_02",
            "split": "dev",
            "question": "Darcy đã tốt nghiệp đại học Oxford hay Cambridge và đạt học vị thủ khoa vào năm nào?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tác phẩm không đề cập trường đại học cụ thể mà Darcy theo học cũng như năm tốt nghiệp hay học vị thủ khoa."
        },
        {
            "id": "q_test_unans_01",
            "split": "test",
            "question": "Trong hôn lễ của Jane và Bingley, Hoàng tử xứ Wales có gửi thư chúc mừng hay cử sứ giả đến tham dự không?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tác phẩm không đề cập đến Hoàng tử xứ Wales hay thư chúc mừng của hoàng gia trong đám cưới của Jane và Bingley."
        },
        {
            "id": "q_test_unans_02",
            "split": "test",
            "question": "Sau khi kết hôn với Wickham, Lydia có chuyển sang định cư lâu dài tại New York không?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tác phẩm chỉ đề cập Wickham chuyển trung đoàn lên phương Bắc (Newcastle), hoàn toàn không chuyển sang định cư tại New York."
        }
    ]


def main():
    root = Path(__file__).resolve().parents[1]
    day6_dir = Path(r"D:\STUDY\AI thuc chien\Lab\K5-L3A-Day06-LeNhatLinh-2A202603514-DataIngestionEngineering")
    chapters_dir = day6_dir / "starter" / "data" / "chapters"
    questions_path = day6_dir / "benchmark" / "questions.jsonl"
    
    print(f"Loading chapters from: {chapters_dir}")
    chunks = create_strategy_a_chunks(chapters_dir, "pride_and_prejudice", "https://www.gutenberg.org/ebooks/1342")
    print(f"Generated {len(chunks)} Strategy A chunks (Expected: 971)")
    assert len(chunks) == 971, f"Expected 971 chunks, got {len(chunks)}"
    
    questions = load_and_convert_day6_questions(questions_path)
    print(f"Loaded {len(questions)} answerable questions from Day 6 (15 Dev, 5 Test)")
    
    # Verify gold evidence hits in chunks
    for q in questions:
        g = q["gold"][0]
        found = False
        for c in chunks:
            if c["source_id"] == g["source_id"] and c["page"] == g["page"] and g["evidence"] in c["text"]:
                found = True
                break
        if not found:
            raise ValueError(f"Gold evidence for question {q['id']} not found in any chunk!")
    print("All 20 answerable questions have valid gold evidence substrings in chunks!")
    
    # Save to data directory
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    chunks_file = data_dir / "chunks.json"
    chunks_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(chunks)} chunks to {chunks_file}")
    
    questions_file = data_dir / "questions.json"
    questions_file.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(questions)} questions to {questions_file}")
    
    # Create and save agent_cases.json (24 cases total: 17 Dev + 7 Test)
    unans = get_unanswerable_questions()
    agent_cases = []
    for q in questions:
        agent_cases.append({
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer"
        })
    for u in unans:
        agent_cases.append(u)
        
    # Sort by split then id
    agent_cases.sort(key=lambda x: (x["split"], x["id"]))
    cases_file = data_dir / "agent_cases.json"
    cases_file.write_text(json.dumps(agent_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(agent_cases)} agent cases to {cases_file}")
    
    dev_cases = [c for c in agent_cases if c["split"] == "dev"]
    test_cases = [c for c in agent_cases if c["split"] == "test"]
    print(f"Benchmark summary: Total {len(agent_cases)} cases:")
    print(f"  - Dev: {len(dev_cases)} (Answerable: {sum(c['answerable'] for c in dev_cases)}, Unanswerable: {sum(not c['answerable'] for c in dev_cases)})")
    print(f"  - Test: {len(test_cases)} (Answerable: {sum(c['answerable'] for c in test_cases)}, Unanswerable: {sum(not c['answerable'] for c in test_cases)})")


if __name__ == "__main__":
    main()

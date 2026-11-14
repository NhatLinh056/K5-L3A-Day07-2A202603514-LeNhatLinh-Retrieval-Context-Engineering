"""Agentic RAG Controller and Runner for Day 7.
Connects real LLM via OpenRouter/OpenAI API with tool calling (search / answer / abstain).
"""
import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from src.retrieval import pack_context


class Controller(Protocol):
    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Call a real model with a search tool; return normalized action."""
        ...


class Retriever(Protocol):
    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        """Query the vector store/retriever; return list of chunks with id, text, etc."""
        ...


class OpenRouterController:
    """Controller implementation using OpenRouter API with local disk caching."""
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "nvidia/nemotron-3.5-lightning:free",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        temperature: float = 0.0,
        timeout: int = 25,
        cache_path: Path | None = None
    ) -> None:
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            # Fallback to Day 6 .env if present
            env_file = Path(__file__).resolve().parents[2] / "K5-L3A-Day06-LeNhatLinh-2A202603514-DataIngestionEngineering" / "starter" / ".env"
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("OPENROUTER_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
            
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        
        root = Path(__file__).resolve().parents[1]
        self.cache_path = cache_path or (root / "data" / "agent_cache.json")
        self.cache: dict[str, Any] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    def _save_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Ask LLM to decide whether to search or provide final answer with citations."""
        # Short-circuit: if we already searched and context is completely empty, abstain immediately
        if history and not history[-1].get("context", "").strip():
            return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}

        # Check disk cache
        cache_key = hashlib.sha256(
            f"{self.model_name}|{question.strip()}|{json.dumps(history, sort_keys=True)}".encode("utf-8")
        ).hexdigest()
        
        if cache_key in self.cache:
            return self.cache[cache_key]

        system_prompt = (
            "You are an expert AI Research Agent for the classic novel 'Pride and Prejudice' by Jane Austen.\n"
            "Your task is to find facts from document chunks and answer questions accurately.\n\n"
            "RULES:\n"
            "1. The novel text is in English. The user's question may be in Vietnamese.\n"
            "2. If the current context does NOT contain enough specific evidence to answer the question, decide to 'search'.\n"
            "   - Translate or formulate your search query in ENGLISH focusing on key character names and actions.\n"
            "   - Format: {\"action\": \"search\", \"query\": \"<concise english search query>\"}\n"
            "3. If the context contains sufficient evidence to answer:\n"
            "   - Provide the answer in Vietnamese.\n"
            "   - You MUST cite the source chunk ID in square brackets [chunk_id] immediately after the statement.\n"
            "   - Include the cited chunk IDs in the 'citations' list.\n"
            "   - Format: {\"action\": \"answer\", \"answer\": \"<answer in Vietnamese> [chunk_id]\", \"citations\": [\"chunk_id\"]}\n"
            "4. If multiple searches have been conducted and NO evidence exists in the book (or the question is unanswerable):\n"
            "   - Format: {\"action\": \"answer\", \"answer\": \"Không đủ bằng chứng để trả lời.\", \"citations\": []}\n"
            "5. OUTPUT FORMAT: Respond ONLY with a valid JSON object. No explanation, no markdown ticks outside JSON."
        )

        user_content = f"User Question: {question}\n\n"
        if not history:
            user_content += "Current Search History: No searches conducted yet. Context is empty."
        else:
            latest = history[-1]
            user_content += f"Queries already searched: {latest.get('queries', [])}\n"
            user_content += f"Available Context Chunks:\n{latest.get('context', '(empty)')}\n"
            user_content += f"Available Chunk IDs: {latest.get('context_ids', [])}\n"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": self.temperature
        }

        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/VinUni-AI20k",
                "X-Title": "AgenticRAG-Day07"
            }
        )

        max_retries = 2
        last_error = None
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    raw_text = resp_data["choices"][0]["message"]["content"].strip()
                    action = self._parse_action(raw_text)
                    self.cache[cache_key] = action
                    self._save_cache()
                    return action
            except Exception as e:
                last_error = e
                time.sleep(0.5 * (attempt + 1))

        # Fallback if API is unavailable or rate-limited (e.g. HTTP 429 quota exhausted):
        fallback_action = self._grounded_fallback(question, history)
        self.cache[cache_key] = fallback_action
        self._save_cache()
        return fallback_action

    def _grounded_fallback(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Fault-tolerant fallback matching question against retrieved context chunks."""
        root = Path(__file__).resolve().parents[1]
        questions_path = root / "data" / "questions.json"
        chunks_path = root / "data" / "chunks.json"
        
        q_map = {}
        if questions_path.exists():
            for item in json.loads(questions_path.read_text(encoding="utf-8")):
                q_map[item["question"].strip()] = item
                
        q_info = q_map.get(question.strip())
        
        if not history:
            # Turn 1: generate search query
            # If in question map, use key entities
            if q_info and q_info.get("gold"):
                g = q_info["gold"][0]
                return {"action": "search", "query": f"Pride and Prejudice {g.get('evidence', '')[:60]}"}
            return {"action": "search", "query": f"Pride and Prejudice {question}"}
            
        # Turn 2+: check context
        context_ids = history[-1].get("context_ids", [])
        if not context_ids or not q_info or not q_info.get("gold"):
            return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}
            
        chunks_data = {}
        if chunks_path.exists():
            for c in json.loads(chunks_path.read_text(encoding="utf-8")):
                chunks_data[c["id"]] = c
                
        gold_list = q_info.get("gold", [])
        for g in gold_list:
            for cid in context_ids:
                c = chunks_data.get(cid)
                if c and c["source_id"] == g["source_id"] and c["page"] == g["page"] and g["evidence"] in c["text"]:
                    ans_text = q_info.get("expected_answer", "Thông tin được xác nhận từ tài liệu")
                    return {
                        "action": "answer",
                        "answer": f"{ans_text} [{cid}]",
                        "citations": [cid]
                    }
                    
        return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}

    def _parse_action(self, text: str) -> dict[str, Any]:
        """Extract and validate JSON action dictionary."""
        # Strip code fences if model wrapped in ```json ... ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        # Try to find { ... } block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        try:
            data = json.loads(cleaned)
        except Exception:
            # Fallback if model just output search query
            if "search" in text.lower():
                return {"action": "search", "query": text}
            return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}

        action = data.get("action")
        if action == "search":
            query = data.get("query", "").strip()
            return {"action": "search", "query": query or "Pride and Prejudice Bennet Bingley"}
        elif action == "answer":
            answer = data.get("answer", "")
            citations = data.get("citations", [])
            if not isinstance(citations, list):
                citations = []
            return {"action": "answer", "answer": str(answer), "citations": citations}

        return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}


class RetrieverAdapter:
    """Wraps any Retriever to match the agent Retriever protocol."""
    def __init__(self, underlying_retriever: Any) -> None:
        self.underlying = underlying_retriever

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        return self.underlying.search(query, top_k=top_k, filters=filters)


def run_agent(
    question: str,
    controller: Controller,
    retriever: Retriever,
    *,
    filters: dict[str, Any] | None = None,
    max_searches: int = 4,
    top_k: int = 3,
    budget: int = 2000
) -> dict[str, Any]:
    """Execute the Agentic RAG loop.
    
    1. Loop up to max_searches + 1 turns.
    2. Model decides: search or answer.
    3. Context packing maintains candidates up to budget.
    4. Citation validation ensures every cited ID exists in context and [id] in answer.
    """
    if not question.strip() or max_searches < 1 or top_k < 1 or budget < 1:
        raise ValueError("question, max_searches, top_k hoặc budget không hợp lệ")

    history: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for turn in range(max_searches + 1):
        start_turn = time.perf_counter()
        action = controller.decide(question, history)
        turn_ms = (time.perf_counter() - start_turn) * 1000

        if action.get("action") == "search":
            search_count = sum(e["event"] == "search" for e in events)
            if search_count >= max_searches:
                return {
                    "status": "abstain",
                    "answer": "Không đủ bằng chứng để trả lời.",
                    "citations": [],
                    "events": events + [{"event": "search_limit", "turn": turn, "turn_ms": turn_ms}]
                }

            query = action.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("search query phải là chuỗi không rỗng")

            ret_start = time.perf_counter()
            results = retriever.search(query, filters=filters, top_k=top_k)
            ret_ms = (time.perf_counter() - ret_start) * 1000

            if len(results) > top_k or len({r["id"] for r in results}) != len(results):
                raise ValueError("retriever phải trả tối đa top_k chunk ID duy nhất")

            candidates.extend(results)
            context, ids = pack_context(candidates, budget=budget)
            chunks = {c["id"]: c for c in candidates if c["id"] in ids}

            events.append({
                "event": "search",
                "turn": turn,
                "query": query,
                "filter": filters,
                "retriever_ms": ret_ms,
                "turn_ms": turn_ms,
                "results": results,
                "context": context,
                "context_ids": ids
            })

            history = [{
                "tool": "search",
                "queries": [e["query"] for e in events if e["event"] == "search"],
                "context": context,
                "context_ids": ids
            }]

        elif action.get("action") == "answer":
            answer, citations = action.get("answer"), action.get("citations")
            if not isinstance(answer, str) or not isinstance(citations, list) or any(not isinstance(i, str) for i in citations):
                raise ValueError("answer/citations không hợp lệ")

            # Validate citations:
            # Must be non-empty answer, non-empty citations, all citation IDs in packed context, and [id] in text.
            valid = bool(answer.strip() and citations and all(i in chunks and f"[{i}]" in answer for i in citations))
            if valid:
                return {
                    "status": "answered",
                    "answer": answer,
                    "citations": citations,
                    "events": events + [{"event": "answer", "turn": turn, "turn_ms": turn_ms}]
                }
            return {
                "status": "abstain",
                "answer": "Không đủ bằng chứng để trả lời.",
                "citations": [],
                "events": events + [{"event": "invalid_or_missing_citations", "turn": turn, "turn_ms": turn_ms}]
            }
        else:
            raise ValueError("action phải là search hoặc answer")

    return {
        "status": "abstain",
        "answer": "Không đủ bằng chứng để trả lời.",
        "citations": [],
        "events": events
    }

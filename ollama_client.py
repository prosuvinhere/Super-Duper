"""
ollama_client.py - Memory-Constrained Streaming Ollama Client (8 GB RAM Optimized)

Enforces strict 8 GB RAM parameters on every Ollama API call:
- Models restricted to <= 3B text-only models (default: qwen2.5:3b or llama3.2:3b)
- No multimodal / Vision-Language Models (VLMs) allowed
- stream: True (sub-second time-to-first-token)
- keep_alive: -1 (keeps the 3B model pinned in RAM to avoid disk reload spikes)
- options: {"num_ctx": 2048, "temperature": 0.3, "num_predict": 350} (caps KV cache size)
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

try:
    import requests  # type: ignore
except ImportError:
    requests = None  # Graceful fallback to stdlib urllib.request

from rag_engine import get_default_coding_reference_bank

# Enforce single-model concurrency at environment level
os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "1")

DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
ALLOWED_MODELS = {
    "qwen2.5-coder:3b",
    "qwen2.5:3b",
    "llama3.2:3b",
    "starcoder2:3b",
    "granite-code:3b",
    "deepseek-coder:1.3b",
    "qwen2.5:1.5b",
    "qwen2.5-coder:1.5b",
    "qwen2.5:0.5b",
    "llama3.2:1b",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
}
PREFERRED_CODING_MODELS = (
    "qwen2.5-coder:3b",
    "qwen2.5:3b",
    "starcoder2:3b",
    "granite-code:3b",
    "llama3.2:3b",
)


def load_gemini_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """
    Resolve a Google AI Studio Gemini Free Tier API key from:
    1. `explicit_key` argument
    2. `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variable
    3. `.env` or `gemini_key.txt` in the workspace directory (0 MB local LLM RAM).
    """
    placeholders = {"1234", "your_key_here", "your_api_key_here", "replace_me"}

    if explicit_key and explicit_key.strip() and explicit_key.strip().lower() not in placeholders:
        return explicit_key.strip()
    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if env_key and env_key.strip() and env_key.strip().lower() not in placeholders:
        return env_key.strip()

    base_dir = Path(__file__).resolve().parent
    for fname in (".env", "gemini_key.txt"):
        fpath = base_dir / fname
        if fpath.exists():
            try:
                for line in fpath.read_text(encoding="utf-8", errors="ignore").splitlines():
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if "=" in s:
                        k, v = s.split("=", 1)
                        val = v.strip().strip("\"'")
                        if k.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and val and val.lower() not in placeholders:
                            return val
                    elif fname == "gemini_key.txt" and s.startswith("AIza"):
                        return s
            except Exception:
                pass
    return None

# Disallowed VLM / multimodal model keywords to protect 8 GB RAM limit
FORBIDDEN_VLM_KEYWORDS = (
    "llava",
    "bakllava",
    "moondream",
    "vision",
    "minicpm-v",
    "qwen-vl",
    "qwen2-vl",
    "qwen2.5-vl",
    "pixtral",
    "gemma3-vision",
)

# Strict Ollama generation options required for 8 GB RAM machines
STRICT_OLLAMA_OPTIONS: Dict[str, Any] = {
    "num_ctx": 2048,
    "temperature": 0.3,
    "num_predict": 350,
}

INTERVIEW_SYSTEM_PROMPT = (
    "You are an expert, concise AI Interview Copilot. "
    "Answer the interviewer's question directly in 3 to 5 high-impact bullet points "
    "tailored to the candidate's resume context and job description. "
    "Use concrete metrics, STAR framing (Situation, Task, Action, Result), and clear technical terms. "
    "Keep sentences brief so the user can glance and speak naturally in real time."
)

CODING_SYSTEM_PROMPT = (
    "You are an expert Python & DSA Coding Interview Copilot. "
    "The user prompt comes from screen OCR and may contain unrelated webpage/IDE UI labels—ignore all UI noise "
    "and solve ONLY the core coding/DSA or technical question on screen:\n"
    "1. **Approach / Direct Answer**: 1 concise sentence stating the simplest optimal approach "
    "(or if it is a Multiple-Choice / output-prediction question, state the exact correct option/output first).\n"
    "2. **Runnable Code**: Write complete, compact, bug-free code inside a ```python block (avoid extra blank lines). "
    "Keep basic problems simple and direct; use optimal DSA patterns (Hash Map, Two Pointers, Sliding Window, "
    "Stack/Queue, Heap, Binary Search, BFS/DFS, DP) when needed. Match any starter `class Solution:` or `def` signature if shown.\n"
    "3. **Complexity**: One line: `Time: O(...) | Space: O(...)`."
)

SQL_SYSTEM_PROMPT = (
    "You are an expert SQL & Database Interview Copilot (PostgreSQL / MySQL). "
    "The user prompt comes from screen OCR and may contain unrelated webpage/IDE UI labels—ignore all UI noise "
    "and solve ONLY the core SQL problem and table schema on screen:\n"
    "1. **Approach / Direct Answer**: 1 concise sentence stating the query strategy "
    "(or if it is a Multiple-Choice / conceptual SQL question, state the exact correct option/answer first).\n"
    "2. **Optimal SQL Query**: Write a complete, runnable query inside a ```sql block with exact required column aliases. "
    "Use simple `SELECT / WHERE / JOIN / GROUP BY / HAVING` for basic queries, and use CTEs (`WITH ... AS`) or "
    "window functions (`DENSE_RANK()`, `ROW_NUMBER()`, `LAG()`, `LEAD()`) when ranking, partitioning, or multi-step aggregation is needed.\n"
    "3. **Edge Cases**: 1 short bullet on how `NULL`s, duplicates, or ties are handled."
)

GENERAL_QA_SYSTEM_PROMPT = (
    "You are an expert AI Copilot that answers ANY question displayed on the user's screen. "
    "The input comes from screen OCR and may contain browser tabs, window titles, or UI buttons—ignore all UI noise "
    "and focus strictly on the question(s) shown on screen (e.g., Multiple-Choice / MCQ, conceptual/theory, "
    "system design, aptitude/math/logic, code output prediction, behavioral/interview, or general knowledge):\n"
    "1. **Direct Answer**: State the exact answer immediately on the first line (for MCQs, give the exact option "
    "letter/number and text; for output/math/fill-in-the-blank, give the exact result; for open-ended questions, "
    "give a crisp 1-sentence core answer).\n"
    "2. **Explanation / Key Points**: Provide 2 to 4 concise, high-impact bullet points explaining the solution or "
    "ready-to-speak talking points (include a short code/query snippet only if the question asks for one)."
)


def detect_coding_problem_type(problem_text: str) -> str:
    """
    Auto-detect whether an OCR-captured coding screen is a SQL problem ('sql')
    or a Data Structures & Algorithms / general coding problem ('dsa').
    """
    text_lower = problem_text.lower()
    # Unambiguous SQL schema / query indicators (avoid generic English words like 'from ' or 'select ')
    sql_strong_indicators = (
        "write an sql query",
        "write a sql query",
        "column name",
        "primary key",
        "foreign key",
        "| column",
        "+---",
        "group by",
        "left join",
        "inner join",
        "right join",
        "having ",
        "varchar",
        "select ",
    )
    has_sql_table_schema = ("column name" in text_lower and "type" in text_lower) or (
        "table:" in text_lower and ("+---" in text_lower or "int" in text_lower or "varchar" in text_lower)
    )
    has_sql_query_syntax = bool(
        re.search(r"\bselect\b.+\bfrom\b", text_lower, re.DOTALL)
        and any(kw in text_lower for kw in ("where ", "group by", "order by", "join ", "having ", "table"))
    )

    dsa_indicators = (
        "class solution",
        "def ",
        "treenode",
        "listnode",
        "vector<",
        "public class",
        "function(",
        "nums",
        "array",
        "string ",
        "linked list",
        "binary tree",
    )
    sql_score = sum(1 for kw in sql_strong_indicators if kw in text_lower)
    if has_sql_table_schema:
        sql_score += 3
    if has_sql_query_syntax:
        sql_score += 2

    dsa_score = sum(1 for kw in dsa_indicators if kw in text_lower)
    if sql_score >= 3 and sql_score > dsa_score:
        return "sql"
    return "dsa"


def retrieve_problem_pattern_hints(problem_text: str, mode: str = "dsa") -> str:
    """
    Zero-RAM (<1ms) pattern matcher that injects bug-free reference templates and
    edge-case invariants for tricky DSA & SQL problems so a 3B model avoids common pitfalls.
    """
    t = problem_text.lower()
    hints: List[str] = []

    if mode == "sql":
        if any(k in t for k in ("second highest", "nth highest", "top 3", "top three", "highest salary", "rank")):
            hints.append(
                "Ranking / Nth Highest Pattern: Use `DENSE_RANK() OVER (PARTITION BY ... ORDER BY salary DESC) AS rnk`. "
                "If the problem requires returning `NULL` when fewer than N distinct values exist, wrap the filtered query in `SELECT MAX(salary) AS ...` or a scalar subquery."
            )
        if any(k in t for k in ("never order", "not in", "who never", "without any", "no order", "didn't", "did not")):
            hints.append(
                "Anti-Join Pattern: Prefer `LEFT JOIN ... ON ... WHERE right_table.id IS NULL` or `NOT EXISTS (...)` instead of `NOT IN` (which fails if NULLs exist)."
            )
        if any(k in t for k in ("consecutive", "three times", "streak", "previous", "yesterday", "rising temperature", "lag")):
            hints.append(
                "Consecutive / Date Comparison Pattern: Use `LAG(col, 1) OVER (ORDER BY id_or_date)` / `LEAD(...)`, or for consecutive calendar days use `DATEDIFF(a.recordDate, b.recordDate) = 1`."
            )
        if any(k in t for k in ("rate", "percentage", "average", "ratio", "confirmation", "cancel", "round(")):
            hints.append(
                "Rate / Conditional Aggregation Pattern: Use `ROUND(AVG(CASE WHEN condition THEN 1.0 ELSE 0.0 END), 2)` or `COALESCE(..., 0)` + `NULLIF(denom, 0)` to avoid integer division or NULL errors."
            )
        if any(k in t for k in ("duplicate", "more than once", "at least", "having")):
            hints.append(
                "Grouping / Duplicates Pattern: Filter aggregated counts with `GROUP BY ... HAVING COUNT(*) > 1` (never put `COUNT()` inside `WHERE`)."
            )
        return "\n".join(f"- {h}" for h in hints[:2])

    # DSA Patterns
    if any(k in t for k in ("subarray sum", "continuous subarray", "sum equals k", "prefix", "divisible by k")):
        hints.append(
            "Prefix Sum + Hash Map Invariant: Initialize `seen = {0: 1}` (or `{0: -1}` for max length) before iterating. Update `curr += x`, check `curr - k in seen` FIRST, then update `seen[curr]`."
        )
    if any(k in t for k in ("longest substring", "sliding window", "minimum window", "at most k", "without repeating")):
        hints.append(
            "Sliding Window Invariant: Expand `right` pointer each step, use a `while` loop to shrink `left` while window constraint is violated, and update best length `right - left + 1`."
        )
    if any(k in t for k in ("next greater", "daily temperature", "histogram", "monotonic", "trapping rain", "span")):
        hints.append(
            "Monotonic Stack Invariant: Store **indices** (not values) on `stack = []`. While `stack and nums[stack[-1]] < nums[i]`, pop index `idx = stack.pop()` and resolve its answer using `i - idx` or `nums[i]`."
        )
    if any(k in t for k in ("rotated sorted", "binary search", "koko", "minimum speed", "capacity to ship", "first and last", "peak")):
        hints.append(
            "Binary Search Invariant: Check `mid = (left + right) // 2` carefully. For monotonic feasibility search, use `while left < right: if feasible(mid): right = mid else: left = mid + 1`."
        )
    if any(k in t for k in ("interval", "overlapping", "meeting room", "merge intervals")):
        hints.append(
            "Intervals Invariant: Sort by start `intervals.sort(key=lambda x: x[0])` to merge (`merged[-1][1] = max(...)`), or sort by end `x[1]` to maximize non-overlapping intervals."
        )
    if any(k in t for k in ("shortest path", "rotting oranges", "word ladder", "level order", "number of islands", "course schedule")):
        hints.append(
            "Graph/Grid BFS & Topo-Sort Invariant: Use `collections.deque`. Mark nodes visited **when enqueuing** (not when popping) to prevent duplicate work/TLE."
        )
    if any(k in t for k in ("coin change", "knapsack", "longest common subsequence", "climb", "rob", "edit distance", "ways to")):
        hints.append(
            "Dynamic Programming Invariant: Define explicit base cases (`n=0`, `n=1`) first; use 1D/2D DP table or `@functools.lru_cache(None)` on helper `dp(i, ...)`."
        )
    return "\n".join(f"- {h}" for h in hints[:2])


def validate_model_for_8gb_ram(model_name: str) -> str:
    """
    Validate that the requested model is a lightweight (<= 3B) text/coding model
    (such as `qwen2.5-coder:3b` or `qwen2.5:3b`).
    Raises ValueError if a VLM or >3B parameter model is specified.
    """
    if not model_name or not model_name.strip():
        raise ValueError("Model name cannot be empty.")

    normalized = model_name.strip().lower()

    for vlm_kw in FORBIDDEN_VLM_KEYWORDS:
        if vlm_kw in normalized:
            raise ValueError(
                f"Model '{model_name}' is a multimodal/vision model. "
                "VLMs are strictly prohibited under the 8 GB RAM limit. Use OCR + qwen2.5-coder:3b instead."
            )

    # Check parameter size suffix if present (e.g., :7b, :8b, :13b, :14b, :32b, :70b)
    size_match = re.search(r"[:\-](\d+(?:\.\d+)?)b\b", normalized)
    if size_match:
        param_billions = float(size_match.group(1))
        if param_billions > 3.2:
            raise ValueError(
                f"Model '{model_name}' ({param_billions}B parameters) exceeds the strict 3B parameter "
                "limit for 8 GB RAM systems. Use 'qwen2.5-coder:3b' or 'qwen2.5:3b'."
            )
    elif normalized not in ALLOWED_MODELS:
        raise ValueError(
            f"Model '{model_name}' does not specify a <=3B parameter tag. "
            f"Allowed models: {sorted(ALLOWED_MODELS)}"
        )

    return model_name.strip()


def auto_select_best_coding_model(
    base_url: str = "http://localhost:11434",
    fallback: str = DEFAULT_MODEL,
    gemini_api_key: Optional[str] = None,
) -> str:
    """
    If a Gemini Free Tier API key is configured, select `gemini-2.5-flash` (0 MB local RAM).
    Otherwise query local Ollama `/api/tags` and select the strongest installed <=3B coding model.
    """
    if load_gemini_api_key(gemini_api_key):
        return DEFAULT_GEMINI_MODEL
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=0.6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            installed = {
                str(m.get("name", "")).strip().lower()
                for m in (data.get("models") or [])
                if m.get("name")
            }
            for preferred in PREFERRED_CODING_MODELS:
                if preferred in installed:
                    return preferred
    except Exception:
        pass
    return fallback


class OllamaClient:
    """
    Streaming HTTP client for Ollama's `/api/chat` endpoint with hard-enforced
    8 GB RAM memory optimizations.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        transport: Optional[Callable[[str, Dict[str, Any]], Iterable[bytes]]] = None,
        gemini_api_key: Optional[str] = None,
    ) -> None:
        self.model = validate_model_for_8gb_ram(model)
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/api/chat"
        self.timeout = timeout
        self._transport = transport
        self._session = requests.Session() if requests is not None else None
        self.gemini_api_key = load_gemini_api_key(gemini_api_key) if transport is None else gemini_api_key

        # Telemetry / test inspection attributes
        self.last_request_payload: Optional[Dict[str, Any]] = None
        self.last_endpoint: Optional[str] = None

    def build_chat_payload(
        self,
        messages: List[Dict[str, str]],
        num_predict: int = 350,
    ) -> Dict[str, Any]:
        """
        Construct the exact JSON payload required for 8 GB RAM optimization:
        - stream: True
        - keep_alive: -1
        - options: {"num_ctx": 2048, "temperature": 0.3, "num_predict": num_predict}
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": -1,
            "options": {
                "num_ctx": 2048,
                "temperature": 0.3,
                "num_predict": num_predict,
            },
        }
        return payload

    def _iter_ndjson_lines(self, url: str, payload: Dict[str, Any]) -> Iterator[bytes]:
        """
        Execute a streaming POST request to Ollama and yield raw NDJSON lines.
        Uses injected transport (if provided), `requests.Session`, or `urllib.request`.
        """
        if self._transport is not None:
            for line in self._transport(url, payload):
                if line:
                    yield line
            return

        if self._session is not None:
            response = self._session.post(
                url,
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()
            try:
                for line in response.iter_lines():
                    if line:
                        yield line
            finally:
                response.close()
            return

        # Fallback to standard library urllib.request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for raw_line in resp:
                stripped = raw_line.strip()
                if stripped:
                    yield stripped

    def _stream_gemini_sse(
        self,
        messages: List[Dict[str, str]],
        num_predict: int = 1200,
    ) -> Iterator[str]:
        """
        Call Google Gemini API (`gemini-3.8-flash` / `gemini-3.6-flash` / `gemini-3.5-flash` / `gemini-3.5-flash-lite`)
        via standard `:generateContent` JSON POST (avoids macOS LibreSSL SSE streaming hangs),
        consuming 0 MB of local LLM RAM.
        """
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        system_texts: List[str] = []
        contents: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if role == "system":
                if text.strip():
                    system_texts.append(text.strip())
            else:
                g_role = "model" if role == "assistant" else "user"
                contents.append({"role": g_role, "parts": [{"text": text}]})

        if system_texts:
            sys_prefix = "SYSTEM INSTRUCTIONS:\n" + "\n".join(system_texts) + "\n\n"
            if contents and contents[0].get("role") == "user":
                contents[0]["parts"][0]["text"] = sys_prefix + contents[0]["parts"][0]["text"]
            else:
                contents.insert(0, {"role": "user", "parts": [{"text": sys_prefix}]})

        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": max(num_predict, 2048),
            },
        }

        primary = self.model if self.model.startswith("gemini-") else DEFAULT_GEMINI_MODEL
        candidate_models = [primary]
        for fallback_m in (
            "gemini-3.8-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
        ):
            if fallback_m not in candidate_models:
                candidate_models.append(fallback_m)

        endpoint_templates = [
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
            "https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent?key={key}",
        ]

        last_err_detail: Optional[str] = None
        req_timeout = max(self.timeout, 60.0)
        for tmpl in endpoint_templates:
            for g_model in candidate_models:
                url = tmpl.format(model=g_model, key=self.gemini_api_key)
                try:
                    if self._session is not None:
                        resp = self._session.post(url, json=body, stream=False, timeout=req_timeout)
                        if resp.status_code >= 400:
                            err_body = resp.text[:300] if resp.text else ""
                            raise RuntimeError(f"HTTP {resp.status_code} ({g_model}): {err_body}")
                        data = resp.json()
                    else:
                        req = urllib.request.Request(
                            url,
                            data=json.dumps(body).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        with urllib.request.urlopen(req, timeout=req_timeout) as resp:
                            data = json.loads(resp.read().decode("utf-8", errors="replace"))

                    yielded_any = False
                    for cand in data.get("candidates", []):
                        parts = cand.get("content", {}).get("parts", [])
                        non_thought_parts = [p for p in parts if not p.get("thought")]
                        target_parts = non_thought_parts if non_thought_parts else parts
                        for part in target_parts:
                            tok = part.get("text", "")
                            if tok:
                                yielded_any = True
                                yield tok
                    if yielded_any:
                        return
                except Exception as exc:
                    last_err_detail = str(exc)
                    continue

        raise RuntimeError(f"Gemini API error: {last_err_detail or 'All Gemini endpoints failed'}")

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        num_predict: int = 350,
    ) -> Iterator[str]:
        """
        Send a chat completion request to Gemini Free Tier API (if `gemini_api_key` is configured)
        or local Ollama `http://localhost:11434/api/chat` and yield token strings as they arrive.
        """
        payload = self.build_chat_payload(messages, num_predict=num_predict)
        self.last_request_payload = payload
        self.last_endpoint = self.chat_url

        if self._transport is None and (self.gemini_api_key or self.model.startswith("gemini-")):
            try:
                yield from self._stream_gemini_sse(messages, num_predict=num_predict)
                return
            except Exception as gemini_err:
                if self.model.startswith("gemini-") or not self.is_server_running():
                    raise gemini_err

        for raw_line in self._iter_ndjson_lines(self.chat_url, payload):
            text_line = (
                raw_line.decode("utf-8", errors="replace")
                if isinstance(raw_line, (bytes, bytearray))
                else str(raw_line)
            ).strip()
            if not text_line:
                continue

            try:
                chunk = json.loads(text_line)
            except json.JSONDecodeError:
                continue

            if "error" in chunk:
                raise RuntimeError(f"Ollama API error: {chunk['error']}")

            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token

            if chunk.get("done", False):
                break

    def stream_interview_answer(
        self,
        question: str,
        context_chunks: Sequence[str],
        job_description: str = "",
        transcript_history: Optional[Sequence[str]] = None,
    ) -> Iterator[str]:
        """
        Build a compact RAG prompt using the top 3 resume chunks, optional job description
        snippet, and rolling 5-turn transcript history, then stream bullet-point answers.
        """
        top_chunks = list(context_chunks)[:3]
        context_block = "\n---\n".join(top_chunks) if top_chunks else "No specific resume context provided."

        parts: List[str] = []
        if job_description and job_description.strip():
            # Truncate JD snippet to preserve the 2048 context window
            jd_snippet = job_description.strip()[:600]
            parts.append(f"TARGET ROLE / JOB DESCRIPTION:\n{jd_snippet}")

        parts.append(f"RELEVANT RESUME HIGHLIGHTS (TOP 3 CHUNKS):\n{context_block}")

        if transcript_history:
            # Strictly cap at last 5 turns
            recent_turns = list(transcript_history)[-5:]
            history_str = "\n".join(f"- {turn}" for turn in recent_turns)
            parts.append(f"RECENT CONVERSATION HISTORY (LAST {len(recent_turns)} TURNS):\n{history_str}")

        parts.append(f"INTERVIEWER QUESTION:\n{question.strip()}\n\nProvide a concise bullet-point response:")

        messages = [
            {"role": "system", "content": INTERVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        yield from self.stream_chat(messages)

    def stream_coding_solution(
        self,
        problem_text: str,
        language: str = "Python",
        mode: str = "auto",
    ) -> Iterator[str]:
        """
        Stream an optimal DSA or SQL coding solution + complexity/edge-case explanation
        for an OCR-extracted coding problem (LeetCode / HackerRank / StrataScratch).
        """
        cleaned_problem = problem_text.strip()[:2500]
        resolved_mode = detect_coding_problem_type(cleaned_problem) if mode == "auto" else mode.lower()

        if resolved_mode == "sql":
            yield from self.stream_sql_solution(cleaned_problem)
            return
        if resolved_mode in ("general", "any", "qa"):
            yield from self.stream_general_question_answer(cleaned_problem)
            return

        pattern_hints = retrieve_problem_pattern_hints(cleaned_problem, mode="dsa")
        hints_block = f"\nREFERENCE INVARIANTS (apply if relevant):\n{pattern_hints}\n" if pattern_hints else ""

        ref_example = get_default_coding_reference_bank().retrieve_reference(cleaned_problem, mode="dsa", top_k=1)
        ref_block = (
            f"\nSIMILAR SOLVED REFERENCE FROM dsa_reference.txt (adapt to exact problem below):\n{ref_example}\n"
            if ref_example
            else ""
        )

        user_prompt = (
            f"Target Language: {language}\n"
            f"{hints_block}"
            f"{ref_block}\n"
            f"SCREEN OCR TEXT (ignore unrelated UI menus/tabs and solve the question):\n{cleaned_problem}\n\n"
            "Verify your logic against any sample Input/Output and edge cases (empty/single element, duplicates, negatives), then provide:\n"
            "1. 1-line direct answer or core approach\n"
            f"2. Complete, simple, bug-free ```{language.lower()} code block (use starter signature if shown)\n"
            "3. 1-line Time & Space Complexity O(...)"
        )
        messages = [
            {"role": "system", "content": CODING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        yield from self.stream_chat(messages, num_predict=768)

    def stream_sql_solution(
        self,
        problem_text: str,
        dialect: str = "PostgreSQL / MySQL",
    ) -> Iterator[str]:
        """
        Stream an optimal SQL query + brief edge-case notes for an OCR-extracted SQL problem.
        """
        cleaned_problem = problem_text.strip()[:2500]
        pattern_hints = retrieve_problem_pattern_hints(cleaned_problem, mode="sql")
        hints_block = f"\nSQL REFERENCE INVARIANTS (apply if relevant):\n{pattern_hints}\n" if pattern_hints else ""

        ref_example = get_default_coding_reference_bank().retrieve_reference(cleaned_problem, mode="sql", top_k=1)
        ref_block = (
            f"\nSIMILAR SOLVED SQL REFERENCE FROM sql_reference.txt (adapt to exact schema/columns below):\n{ref_example}\n"
            if ref_example
            else ""
        )

        user_prompt = (
            f"Target SQL Dialect: {dialect}\n"
            f"{hints_block}"
            f"{ref_block}\n"
            f"SCREEN OCR TEXT (ignore unrelated UI menus/tabs and solve the SQL problem):\n{cleaned_problem}\n\n"
            "Verify exact table/column names and output aliases against the schema, then provide:\n"
            "1. 1-line direct answer or query strategy\n"
            "2. Complete, runnable ```sql query block (keep basic queries simple; use CTEs/window functions when ranking/partitioning is needed)\n"
            "3. 1-line note on edge cases (NULLs, duplicates, ties)"
        )
        messages = [
            {"role": "system", "content": SQL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        yield from self.stream_chat(messages, num_predict=768)

    def stream_general_question_answer(
        self,
        question_text: str,
        context_chunks: Optional[Sequence[str]] = None,
        job_description: str = "",
    ) -> Iterator[str]:
        """
        Stream a direct, accurate answer to ANY question captured from the screen
        (MCQ, conceptual/theory, system design, behavioral/interview, math/aptitude, or code/SQL).
        """
        cleaned_question = question_text.strip()[:2500]
        parts: List[str] = []

        if context_chunks:
            top_chunks = list(context_chunks)[:2]
            if top_chunks:
                parts.append(
                    "CANDIDATE RESUME CONTEXT (use ONLY if the question is behavioral/experience-based):\n"
                    + "\n---\n".join(top_chunks)
                )
        if job_description and job_description.strip():
            parts.append(
                f"TARGET ROLE CONTEXT (optional):\n{job_description.strip()[:400]}"
            )

        parts.append(
            f"SCREEN TEXT (ignore unrelated UI menus/tabs and answer the question in front of the user):\n"
            f"{cleaned_question}\n\n"
            "Provide:\n"
            "1. **Direct Answer**: Exact option/result or 1-sentence core answer first.\n"
            "2. **Key Explanation**: 2-4 concise bullet points (include a code/SQL block if applicable)."
        )

        messages = [
            {"role": "system", "content": GENERAL_QA_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        yield from self.stream_chat(messages, num_predict=600)

    def unload_model(self, model_name: str = "all-minilm") -> bool:
        """
        Immediately unload an auxiliary model (such as `all-minilm` after embedding indexing)
        by sending `keep_alive: 0` so its RAM is freed before the interview session starts.
        """
        url = f"{self.base_url}/api/generate"
        payload = {"model": model_name, "prompt": "", "keep_alive": 0}
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5.0) as _:
                return True
        except Exception:
            return False

    def is_server_running(self) -> bool:
        """Check whether the local Ollama daemon is reachable at `base_url`."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return resp.status == 200
        except Exception:
            return False

"""
test_copilot.py - Unit & Integration Verification Suite for 8 GB RAM AI Interview Copilot

Verifies:
1. Ollama streaming client yields tokens with `stream: True`, `keep_alive: -1`, and
   `options: {"num_ctx": 2048, "temperature": 0.3, "num_predict": 350}`, while blocking >3B and VLM models.
2. RAG pipeline (`RAGEngine`) indexes sample resume + JD and retrieves top-3 relevant chunks in < 50ms.
3. AudioTranscriber enforces `base.en` (`int8`, `cpu`) and caps transcript history at a rolling 5-turn ring buffer.
4. OCRCapture runs explicit garbage collection (`gc.collect()`) after coding problem capture.
5. Total Python process RSS memory stays strictly under 600 MB.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, Iterable, List

from audio_transcriber import AudioTranscriber, is_interview_question
from main import MAX_PYTHON_MEMORY_MB, InterviewCopilotApp, get_process_memory_mb
from ocr_capture import OCRCapture
from ollama_client import OllamaClient, detect_coding_problem_type, validate_model_for_8gb_ram
from overlay_ui import extract_latest_code_block
from rag_engine import RAGEngine


def create_mock_ollama_transport(tokens: List[str]):
    """
    Create a deterministic NDJSON streaming transport that records the request URL
    and payload and yields realistic Ollama `/api/chat` stream lines.
    """
    recorded_calls: List[Dict[str, Any]] = []

    def _transport(url: str, payload: Dict[str, Any]) -> Iterable[bytes]:
        recorded_calls.append({"url": url, "payload": payload})
        for idx, tok in enumerate(tokens):
            is_last = idx == len(tokens) - 1
            chunk = {
                "model": payload["model"],
                "message": {"role": "assistant", "content": tok},
                "done": is_last,
            }
            yield (json.dumps(chunk) + "\n").encode("utf-8")

    return _transport, recorded_calls


class TestInterviewCopilot8GB(unittest.TestCase):
    """Comprehensive verification tests for the 8 GB RAM Interview Copilot."""

    def test_01_ollama_streaming_parameters_and_num_ctx_2048(self) -> None:
        """
        Verify OllamaClient yields streaming tokens and explicitly passes:
        - stream: True
        - keep_alive: -1
        - options: {"num_ctx": 2048, "temperature": 0.3, "num_predict": 350}
        """
        expected_tokens = ["• ", "Scaled ", "Kafka ", "pipeline ", "to ", "85,000 RPS ", "at 14ms p99."]
        transport, calls = create_mock_ollama_transport(expected_tokens)

        client = OllamaClient(model="qwen2.5:3b", transport=transport)
        yielded = list(
            client.stream_interview_answer(
                question="How have you scaled high-throughput backend systems?",
                context_chunks=["[EXPERIENCE] Architected Kafka & FastAPI pipeline at 85,000 RPS."],
                job_description="Staff Backend Engineer",
                transcript_history=["[Interviewer] Welcome to the interview."],
            )
        )

        self.assertEqual(yielded, expected_tokens)
        self.assertEqual("".join(yielded), "• Scaled Kafka pipeline to 85,000 RPS at 14ms p99.")
        self.assertEqual(len(calls), 1)

        sent_url = calls[0]["url"]
        sent_payload = calls[0]["payload"]
        self.assertEqual(sent_url, "http://localhost:11434/api/chat")
        self.assertEqual(sent_payload["model"], "qwen2.5:3b")
        self.assertIs(sent_payload["stream"], True)
        self.assertEqual(sent_payload["keep_alive"], -1)
        self.assertEqual(
            sent_payload["options"],
            {
                "num_ctx": 2048,
                "temperature": 0.3,
                "num_predict": 350,
            },
        )

    def test_02_model_guard_rejects_large_and_vlm_models(self) -> None:
        """Verify >3B models and multimodal vision models are strictly blocked."""
        self.assertEqual(validate_model_for_8gb_ram("qwen2.5:3b"), "qwen2.5:3b")
        self.assertEqual(validate_model_for_8gb_ram("llama3.2:3b"), "llama3.2:3b")

        for forbidden_model in ("qwen2.5:7b", "llama3.1:8b", "llava:7b", "moondream:1.8b", "qwen2.5-vl:3b"):
            with self.subTest(model=forbidden_model):
                with self.assertRaises(ValueError):
                    validate_model_for_8gb_ram(forbidden_model)

    def test_03_rag_pipeline_retrieves_relevant_context_under_50ms(self) -> None:
        """
        Verify RAGEngine indexes the sample resume & job description and retrieves
        top-3 relevant chunks in strictly under 50 milliseconds.
        """
        base_dir = Path(__file__).resolve().parent
        resume_path = base_dir / "resume.txt"
        jd_path = base_dir / "job_description.txt"

        rag = RAGEngine()
        chunk_count = rag.load_documents(resume_path=resume_path, job_description_path=jd_path)
        self.assertGreater(chunk_count, 0, "RAGEngine should index at least 1 chunk from resume.txt")

        queries_and_expected_keywords = [
            ("How did you reduce memory usage for ONNX inference on Kubernetes?", "480 mb"),
            ("Tell me about your Kafka and Redis real-time fraud pipeline throughput.", "85,000 rps"),
            ("How did you handle PostgreSQL connection pool exhaustion during traffic spikes?", "pgbouncer"),
        ]

        for query, expected_kw in queries_and_expected_keywords:
            t0 = time.perf_counter()
            top_chunks = rag.retrieve(query, top_k=3)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            self.assertLessEqual(len(top_chunks), 3)
            self.assertGreater(len(top_chunks), 0)
            self.assertLess(
                elapsed_ms,
                50.0,
                f"RAG retrieval took {elapsed_ms:.2f} ms, exceeding the 50 ms limit!",
            )
            combined_retrieved = " ".join(top_chunks).lower()
            self.assertIn(
                expected_kw,
                combined_retrieved,
                f"Expected '{expected_kw}' in top-3 retrieved chunks for query: {query}",
            )

    def test_04_rag_pdf_extraction_support(self) -> None:
        """Verify RAGEngine can load and index a `.pdf` resume file."""
        minimal_pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
            b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
            b"3 0 obj << /Type /Page /Parent 2 0 R /Contents 4 0 R >> endobj\n"
            b"4 0 obj << /Length 88 >>\n"
            b"stream\n"
            b"BT /F1 12 Tf 72 712 Td (Distributed Systems Engineer Kubernetes Rust gRPC 99.99% SLA) Tj ET\n"
            b"endstream\n"
            b"endobj\n"
            b"xref\n0 5\n0000000000 65535 f \ntrailer << /Root 1 0 R /Size 5 >>\n%%EOF\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(minimal_pdf)
            tmp_pdf_path = tmp.name

        try:
            rag = RAGEngine()
            count = rag.load_documents(resume_path=tmp_pdf_path)
            self.assertGreater(count, 0)
            results = rag.retrieve("Tell me about your Rust and gRPC experience", top_k=3)
            self.assertTrue(any("grpc" in c.lower() for c in results))
        finally:
            if os.path.exists(tmp_pdf_path):
                os.unlink(tmp_pdf_path)

    def test_05_audio_transcriber_5_turn_ring_buffer_and_int8_cpu_enforcement(self) -> None:
        """
        Verify AudioTranscriber enforces `compute_type='int8'` and `device='cpu'`
        and keeps transcript history strictly in a rolling ring buffer of the last 5 turns.
        """
        with self.assertRaises(ValueError):
            AudioTranscriber(compute_type="float16")
        with self.assertRaises(ValueError):
            AudioTranscriber(device="cuda")

        detected_questions: List[str] = []
        transcriber = AudioTranscriber(
            on_question_detected=lambda q, _hist: detected_questions.append(q),
            lazy_load_model=True,
        )

        # Record 8 turns; only the last 5 should remain in the ring buffer
        for i in range(1, 9):
            transcriber.record_turn(f"Statement number {i}", speaker="Candidate")

        recent = transcriber.get_recent_turns()
        self.assertEqual(len(recent), 5)
        self.assertEqual(recent[0], "[Candidate] Statement number 4")
        self.assertEqual(recent[-1], "[Candidate] Statement number 8")

        # Record an interviewer question
        transcriber.record_turn(
            "Can you walk me through how you designed the Kafka pipeline?",
            speaker="Interviewer",
        )
        self.assertEqual(len(transcriber.get_recent_turns()), 5)
        self.assertEqual(len(detected_questions), 1)
        self.assertTrue(is_interview_question(detected_questions[0]))

    def test_06_ocr_coding_capture_and_garbage_collection(self) -> None:
        """
        Verify OCRCapture solves a coding problem via OllamaClient with `num_ctx: 2048`
        and runs explicit garbage collection (`cleanup_count` incremented).
        """
        solution_tokens = [
            "1. **Approach**: Use a hash map to store complements in a single pass.\n",
            "2. **Optimal Solution**:\n```python\ndef two_sum(nums, target):\n    seen = {}\n    for i, x in enumerate(nums):\n        if target - x in seen:\n            return [seen[target - x], i]\n        seen[x] = i\n```\n",
            "3. **Complexity**: Time O(n), Space O(n).",
        ]
        transport, calls = create_mock_ollama_transport(solution_tokens)
        client = OllamaClient(model="qwen2.5:3b", transport=transport)
        ocr = OCRCapture(release_engine_after_capture=True)

        initial_cleanups = ocr.cleanup_count
        problem_text, stream = ocr.capture_and_solve_coding_problem(
            ollama_client=client,
            fallback_problem_text="Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
        )
        output = "".join(stream)

        self.assertIn("two_sum", output)
        self.assertIn("Time O(n)", output)
        self.assertGreater(ocr.cleanup_count, initial_cleanups)
        self.assertEqual(calls[0]["payload"]["options"]["num_ctx"], 2048)

    def test_07_total_python_process_memory_under_600mb(self) -> None:
        """
        Verify total Python process memory (RSS) with all copilot modules active
        stays strictly below 600 MB.
        """
        base_dir = Path(__file__).resolve().parent
        transport, _calls = create_mock_ollama_transport(["• Bullet 1\n", "• Bullet 2\n"])
        client = OllamaClient(model="qwen2.5:3b", transport=transport)

        app = InterviewCopilotApp(
            resume_path=base_dir / "resume.txt",
            jd_path=base_dir / "job_description.txt",
            ollama_client=client,
        )

        # Simulate end-to-end interview question + streaming response
        tokens = list(app.answer_question_stream("How did you optimize ONNX memory on Kubernetes?"))
        self.assertTrue(len(tokens) > 0)

        rss_mb = get_process_memory_mb()
        self.assertLess(
            rss_mb,
            MAX_PYTHON_MEMORY_MB,
            f"Python process RSS ({rss_mb:.2f} MB) exceeded the 600 MB limit!",
        )

    def test_08_sql_and_dsa_specialization(self) -> None:
        """
        Verify SQL vs DSA auto-detection, OCR UI noise stripping, SQL prompt routing,
        and 1-click code block extraction from markdown responses.
        """
        sql_problem = (
            "Table: Employee\n"
            "+-------------+---------+\n"
            "| Column Name | Type    |\n"
            "+-------------+---------+\n"
            "| id          | int     |\n"
            "| salary      | int     |\n"
            "Write a solution to find the second highest distinct salary from the Employee table."
        )
        dsa_problem = (
            "class Solution:\n"
            "    def lengthOfLongestSubstring(self, s: str) -> int:\n"
            "Given a string s, find the length of the longest substring without repeating characters.\n"
            "Constraints: 0 <= s.length <= 5 * 10^4"
        )

        self.assertEqual(detect_coding_problem_type(sql_problem), "sql")
        self.assertEqual(detect_coding_problem_type(dsa_problem), "dsa")

        # Basic coding questions containing English words "from" and "select" must NOT be misclassified as SQL
        basic_dsa_question = (
            "Write a solution to find the largest number from a list of integers "
            "and select the index of the first palindrome."
        )
        self.assertEqual(detect_coding_problem_type(basic_dsa_question), "dsa")

        # Verify OCR UI noise cleaner removes browser/editor chrome while preserving problem text
        noisy_ocr = "Run\nSubmit\nhttps://leetcode.com/problems/second-highest-salary/\n" + sql_problem + "\nAccepted"
        cleaned = OCRCapture.clean_coding_ocr_text(noisy_ocr)
        self.assertNotIn("https://leetcode.com", cleaned)
        self.assertIn("second highest distinct salary", cleaned)

        # Verify SQL mode routes to SQL system prompt and injects sql_reference.txt example
        sql_tokens = [
            "```sql\nWITH Ranked AS (\n  SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk\n  FROM Employee\n)\nSELECT MAX(salary) AS SecondHighestSalary FROM Ranked WHERE rnk = 2;\n```\n",
            "- Uses DENSE_RANK() to handle duplicate salaries cleanly.\n",
            "- MAX(salary) returns NULL when fewer than 2 distinct salaries exist.",
        ]
        transport, calls = create_mock_ollama_transport(sql_tokens)
        client = OllamaClient(model="qwen2.5-coder:3b", transport=transport)
        output = "".join(client.stream_coding_solution(sql_problem, mode="auto"))
        self.assertIn("DENSE_RANK()", output)
        system_msg = calls[0]["payload"]["messages"][0]["content"]
        user_msg = calls[0]["payload"]["messages"][1]["content"]
        self.assertIn("SQL", system_msg)
        self.assertIn("sql_reference.txt", user_msg)

        # Verify DSA mode injects dsa_reference.txt example, enforces <=3B, and uses num_predict=768
        self.assertEqual(validate_model_for_8gb_ram("qwen2.5-coder:3b"), "qwen2.5-coder:3b")
        with self.assertRaises(ValueError):
            validate_model_for_8gb_ram("qwen2.5-coder:7b")
        dsa_transport, dsa_calls = create_mock_ollama_transport(["```python\npass\n```"])
        dsa_client = OllamaClient(model="qwen2.5-coder:3b", transport=dsa_transport)
        _ = "".join(dsa_client.stream_coding_solution(dsa_problem, mode="dsa"))
        self.assertIn("dsa_reference.txt", dsa_calls[0]["payload"]["messages"][1]["content"])
        self.assertEqual(dsa_calls[0]["payload"]["options"]["num_predict"], 768)

        # Verify extract_latest_code_block extracts only the runnable query/code
        extracted_code = extract_latest_code_block(output)
        self.assertTrue(extracted_code.startswith("WITH Ranked AS"))
        self.assertNotIn("```", extracted_code)

    def test_09_general_question_answering(self) -> None:
        """
        Verify 'Answer Any' general screen question mode answers any MCQ, conceptual,
        system design, or behavioral question on screen and wires into StealthOverlayUI.
        """
        qa_tokens = [
            "1. **Direct Answer**: Option B — Consistency and Partition Tolerance (CP).\n",
            "2. **Key Explanation**:\n- In the CAP theorem, during a network partition, a distributed store must choose between Consistency and Availability.",
        ]
        transport, calls = create_mock_ollama_transport(qa_tokens)
        client = OllamaClient(model="qwen2.5:3b", transport=transport)

        any_question = (
            "Question 4: Under the CAP theorem, which two guarantees does a ZooKeeper/etcd quorum prioritize "
            "when a network partition occurs?\n"
            "A) Availability & Partition Tolerance\n"
            "B) Consistency & Partition Tolerance\n"
            "C) Consistency & Availability"
        )
        output = "".join(client.stream_general_question_answer(any_question))
        self.assertIn("Option B", output)
        self.assertEqual(len(calls), 1)
        self.assertIn("answers ANY question", calls[0]["payload"]["messages"][0]["content"])
        self.assertIn("CAP theorem", calls[0]["payload"]["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

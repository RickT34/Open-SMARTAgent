#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import error, request


SEARCH_PATTERNS = [
    re.compile(r"\(tool: Search\)\n([^\n]+)"),
    re.compile(r"\[\[Search\]\][^\n]*\n([^\n]+)"),
    re.compile(r"Search\((.+?)\)"),
]
FINAL_PATTERNS = [
    re.compile(r"### Final Response\s*\n([\s\S]*?)\s*$"),
    re.compile(r"Final Answer:\s*([\s\S]*?)\s*$"),
]


@dataclass
class QueryEntry:
    query: str
    task: str
    final_answer: str
    gold_reasoning: str
    source_file: str


@dataclass
class OrganicItem:
    title: str
    link: str
    date: str
    snippet: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Serper-compatible sidecar for SMART TIME.")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset JSON used to build a query index. Can be passed multiple times.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the local sidecar server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the local sidecar server.",
    )
    parser.add_argument(
        "--secret-file",
        default="secret.json",
        help="OpenAI-compatible secret file with api_key and base_url.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model name exposed by your local OpenAI-compatible server.",
    )
    parser.add_argument(
        "--cache-file",
        default="serper_sidecar/cache/generated_search_results.json",
        help="Where generated search responses are cached.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many organic results to generate per query.",
    )
    return parser.parse_args()


def extract_task(text: str) -> str:
    if "### Task" not in text:
        return text.strip()
    return text.split("### Task", 1)[1].split("###", 1)[0].strip()


def extract_query(text: str) -> str | None:
    for pattern in SEARCH_PATTERNS:
        matched = pattern.search(text)
        if matched:
            return matched.group(1).strip()
    return None


def extract_final_answer(text: str) -> str:
    for pattern in FINAL_PATTERNS:
        matched = pattern.search(text)
        if matched:
            return matched.group(1).strip()
    return ""


def load_secret(secret_file: str) -> tuple[str, str]:
    with open(secret_file, "r", encoding="utf-8") as f:
        secret = json.load(f)
    api_key = secret.get("api_key", "local")
    base_url = secret.get("base_url")
    if not base_url:
        raise ValueError(f"`base_url` is missing in {secret_file}")
    return api_key, base_url.rstrip("/")


def load_query_index(dataset_files: list[str]) -> dict[str, QueryEntry]:
    index: dict[str, QueryEntry] = {}
    for dataset_file in dataset_files:
        with open(dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            input_text = item.get("input", "")
            output_text = item.get("output", "")
            task = extract_task(input_text)
            query = extract_query(output_text) or extract_query(input_text)
            if not query:
                continue
            entry = QueryEntry(
                query=query,
                task=task,
                final_answer=extract_final_answer(output_text),
                gold_reasoning=output_text.strip(),
                source_file=dataset_file,
            )
            index[query] = entry
    return index


class LLMBackedSearchGenerator:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        cache_file: str,
        top_k: int,
        query_index: dict[str, QueryEntry],
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.cache_file = Path(cache_file)
        self.top_k = top_k
        self.query_index = query_index
        self.lock = threading.Lock()
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_file.exists():
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def generate(self, query: str) -> dict[str, Any]:
        with self.lock:
            if query in self.cache:
                return self.cache[query]

        entry = self.query_index.get(query)
        payload = self._call_llm(query, entry)
        self._validate_payload(payload, query)

        with self.lock:
            self.cache[query] = payload
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        return payload

    def _call_llm(self, query: str, entry: QueryEntry | None) -> dict[str, Any]:
        system_prompt = (
            "You simulate a Serper-like web search API for benchmark replay.\n"
            "Return only strict JSON with this schema:\n"
            "{\"organic\": [{\"title\": str, \"link\": str, \"date\": str, \"snippet\": str}, ...]}\n"
            f"Generate exactly {self.top_k} organic results.\n"
            "The snippets should look like plausible search snippets and support answering the search query.\n"
            "Do not mention that the results are simulated, benchmarked, or generated from gold reasoning.\n"
            "Use concise snippets. Dates should look like 'Nov 21, 2024' or 'N/A'.\n"
            "Every result must contain title, link, date, and snippet."
        )
        user_prompt = {
            "search_query": query,
            "origin_task": entry.task if entry else "",
            "gold_final_answer": entry.final_answer if entry else "",
            "gold_reasoning": entry.gold_reasoning if entry else "",
            "notes": (
                "Use the origin task and gold reasoning only as hidden guidance to infer what the search results "
                "should reveal. The returned JSON should look like normal web search output."
            ),
        }

        req_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
            ],
            "temperature": 0,
        }
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(req_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=180) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTPError {exc.code}: {body}") from exc
        except Exception as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        content = raw["choices"][0]["message"]["content"].strip()
        json_text = self._extract_json_text(content)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM did not return valid JSON: {content}") from exc

    @staticmethod
    def _extract_json_text(text: str) -> str:
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 3:
                return parts[1].replace("json", "", 1).strip()
        return text

    @staticmethod
    def _validate_payload(payload: dict[str, Any], query: str) -> None:
        if not isinstance(payload, dict) or "organic" not in payload:
            raise RuntimeError(f"Generated payload for query {query!r} misses `organic`.")
        organic = payload["organic"]
        if not isinstance(organic, list) or not organic:
            raise RuntimeError(f"Generated payload for query {query!r} has empty `organic`.")
        for idx, item in enumerate(organic, start=1):
            for field in ("title", "link", "date", "snippet"):
                if field not in item or not str(item[field]).strip():
                    raise RuntimeError(
                        f"Generated organic item #{idx} for query {query!r} misses field `{field}`."
                    )


class SearchHandler(BaseHTTPRequestHandler):
    generator: LLMBackedSearchGenerator

    def do_POST(self) -> None:
        if self.path != "/search":
            self._json_response(404, {"error": "not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length).decode("utf-8")
            body = json.loads(raw_body or "{}")
            query = body.get("q", "").strip()
            if not query:
                self._json_response(400, {"error": "missing query field `q`"})
                return

            payload = self.generator.generate(query)
            self._json_response(200, payload)
        except Exception as exc:
            self._json_response(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[serper-sidecar] " + fmt % args + "\n")

    def _json_response(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    args = parse_args()
    api_key, base_url = load_secret(args.secret_file)
    query_index = load_query_index(args.dataset)
    if not query_index:
        raise RuntimeError("No search query was extracted from the provided dataset files.")

    generator = LLMBackedSearchGenerator(
        api_key=api_key,
        base_url=base_url,
        model=args.model,
        cache_file=args.cache_file,
        top_k=args.top_k,
        query_index=query_index,
    )
    SearchHandler.generator = generator

    server = ThreadingHTTPServer((args.host, args.port), SearchHandler)
    print(
        json.dumps(
            {
                "status": "ready",
                "listen": f"http://{args.host}:{args.port}",
                "indexed_queries": len(query_index),
                "model": args.model,
                "cache_file": str(Path(args.cache_file).resolve()),
            },
            ensure_ascii=False,
        )
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import error, parse, request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Serper-compatible sidecar backed by an OpenAI-compatible model."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Optional legacy argument kept for compatibility. It is ignored in query-only mode.",
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
        default="cache/generated_search_results.json",
        help="Where generated search responses are cached.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Preferred number of organic results to generate.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="How many times to retry when the model does not return valid JSON.",
    )
    return parser.parse_args()


def load_secret(secret_file: str) -> tuple[str, str]:
    with open(secret_file, "r", encoding="utf-8") as f:
        secret = json.load(f)
    api_key = secret.get("api_key", "local")
    base_url = secret.get("base_url")
    if not base_url:
        raise ValueError(f"`base_url` is missing in {secret_file}")
    return api_key, base_url.rstrip("/")


class LLMBackedSearchGenerator:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        cache_file: str,
        top_k: int,
        max_retries: int,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.cache_file = Path(cache_file)
        self.top_k = top_k
        self.max_retries = max_retries
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

        payload = self._generate_uncached(query)

        with self.lock:
            self.cache[query] = payload
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        return payload

    def _generate_uncached(self, query: str) -> dict[str, Any]:
        messages = self._build_messages(query)
        last_content = ""

        for attempt in range(1, self.max_retries + 1):
            content = self._chat(messages)
            last_content = content
            try:
                payload = self._parse_payload(content)
                payload = self._normalize_payload(payload, query)
                self._validate_payload(payload, query)
                return payload
            except Exception as exc:
                if attempt >= self.max_retries:
                    break
                messages.extend(
                    [
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": (
                                "Your previous reply was not valid JSON for the required schema. "
                                "Return only valid JSON with an `organic` list. No markdown fences, no commentary."
                            ),
                        },
                    ]
                )

        return self._fallback_payload(query, last_content)

    def _build_messages(self, query: str) -> list[dict[str, str]]:
        system_prompt = (
            "You simulate a Serper-like search backend for an agent benchmark.\n"
            "Use only your internal knowledge to answer the search query.\n"
            "Return only strict JSON with this schema:\n"
            "{\"organic\": [{\"title\": str, \"link\": str, \"date\": str, \"snippet\": str}, ...]}\n"
            f"Try to produce {self.top_k} organic results.\n"
            "Each result should look like a plausible web search result snippet that helps answer the query.\n"
            "Do not say the results are simulated. Do not use markdown. Do not add explanations outside JSON.\n"
            "If you are uncertain, reflect that in the snippet wording instead of refusing."
        )
        user_prompt = {
            "search_query": query,
            "instructions": (
                "Generate plausible search results for this query based on your internal knowledge. "
                "The result list should make it easy for a downstream agent to infer the answer."
            ),
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ]

    def _chat(self, messages: list[dict[str, str]]) -> str:
        req_body = {
            "model": self.model,
            "messages": messages,
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

        return raw["choices"][0]["message"]["content"].strip()

    def _parse_payload(self, text: str) -> dict[str, Any]:
        json_text = self._extract_json_text(text)
        return json.loads(json_text)

    @staticmethod
    def _extract_json_text(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            parts = stripped.split("```")
            if len(parts) >= 3:
                stripped = parts[1]
                if stripped.lstrip().startswith("json"):
                    stripped = stripped.lstrip()[4:]
                stripped = stripped.strip()

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            return stripped[start:end + 1]
        return stripped

    def _normalize_payload(self, payload: dict[str, Any], query: str) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise RuntimeError(f"Model output is not a JSON object for query {query!r}.")

        organic = payload.get("organic")
        if isinstance(organic, dict):
            organic = [organic]
        if not isinstance(organic, list):
            raise RuntimeError(f"Model output misses a valid `organic` list for query {query!r}.")

        normalized = []
        for idx, item in enumerate(organic, start=1):
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "title": str(item.get("title", f"Result {idx} for {query}")).strip() or f"Result {idx} for {query}",
                    "link": str(item.get("link", self._placeholder_link(query, idx))).strip() or self._placeholder_link(query, idx),
                    "date": str(item.get("date", "N/A")).strip() or "N/A",
                    "snippet": str(item.get("snippet", "")).strip() or f"No snippet available for {query}.",
                }
            )

        return {"organic": normalized}

    @staticmethod
    def _placeholder_link(query: str, idx: int) -> str:
        slug = parse.quote_plus(query[:80])
        return f"https://search.local/result/{idx}?q={slug}"

    def _fallback_payload(self, query: str, raw_text: str) -> dict[str, Any]:
        snippet = raw_text.strip().replace("\n", " ")
        if not snippet:
            snippet = (
                "The backend model did not return valid JSON, so this fallback result exposes "
                "the model's direct answer path instead of structured search evidence."
            )
        snippet = snippet[:600]
        return {
            "organic": [
                {
                    "title": f"Direct answer for: {query}",
                    "link": self._placeholder_link(query, 1),
                    "date": "N/A",
                    "snippet": snippet,
                }
            ]
        }

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
            fallback_payload = {
                "organic": [
                    {
                        "title": f"Search backend error for: {body.get('q', '') if 'body' in locals() else ''}".strip(),
                        "link": "https://search.local/error",
                        "date": "N/A",
                        "snippet": str(exc),
                    }
                ]
            }
            self._json_response(200, fallback_payload)

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

    generator = LLMBackedSearchGenerator(
        api_key=api_key,
        base_url=base_url,
        model=args.model,
        cache_file=args.cache_file,
        top_k=args.top_k,
        max_retries=args.max_retries,
    )
    SearchHandler.generator = generator

    server = ThreadingHTTPServer((args.host, args.port), SearchHandler)
    print(
        json.dumps(
            {
                "status": "ready",
                "listen": f"http://{args.host}:{args.port}",
                "mode": "query_only_internal_knowledge",
                "model": args.model,
                "cache_file": str(Path(args.cache_file).resolve()),
                "ignored_datasets": len(args.dataset),
            },
            ensure_ascii=False,
        )
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

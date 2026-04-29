#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Quick smoke test for trsexp/serpar_server.py"
    )
    parser.add_argument(
        "--server-script",
        default=str(repo_root / "trsexp" / "serpar_server.py"),
        help="Path to serpar_server.py",
    )
    parser.add_argument(
        "--dataset",
        default=str(repo_root / "data_inference" / "domain_time_smart.json"),
        help="Dataset used both for indexing and for picking a sample query",
    )
    parser.add_argument(
        "--secret-file",
        default=str(repo_root / "secret.json"),
        help="OpenAI-compatible secret.json used by the sidecar",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model name passed to serpar_server.py",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the local Serper-compatible server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the local Serper-compatible server",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=30,
        help="Seconds to wait for the sidecar process to report ready",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=240,
        help="Seconds to wait for /search to return",
    )
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="Keep the sidecar running after the test finishes",
    )
    return parser.parse_args()


def print_step(msg: str) -> None:
    print(f"[test] {msg}", flush=True)


def load_secret(secret_file: str) -> tuple[str, str]:
    with open(secret_file, "r", encoding="utf-8") as f:
        secret = json.load(f)
    api_key = secret.get("api_key", "local")
    base_url = secret.get("base_url", "").rstrip("/")
    if not base_url:
        raise RuntimeError(f"`base_url` is missing in {secret_file}")
    return api_key, base_url


def load_sample_query(dataset_path: str) -> str:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        output = item.get("output", "")
        marker = "(tool: Search)\n"
        if marker in output:
            return output.split(marker, 1)[1].splitlines()[0].strip()
    raise RuntimeError(f"No search query found in dataset: {dataset_path}")


def openai_chat(api_key: str, base_url: str, model: str, timeout: int) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer in one word."},
            {"role": "user", "content": "Say pong"},
        ],
        "temperature": 0,
    }
    req = request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    return raw["choices"][0]["message"]["content"].strip()


def start_sidecar(args: argparse.Namespace, repo_root: Path) -> subprocess.Popen[str]:
    cmd = [
        sys.executable,
        "-u",
        args.server_script,
        "--dataset",
        args.dataset,
        "--secret-file",
        args.secret_file,
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    print_step(f"starting sidecar: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def wait_until_ready(proc: subprocess.Popen[str], timeout: int) -> dict[str, Any]:
    start_time = time.time()
    stdout_lines: list[str] = []
    while time.time() - start_time < timeout:
        if proc.poll() is not None:
            stderr = proc.stderr.read() if proc.stderr else ""
            stdout = "".join(stdout_lines)
            raise RuntimeError(
                "sidecar exited before becoming ready.\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )

        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            stdout_lines.append(line)
            line = line.strip()
            print_step(f"sidecar stdout: {line}")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("status") == "ready":
                return payload
        else:
            time.sleep(0.2)

    raise RuntimeError(
        f"sidecar did not report ready within {timeout}s."
    )


def post_search(host: str, port: int, query: str, timeout: int) -> dict[str, Any]:
    body = {"q": query, "tbs": "qdr:y"}
    req = request.Request(
        f"http://{host}:{port}/search",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def validate_search_payload(payload: dict[str, Any]) -> None:
    if "organic" not in payload:
        raise RuntimeError(f"missing `organic` in search response: {payload}")
    organic = payload["organic"]
    if not isinstance(organic, list) or len(organic) == 0:
        raise RuntimeError(f"`organic` is empty or invalid: {payload}")

    first = organic[0]
    for field in ("title", "link", "date", "snippet"):
        if field not in first or not str(first[field]).strip():
            raise RuntimeError(f"first result misses `{field}`: {first}")


def stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.server_script).resolve().parents[1]
    proc: subprocess.Popen[str] | None = None

    try:
        print_step(f"repo root: {repo_root}")
        print_step(f"dataset: {args.dataset}")
        print_step(f"secret file: {args.secret_file}")

        api_key, base_url = load_secret(args.secret_file)
        sample_query = load_sample_query(args.dataset)
        print_step(f"sample query: {sample_query}")

        print_step("checking OpenAI-compatible backend")
        chat_reply = openai_chat(api_key, base_url, args.model, timeout=60)
        print_step(f"backend reply preview: {chat_reply[:120]!r}")

        proc = start_sidecar(args, repo_root)
        ready_payload = wait_until_ready(proc, args.startup_timeout)
        print_step(f"sidecar ready: {json.dumps(ready_payload, ensure_ascii=False)}")

        print_step("sending /search request")
        started = time.time()
        search_payload = post_search(args.host, args.port, sample_query, args.request_timeout)
        elapsed = time.time() - started
        validate_search_payload(search_payload)

        organic = search_payload["organic"]
        first = organic[0]
        print_step(f"/search returned {len(organic)} organic results in {elapsed:.1f}s")
        print_step(
            "first result: "
            + json.dumps(
                {
                    "title": first["title"],
                    "date": first["date"],
                    "link": first["link"],
                    "snippet_preview": first["snippet"][:160],
                },
                ensure_ascii=False,
            )
        )
        print_step("PASS")

        if args.keep_server and proc.poll() is None:
            print_step(
                f"keeping sidecar alive at http://{args.host}:{args.port} "
                f"(pid={proc.pid})"
            )
            return 0

        return 0
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print_step(f"HTTPError {exc.code}: {body}")
        return 1
    except Exception as exc:
        print_step(f"FAIL: {exc}")
        return 1
    finally:
        if proc is not None and not args.keep_server:
            stop_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())

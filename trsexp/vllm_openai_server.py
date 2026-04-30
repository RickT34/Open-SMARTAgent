import importlib
import json
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os


# =========================
# 你主要只需要改这里
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

API_KEYS = ["local"]
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 9000
REQUEST_TIMEOUT = 6000
WAIT_BACKEND_TIMEOUT = 18000
DIR_DATA = "/share/trsdata/trsdata"

BACKENDS = {
    "big": {
        "source_model": f"{DIR_DATA}/models/Qwen3-32B-AWQ",
        "served_model_name": "qwen3-32b",
        # "source_model": f"{DIR_DATA}/models/Qwen2.5-7B-Instruct",
        # "served_model_name": "qwen2.5-7b",
        # 对 Qwen3/Qwen3.5 这类支持 thinking 开关的模型，
        # 设为 False 可默认关闭长推理；None 表示不干预模型默认行为。
        "enable_thinking": None,
        "host": "127.0.0.1",
        "port": 8001,
        "vllm_args": [
            "--gpu-memory-utilization", "0.95",
            "--max-model-len", "32768",
            "--max-num-seqs", "64",
            "--tensor-parallel-size", str(len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
        ],
    },
}

MODEL_ALIASES = {
    "gpt-4o": {
        "backend": "big",
        # 别名级开关优先级高于 backend 默认值。
        "enable_thinking": None,
        "extra_body": {},
    },
    "gpt-4o-mini": {
        "backend": "big",
        "enable_thinking": None,
        "extra_body": {},
    },
}



import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


@dataclass
class BackendSpec:
    name: str
    source_model: str
    served_model_name: str
    host: str
    port: int
    enable_thinking: bool | None = None
    vllm_args: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def is_local_path(self) -> bool:
        source = self.source_model
        return source.startswith("/") or source.startswith("./") or source.startswith("../") or source.startswith("~")


@dataclass
class AliasSpec:
    alias: str
    backend_name: str
    enable_thinking: bool | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProxyConfig:
    api_keys: set[str]
    request_timeout: float
    backends: dict[str, BackendSpec]
    aliases: dict[str, AliasSpec]


class ManagedProcess:
    def __init__(self, name: str, command: list[str]):
        self.name = name
        self.command = command
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        print(f"[start] 启动 {self.name}: {' '.join(self.command)}")
        self.process = subprocess.Popen(self.command, cwd=str(Path(__file__).parent))

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def terminate(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        print(f"[stop] 关闭 {self.name}")
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"[stop] 强制结束 {self.name}")
            self.process.kill()
            self.process.wait(timeout=5)


def resolve_model_source(source_model: str) -> str:
    path_like = (
        source_model.startswith("/")
        or source_model.startswith("./")
        or source_model.startswith("../")
        or source_model.startswith("~")
    )
    if not path_like:
        return source_model

    candidate = Path(source_model).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(
            "你配置的是本地模型路径，但目录不存在:\n"
            f"  {candidate}\n"
            "请确认这是不是实际模型目录，而不是别的机器上的路径、挂载前路径，或父目录。"
        )
    if not candidate.is_dir():
        raise FileNotFoundError(
            "你配置的本地模型路径不是目录:\n"
            f"  {candidate}"
        )

    required_files = ["config.json"]
    missing_files = [name for name in required_files if not (candidate / name).exists()]
    if missing_files:
        raise FileNotFoundError(
            "这个目录看起来不像 Hugging Face 模型目录，至少缺少这些文件:\n"
            f"  {', '.join(missing_files)}\n"
            f"当前目录: {candidate}\n"
            "通常你需要把 source_model 指向真正包含 config.json / tokenizer.json / model.safetensors 的那一层目录。"
        )

    return str(candidate.resolve())


def build_config() -> ProxyConfig:
    backends: dict[str, BackendSpec] = {}
    for name, raw in BACKENDS.items():
        backends[name] = BackendSpec(
            name=name,
            source_model=resolve_model_source(raw["source_model"]),
            served_model_name=raw["served_model_name"],
            host=raw.get("host", "127.0.0.1"),
            port=int(raw["port"]),
            enable_thinking=raw.get("enable_thinking"),
            vllm_args=list(raw.get("vllm_args", [])),
        )

    aliases: dict[str, AliasSpec] = {}
    for alias, raw in MODEL_ALIASES.items():
        backend_name = raw["backend"]
        if backend_name not in backends:
            raise RuntimeError(f"模型别名 {alias} 引用了不存在的 backend: {backend_name}")
        aliases[alias] = AliasSpec(
            alias=alias,
            backend_name=backend_name,
            enable_thinking=raw.get("enable_thinking"),
            extra_body=raw.get("extra_body", {}) or {},
        )

    return ProxyConfig(
        api_keys=set(API_KEYS),
        request_timeout=float(REQUEST_TIMEOUT),
        backends=backends,
        aliases=aliases,
    )


def pick_vllm_command(backend: BackendSpec) -> list[str]:
    default_chat_template_args: list[str] = []
    if backend.enable_thinking is not None:
        default_chat_template_args = [
            "--default-chat-template-kwargs",
            json.dumps({"enable_thinking": backend.enable_thinking}),
        ]

    vllm_exe = shutil.which("vllm")
    if vllm_exe:
        return [
            vllm_exe,
            "serve",
            backend.source_model,
            "--host",
            backend.host,
            "--port",
            str(backend.port),
            "--served-model-name",
            backend.served_model_name,
            "--tokenizer",
            backend.source_model,
            *default_chat_template_args,
            *backend.vllm_args,
        ]

    try:
        importlib.import_module("vllm")
    except ModuleNotFoundError:
        if not AUTO_INSTALL_MISSING:
            raise RuntimeError(
                "没有找到 vllm。请先安装，或把 AUTO_INSTALL_MISSING 改成 True。"
            ) from None

        print("[setup] 缺少依赖 vllm，开始自动安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "vllm"])

    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        backend.source_model,
        "--tokenizer",
        backend.source_model,
        "--host",
        backend.host,
        "--port",
        str(backend.port),
        "--served-model-name",
        backend.served_model_name,
        *default_chat_template_args,
        *backend.vllm_args,
    ]


def wait_backend_ready(backend: BackendSpec, timeout: float) -> None:
    deadline = time.time() + timeout
    url = f"{backend.base_url}/models"
    printed = False
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=10)
            if response.status_code == 200:
                print(f"[ready] {backend.name} 已就绪: {backend.base_url}")
                return
        except Exception:
            pass

        if not printed:
            print(f"[wait] 等待 {backend.name} 加载模型，这一步可能需要几分钟...")
            printed = True
        time.sleep(2)

    raise TimeoutError(f"等待后端 {backend.name} 超时: {url}")


def merge_extra_body(payload: dict[str, Any], extra_body: dict[str, Any]) -> dict[str, Any]:
    if not extra_body:
        return payload
    merged = dict(payload)
    for key, value in extra_body.items():
        merged.setdefault(key, value)
    return merged


def merge_chat_template_kwargs(
    payload: dict[str, Any],
    enable_thinking: bool | None,
) -> dict[str, Any]:
    if enable_thinking is None:
        return payload

    merged = dict(payload)
    chat_template_kwargs = dict(merged.get("chat_template_kwargs") or {})
    chat_template_kwargs.setdefault("enable_thinking", enable_thinking)
    merged["chat_template_kwargs"] = chat_template_kwargs
    return merged


def rewrite_model_name(obj: Any, alias: str) -> Any:
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], str):
            obj["model"] = alias
        for key, value in obj.items():
            obj[key] = rewrite_model_name(value, alias)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            obj[index] = rewrite_model_name(value, alias)
    return obj


def prepare_upstream_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    skip_headers = {"host", "content-length", "authorization"}
    for key, value in request.headers.items():
        if key.lower() not in skip_headers:
            headers[key] = value
    return headers


def prepare_response_headers(headers: httpx.Headers) -> dict[str, str]:
    skip_headers = {
        "content-length",
        "transfer-encoding",
        "connection",
        "content-encoding",
    }
    return {k: v for k, v in headers.items() if k.lower() not in skip_headers}


def create_app(config: ProxyConfig) -> FastAPI:
    app = FastAPI(title="One-Click OpenAI to vLLM Proxy")
    app.state.config = config
    app.state.client = None

    @app.on_event("startup")
    async def startup() -> None:
        timeout = httpx.Timeout(
            connect=30.0,
            read=config.request_timeout,
            write=config.request_timeout,
            pool=30.0,
        )
        app.state.client = httpx.AsyncClient(timeout=timeout)

    @app.on_event("shutdown")
    async def shutdown() -> None:
        if app.state.client is not None:
            await app.state.client.aclose()

    def check_api_key(request: Request) -> None:
        if not config.api_keys:
            return

        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Bearer token.")

        token = auth.removeprefix("Bearer ").strip()
        if token not in config.api_keys:
            raise HTTPException(status_code=401, detail="Invalid API key.")

    def resolve_alias(requested_model: str | None) -> AliasSpec:
        if requested_model:
            alias = config.aliases.get(requested_model)
            if alias is None:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "message": f"Unknown model: {requested_model}",
                        "available_models": sorted(config.aliases.keys()),
                    },
                )
            return alias

        if len(config.aliases) == 1:
            return next(iter(config.aliases.values()))

        raise HTTPException(
            status_code=400,
            detail="Request body must include a model field when multiple aliases exist.",
        )

    async def stream_sse_response(
        upstream_response: httpx.Response,
        alias_name: str,
    ) -> StreamingResponse:
        headers = prepare_response_headers(upstream_response.headers)

        async def iterator():
            try:
                async for line in upstream_response.aiter_lines():
                    if line == "":
                        yield b"\n"
                        continue

                    if not line.startswith("data: "):
                        yield f"{line}\n".encode("utf-8")
                        continue

                    data = line[6:]
                    if data == "[DONE]":
                        yield b"data: [DONE]\n\n"
                        continue

                    try:
                        payload = json.loads(data)
                        payload = rewrite_model_name(payload, alias_name)
                        encoded = json.dumps(payload, ensure_ascii=False)
                        yield f"data: {encoded}\n\n".encode("utf-8")
                    except json.JSONDecodeError:
                        yield f"{line}\n".encode("utf-8")
            finally:
                await upstream_response.aclose()

        media_type = upstream_response.headers.get("content-type", "text/event-stream")
        return StreamingResponse(
            iterator(),
            status_code=upstream_response.status_code,
            media_type=media_type,
            headers=headers,
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "models": sorted(config.aliases.keys()),
            "proxy": f"http://{PROXY_HOST}:{PROXY_PORT}/v1",
            "timestamp": int(time.time()),
        }

    @app.get("/v1/models")
    async def list_models(request: Request) -> JSONResponse:
        check_api_key(request)
        data = [
            {
                "id": alias,
                "object": "model",
                "created": 0,
                "owned_by": "local-vllm",
            }
            for alias in sorted(config.aliases.keys())
        ]
        return JSONResponse({"object": "list", "data": data})

    @app.get("/v1/models/{model_name}")
    async def get_model(model_name: str, request: Request) -> JSONResponse:
        check_api_key(request)
        if model_name not in config.aliases:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")
        return JSONResponse(
            {
                "id": model_name,
                "object": "model",
                "created": 0,
                "owned_by": "local-vllm",
            }
        )

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def proxy(path: str, request: Request) -> Response:
        check_api_key(request)

        body = await request.body()
        headers = dict(request.headers)
        content_type = headers.get("content-type", "")
        payload: dict[str, Any] | None = None

        if body and "application/json" in content_type.lower():
            try:
                candidate = json.loads(body)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
            if isinstance(candidate, dict):
                payload = candidate

        requested_model = payload.get("model") if payload is not None else None
        alias = resolve_alias(requested_model)
        backend = config.backends[alias.backend_name]

        if payload is not None:
            payload["model"] = backend.served_model_name
            payload = merge_extra_body(payload, alias.extra_body)
            payload = merge_chat_template_kwargs(payload, alias.enable_thinking)
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        upstream_url = f"{backend.base_url}/{path}"
        if request.query_params:
            upstream_url = f"{upstream_url}?{request.query_params}"

        upstream_request = app.state.client.build_request(
            method=request.method,
            url=upstream_url,
            headers=prepare_upstream_headers(request),
            content=body,
        )

        is_stream = bool(payload and payload.get("stream") is True)
        if is_stream:
            upstream_response = await app.state.client.send(upstream_request, stream=True)
            if upstream_response.status_code >= 400:
                error_body = await upstream_response.aread()
                response_headers = prepare_response_headers(upstream_response.headers)
                await upstream_response.aclose()
                return Response(
                    content=error_body,
                    status_code=upstream_response.status_code,
                    headers=response_headers,
                    media_type=upstream_response.headers.get("content-type"),
                )
            return await stream_sse_response(upstream_response, alias.alias)

        upstream_response = await app.state.client.send(upstream_request, stream=False)
        response_headers = prepare_response_headers(upstream_response.headers)
        media_type = upstream_response.headers.get("content-type")

        if "application/json" in (media_type or ""):
            try:
                response_json = upstream_response.json()
                response_json = rewrite_model_name(response_json, alias.alias)
                return JSONResponse(
                    content=response_json,
                    status_code=upstream_response.status_code,
                    headers=response_headers,
                )
            except json.JSONDecodeError:
                pass

        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=media_type,
        )

    return app


def main() -> None:
    config = build_config()
    managed_processes: list[ManagedProcess] = []

    def cleanup(*_: Any) -> None:
        for process in reversed(managed_processes):
            process.terminate()

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        for backend in config.backends.values():
            process = ManagedProcess(backend.name, pick_vllm_command(backend))
            process.start()
            managed_processes.append(process)

        for backend in config.backends.values():
            wait_backend_ready(backend, WAIT_BACKEND_TIMEOUT)

        print()
        print("[proxy] 所有后端已就绪，正在启动代理...")
        print(f"[proxy] OpenAI 风格地址: http://127.0.0.1:{PROXY_PORT}/v1")
        print(f"[proxy] API Key: {API_KEYS[0] if API_KEYS else '(empty)'}")
        print(f"[proxy] 可用模型: {', '.join(sorted(config.aliases.keys()))}")
        print()

        app = create_app(config)
        uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level="info")
    finally:
        cleanup()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import inspect
import json
import os
import re
import time
import uuid
import threading
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from flask import Flask, Response, jsonify, request, stream_with_context

try:
    from llama_cpp import Llama
except ImportError as exc:
    raise SystemExit(
        "llama-cpp-python is required for GGUF inference. "
        "Install it with: pip install llama-cpp-python"
    ) from exc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Core Configurations ---
SYSTEM_PROMPT = "You are a database engineer. Generate valid SQL for the given schema and request."
MODEL_DIR = os.environ.get("MODEL_DIR", "./qwen_sql_gguf")
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "16384"))
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", "16384"))
CHAT_FORMAT = os.environ.get("CHAT_FORMAT", "chatml")
N_CTX = int(os.environ.get("N_CTX", "16384"))
N_BATCH = int(os.environ.get("N_BATCH", "512"))
N_THREADS_BATCH = int(os.environ.get("N_THREADS_BATCH", "0"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "0"))
SEED = int(os.environ.get("SEED", "3407"))
VERBOSE = os.environ.get("VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen_sql_gguf")

A2A_PROTOCOL_VERSION = os.environ.get("A2A_PROTOCOL_VERSION", "0.3.0")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["JSON_SORT_KEYS"] = False


# --- GGUF Utility Functions ---
def default_threads() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


N_THREADS = int(os.environ.get("N_THREADS", str(default_threads())))


def _supported_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    filtered = {k: v for k, v in kwargs.items() if v is not None}
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return filtered

    accepted = set(signature.parameters.keys())
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return filtered
    return {k: v for k, v in filtered.items() if k in accepted}


def resolve_model_path(model_arg: str) -> Path:
    path = Path(model_arg).expanduser().resolve()

    if path.is_file():
        if path.suffix.lower() != ".gguf":
            raise ValueError(f"Expected a .gguf file, got: {path}")
        return path

    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Model path must be a GGUF file or directory: {path}")

    candidates = sorted([p for p in path.glob("*.gguf") if p.is_file()])
    if not candidates:
        candidates = sorted([p for p in path.rglob("*.gguf") if p.is_file()])
    if not candidates:
        raise FileNotFoundError(f"No GGUF files found under: {path}")

    quantized = [p for p in candidates if "unquantized" not in p.name.lower()]
    pool = quantized or candidates

    def sort_key(p: Path) -> Tuple[int, float, str]:
        name = p.name.lower()
        quant_hint = 0 if re.search(r"(?:^|[._-])q\d", name) else 1
        return (quant_hint, -p.stat().st_mtime, name)

    return sorted(pool, key=sort_key)[0].resolve()


# --- Hardware Allocation & Model Initialization ---
resolved_model_path = resolve_model_path(MODEL_DIR)
print(">> Inference Runtime: llama.cpp / GGUF")
print(f"Loading GGUF model from {resolved_model_path}...")

_init_kwargs: Dict[str, Any] = {
    "model_path": str(resolved_model_path),
    "n_ctx": N_CTX,
    "n_batch": N_BATCH,
    "n_threads": N_THREADS,
    "n_threads_batch": N_THREADS_BATCH if N_THREADS_BATCH > 0 else None,
    "n_gpu_layers": N_GPU_LAYERS,
    "seed": SEED,
    "verbose": VERBOSE,
}
if CHAT_FORMAT.lower() != "auto":
    _init_kwargs["chat_format"] = CHAT_FORMAT

_init_kwargs = _supported_kwargs(Llama, _init_kwargs)
model = Llama(**_init_kwargs)
model_lock = threading.Lock()
print("Model initialized and ready for inference.")


# --- Utility Functions ---
def make_error_response(
    message: str,
    *,
    status_code: int,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
):
    payload = {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }
    return jsonify(payload), status_code


def get_json_body() -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[Response, int]]]:
    if not request.is_json:
        return None, make_error_response("Request body must be application/json.", status_code=415, param="body")
    payload = request.get_json(silent=True)
    if payload is None or not isinstance(payload, dict):
        return None, make_error_response("Malformed JSON body object.", status_code=400, param="body")
    return payload, None


# --- Local Inference Logic ---
def generate_local_response(messages: List[Dict[str, str]], max_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> Tuple[str, dict]:
    """Executes synchronous text generation using the localized GGUF model."""
    create_kwargs: Dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "repeat_penalty": 1.0,
    }
    create_kwargs = _supported_kwargs(model.create_chat_completion, create_kwargs)

    with model_lock:
        response = model.create_chat_completion(**create_kwargs)

    try:
        text = response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected response format from llama-cpp-python: {response!r}") from exc

    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    usage_metrics = {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }
    return text, usage_metrics


def stream_local_response(messages: List[Dict[str, str]], max_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> Iterable[str]:
    """Yields generated text chunks from the localized GGUF model."""
    create_kwargs: Dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "repeat_penalty": 1.0,
        "stream": True,
    }
    create_kwargs = _supported_kwargs(model.create_chat_completion, create_kwargs)

    with model_lock:
        for chunk in model.create_chat_completion(**create_kwargs):
            try:
                choice = chunk["choices"][0]
            except (KeyError, IndexError, TypeError):
                continue

            delta = ""
            delta_payload = choice.get("delta")
            if isinstance(delta_payload, dict):
                delta = delta_payload.get("content", "") or ""

            if not delta:
                message_payload = choice.get("message")
                if isinstance(message_payload, dict):
                    delta = message_payload.get("content", "") or ""

            if not delta and isinstance(choice.get("text"), str):
                delta = choice["text"]

            if delta:
                yield delta


# --- REST API Endpoints ---
@app.route("/health", methods=["GET"])
@app.route("/healthz", methods=["GET"])
def health() -> Response:
    return jsonify({"status": "ok", "backend": "local_gguf"})


# --- A2A Protocol Implementation ---
def _jsonrpc_success_payload(request_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}

def _jsonrpc_error_payload(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

def _build_a2a_message_payload(text: str, *, context_id: str, final: Optional[bool] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "kind": "message",
        "role": "agent",
        "messageId": uuid.uuid4().hex,
        "contextId": context_id,
        "parts": [{"type": "text", "kind": "text", "text": text}],
        "metadata": {},
    }
    if final is not None:
        payload["final"] = final
    return payload


@app.route("/.well-known/agent-card.json", methods=["GET"])
def a2a_agent_card() -> Response:
    base_url = request.url_root.rstrip("/")
    return jsonify({
        "name": "SQL Agent (Local GGUF)",
        "description": "Generates read-only SQL from natural-language prompts using a local GGUF model.",
        "version": "3.0.0",
        "protocolVersion": A2A_PROTOCOL_VERSION,
        "preferredTransport": "JSONRPC",
        "url": f"{base_url}/",
        "capabilities": {"streaming": True},
    })


@app.route("/v1/sql/generate", methods=["POST"])
def generate_sql() -> Response:
    payload, error = get_json_body()
    if error is not None:
        return error

    schema = payload.get("schema")
    question = payload.get("question")
    system_prompt = payload.get("system_prompt", SYSTEM_PROMPT)

    if not isinstance(schema, str) or not schema.strip():
        return make_error_response("`schema` must be a non-empty string.", status_code=400, param="schema")
    if not isinstance(question, str) or not question.strip():
        return make_error_response("`question` must be a non-empty string.", status_code=400, param="question")
    
    print(f"\n[TRACE/REST] --- INBOUND REQUEST ---")
    print(f"Schema:\n{schema}")
    print(f"Question:\n{question}")
    print(f"------------------------------------\n", flush=True)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Schema: {schema}\nQuestion: {question}"}
    ]

    try:
        start_time = time.perf_counter()
        
        max_tokens = int(payload.get("max_tokens", DEFAULT_MAX_NEW_TOKENS))
        text, usage_metrics = generate_local_response(messages=messages, max_tokens=max_tokens)
        
        latency_seconds = time.perf_counter() - start_time

        response_payload = {
            "id": f"sqlcmpl-{uuid.uuid4().hex}",
            "object": "sql.completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "sql": text,
            "latency_seconds": round(latency_seconds, 6),
            "finish_reason": "stop",
            "usage": usage_metrics,
        }
        
        print(f"\n[TRACE/REST] --- OUTBOUND SQL RESPONSE ---")
        print(f"{text}")
        print(f"------------------------------------------\n", flush=True)
        
        return jsonify(response_payload)
    except Exception as exc:
        err_msg = f"Local generation failed: {exc}"
        print(f"[ERROR/REST] {err_msg}", flush=True)
        return make_error_response(err_msg, status_code=500, error_type="server_error")


@app.route("/", methods=["POST"])
def a2a_rpc() -> Response:
    payload = request.get_json(silent=True) or {}
    request_id = payload.get("id")
    method = payload.get("method")

    if payload.get("jsonrpc") != "2.0" or method not in {"message/send", "message/stream"}:
        return jsonify(_jsonrpc_error_payload(request_id, -32600, "Invalid JSON-RPC or unsupported method.")), 400

    params = payload.get("params", {})
    message = params.get("message", {})
    
    prompt = ""
    if isinstance(message, str):
        prompt = message
    elif isinstance(message, dict):
        parts = message.get("parts")
        if isinstance(parts, list):
            prompt = "".join(p.get("text", "") for p in parts if "text" in p)
        else:
            prompt = message.get("content", "")

    context_id = message.get("contextId", uuid.uuid4().hex) if isinstance(message, dict) else uuid.uuid4().hex

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    print(f"\n[TRACE/A2A] --- INBOUND A2A REQUEST ({method}) ---")
    print(f"Prompt:\n{prompt}")
    print(f"----------------------------------------------\n", flush=True)

    if method == "message/send":
        try:
            text, _ = generate_local_response(messages=messages)
            success_payload = _jsonrpc_success_payload(request_id, _build_a2a_message_payload(text, context_id=context_id))
            
            print(f"\n[TRACE/A2A-SYNC] --- OUTBOUND RESPONSE ---")
            print(f"{text}")
            print(f"------------------------------------------\n", flush=True)
            
            return jsonify(success_payload)
        except Exception as exc:
            err_payload = _jsonrpc_error_payload(request_id, -32000, f"Local inference error: {exc}")
            print(f"[ERROR/A2A-SYNC] Returning error to Orchestrator: {json.dumps(err_payload)}", flush=True)
            return jsonify(err_payload), 500

    # Streaming RPC implementation
    def event_stream() -> Iterable[str]:
        ack_payload = _jsonrpc_success_payload(request_id, _build_a2a_message_payload("", context_id=context_id, final=False))
        yield f"data: {json.dumps(ack_payload, ensure_ascii=False)}\n\n"

        full_text = ""
        try:
            for delta in stream_local_response(messages=messages):
                if delta:
                    full_text += delta
                    chunk_payload = _jsonrpc_success_payload(
                        request_id, 
                        _build_a2a_message_payload(delta, context_id=context_id, final=False)
                    )
                    yield f"data: {json.dumps(chunk_payload, ensure_ascii=False)}\n\n"
                            
            print(f"\n[TRACE/A2A-STREAM] --- OUTBOUND FINAL RESPONSE ---")
            print(f"{full_text}")
            print(f"--------------------------------------------------\n", flush=True)

            # Final chunk
            final_payload = _jsonrpc_success_payload(request_id, _build_a2a_message_payload("", context_id=context_id, final=True))
            yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
        except Exception as exc:
            err_payload = _jsonrpc_error_payload(request_id, -32000, f"Local streaming error: {exc}")
            print(f"[ERROR/A2A-STREAM] Returning error to Orchestrator: {json.dumps(err_payload)}", flush=True)
            yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8001"))
    
    print(f"Starting Local GGUF SQL Agent Server on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)

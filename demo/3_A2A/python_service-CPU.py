#!/usr/bin/env python3
"""
Local Qwen Fine-Tuned Flask REST API and A2A Proxy.

This service replaces the upstream OpenAI proxy logic, utilizing a local
GGUF-quantized Qwen 2.5 7B model for inference while maintaining the exact
HTTP contracts required by the orchestration loop.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

from flask import Flask, Response, jsonify, request, stream_with_context
import torch
from llama_cpp import Llama

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# CONFIGURATION & SETUP
# -----------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("qwen_local_api")

SYSTEM_PROMPT = (
    "You are the Python Agent in a bounded orchestration loop. "
    "Generate concise runnable Python. Prefer Python's built-in sqlite3 module "
    "and a local SQLite database file path unless the prompt explicitly asks for "
    "a different database backend."
)

MODEL_NAME = "qwen-python-local-agent"
MODEL_DIR = os.getenv("MODEL_DIR", "./qwen_python_gguf")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8002"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "16384"))

A2A_PROTOCOL_VERSION = "0.3.0"
A2A_STREAM_CHUNK_SIZE = 256

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# -----------------------------
# HARDWARE & INFERENCE INIT
# -----------------------------
n_gpu_layers = 0
if torch.backends.mps.is_available():
    n_gpu_layers = -1
    logger.info("Hardware Accelerator Selected: Apple-Silicon MPS")
elif torch.cuda.is_available():
    n_gpu_layers = -1
    logger.info("Hardware Accelerator Selected: CUDA GPU")
else:
    logger.info("Hardware Accelerator Selected: CPU")


def resolve_model_path(model_path: str) -> str:
    logger.info("Resolving GGUF model from %s...", model_path)

    if os.path.isfile(model_path):
        if model_path.lower().endswith(".gguf"):
            return model_path
        raise ValueError("MODEL_DIR must point to a .gguf file or a directory containing GGUF files.")

    if os.path.isdir(model_path):
        gguf_files = sorted(
            os.path.join(model_path, name)
            for name in os.listdir(model_path)
            if name.lower().endswith(".gguf")
        )
        if not gguf_files:
            raise FileNotFoundError(f"No GGUF files found in directory: {model_path}")
        if len(gguf_files) > 1:
            logger.warning("Multiple GGUF files found. Using the first one: %s", gguf_files[0])
        return gguf_files[0]

    raise FileNotFoundError(f"Model path does not exist: {model_path}")


def load_model(model_path: str):
    resolved_model_path = resolve_model_path(model_path)
    logger.info("Loading GGUF model from %s...", resolved_model_path)
    return Llama(
        model_path=resolved_model_path,
        n_ctx=0,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )


# Initialize globally to keep the API responsive
model_lock = threading.Lock()
global_model = load_model(MODEL_DIR)
logger.info("Model initialization complete. Ready for inference.")


def _generate_local_completion(messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
    """Handles chat completion against the loaded GGUF model."""
    start_time = time.time()
    with model_lock:
        response = global_model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            stream=False,
        )
    end_time = time.time()

    logger.debug("Local generation completed in %.2f seconds.", end_time - start_time)
    return response



def _generate_local_text(messages: List[Dict[str, str]], max_tokens: int) -> str:
    completion = _generate_local_completion(messages, max_tokens)
    choice = completion["choices"][0]
    message = choice.get("message", {})
    content = message.get("content", "")
    return str(content).strip()


# -----------------------------
# API HELPERS & ROUTES
# -----------------------------
def _api_error(message: str, status_code: int = 400, error_type: str = "invalid_request_error", code: Optional[str] = None) -> Any:
    return jsonify({"error": {"message": message, "type": error_type, "code": code}}), status_code


def _build_python_agent_card() -> Dict[str, Any]:
    rpc_url = request.url_root
    return {
        "name": "Python Local Agent",
        "description": "Generates concise runnable Python code for database-oriented tasks via local Qwen model.",
        "version": "2.0.0",
        "protocolVersion": A2A_PROTOCOL_VERSION,
        "preferredTransport": "JSONRPC",
        "url": rpc_url,
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "capabilities": {"streaming": True},
        "skills": [
            {
                "id": "python_generation",
                "name": "Generate Python",
                "description": "Generate Python code.",
            }
        ],
    }


def _extract_a2a_prompt(body: Dict[str, Any]) -> Tuple[str, str]:
    params = body.get("params", {})
    message = params.get("message", {})
    parts = []
    for p in message.get("parts", []):
        if "text" in p:
            parts.append(p["text"])
    prompt = "".join(parts)
    return prompt, message.get("contextId", uuid.uuid4().hex)


@app.get("/health")
@app.get("/healthz")
def health() -> Any:
    return jsonify({"status": "ok", "model": MODEL_NAME, "backend": "local_inference"})


@app.get("/v1/models")
def list_models() -> Any:
    return jsonify({
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local proxy"}]
    })


@app.route("/.well-known/agent-card.json", methods=["GET"])
@app.route("/.well-known/agent.json", methods=["GET"])
def a2a_agent_card() -> Response:
    return jsonify(_build_python_agent_card())


@app.post("/v1/chat/completions")
def create_chat_completion() -> Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _api_error("Request body must be valid JSON.")

    if payload.get("stream"):
        return _api_error("stream=true not supported via REST mapping yet.", code="unsupported_value")

    raw_messages = payload.get("messages", [])
    if not raw_messages:
        return _api_error("Messages are required.")

    logger.info("REST INCOMING - Messages received:\n%s", json.dumps(raw_messages, indent=2))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + raw_messages

    try:
        completion = _generate_local_completion(messages, DEFAULT_MAX_NEW_TOKENS)
        choice = completion["choices"][0]
        generated_text = str(choice.get("message", {}).get("content", "")).strip()
        usage = completion.get("usage", {})

        # Construct the OpenAI-compatible response object
        response_data = {
            "id": completion.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
            "object": completion.get("object", "chat.completion"),
            "created": completion.get("created", int(time.time())),
            "model": MODEL_NAME,
            "choices": [
                {
                    "index": choice.get("index", 0),
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": choice.get("finish_reason", "stop")
                }
            ],
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0))
            }
        }

        logger.info("REST Chat Completion Trace - Returning to Client:\n%s", generated_text)
        return jsonify(response_data)

    except Exception as exc:
        logger.exception("Local inference failure")
        return _api_error(f"Inference error: {exc}", status_code=500, error_type="internal_server_error")


@app.route("/", methods=["POST"])
def a2a_rpc() -> Response:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid JSON"}}), 400

    request_id = payload.get("id")
    method = payload.get("method")

    if method not in {"message/send", "message/stream"}:
        return jsonify({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "Method not supported"}})

    prompt, context_id = _extract_a2a_prompt(payload)
    logger.info("A2A INCOMING [Req ID: %s] - Extracted Prompt:\n%s", request_id, prompt)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

    if method == "message/send":
        try:
            text = _generate_local_text(messages, DEFAULT_MAX_NEW_TOKENS)
            logger.info("A2A Sync Trace - Returning to Orchestrator [Req ID: %s]:\n%s", request_id, text)

            result = {
                "kind": "message", "role": "agent", "messageId": uuid.uuid4().hex, "contextId": context_id,
                "parts": [{"type": "text", "kind": "text", "text": text}], "final": True
            }
            return jsonify({"jsonrpc": "2.0", "id": request_id, "result": result})

        except Exception as exc:
            logger.error("A2A Sync Error: %s", str(exc))
            return jsonify({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32000, "message": str(exc)}}), 500

    def event_stream() -> Iterable[str]:
        ack = {"jsonrpc": "2.0", "id": request_id, "result": {
            "kind": "message", "role": "agent", "messageId": uuid.uuid4().hex, "contextId": context_id,
            "parts": [{"type": "text", "kind": "text", "text": ""}], "final": False
        }}
        yield f"data: {json.dumps(ack)}\n\n"

        try:
            text = _generate_local_text(messages, DEFAULT_MAX_NEW_TOKENS)
            logger.info("A2A Stream Trace - Assembled text to Orchestrator [Req ID: %s]:\n%s", request_id, text)

            chunks = [text[i:i+A2A_STREAM_CHUNK_SIZE] for i in range(0, len(text), A2A_STREAM_CHUNK_SIZE)]
            for i, chunk in enumerate(chunks):
                payload = {"jsonrpc": "2.0", "id": request_id, "result": {
                    "kind": "message", "role": "agent", "messageId": uuid.uuid4().hex, "contextId": context_id,
                    "parts": [{"type": "text", "kind": "text", "text": chunk}], "final": i == len(chunks)-1
                }}
                yield f"data: {json.dumps(payload)}\n\n"

        except Exception as exc:
            logger.error("A2A Stream Error: %s", str(exc))
            err = {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32000, "message": str(exc)}}
            yield f"data: {json.dumps(err)}\n\n"

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")



def main() -> None:
    logger.info("Starting up Local API Server on %s:%s...", HOST, PORT)
    app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()

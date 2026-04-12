#!/usr/bin/env python3
"""
MLX-Powered Flask REST wrapper and A2A Proxy for Qwen SQL/Python Model.

This service replaces the external OpenAI upstream proxy with a local 
MLX-quantized inference engine running directly on Apple Silicon Unified Memory.
It maintains exact HTTP contracts for both REST and A2A (Agent2Agent) protocols.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, Iterable, List, Tuple, Optional

from flask import Flask, Response, jsonify, request, stream_with_context
from mlx_lm import load, stream_generate

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("qwen_python_api_mlx")

SYSTEM_PROMPT = (
    "You are the Python Agent in a bounded orchestration loop. "
    "Generate concise runnable Python. Prefer Python's built-in sqlite3 module "
    "and a local SQLite database file path unless the prompt explicitly asks for "
    "a different database backend."
)

MODEL_NAME = "python-agent-mlx-local"
MODEL_DIR = os.getenv("MODEL_DIR", "./qwen_python_mlx_int4")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8002"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "512"))

A2A_PROTOCOL_VERSION = "0.3.0"

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

logger.info(f"Loading MLX model and tokenizer into Unified Memory from: {MODEL_DIR}")
load_start = time.perf_counter()
model, tokenizer = load(MODEL_DIR)
logger.info(f"Model initialization complete in {time.perf_counter() - load_start:.2f} seconds.")


def _api_error(message: str, status_code: int = 400, error_type: str = "invalid_request_error", code: Optional[str] = None) -> Any:
    return jsonify({"error": {"message": message, "type": error_type, "code": code}}), status_code


def _build_python_agent_card() -> Dict[str, Any]:
    rpc_url = request.url_root
    return {
        "name": "Python Agent MLX",
        "description": "Generates concise runnable Python code for database-oriented tasks via local MLX engine.",
        "version": "1.0.0",
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

@app.get("/health")
@app.get("/healthz")
def health() -> Any:
    return jsonify({"status": "ok", "model": MODEL_NAME, "backend": "Apple MLX"})


@app.get("/v1/models")
def list_models() -> Any:
    return jsonify({
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local proxy"}]
    })


def _extract_a2a_prompt(body: Dict[str, Any]) -> Tuple[str, str]:
    params = body.get("params", {})
    message = params.get("message", {})
    parts = []
    for p in message.get("parts", []):
        if "text" in p:
            parts.append(p["text"])
    prompt = "".join(parts)
    return prompt, message.get("contextId", uuid.uuid4().hex)


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
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        generated_text = ""
        token_count = 0
        for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=DEFAULT_MAX_NEW_TOKENS):
            generated_text += response.text
            token_count += 1
            
        logger.info("REST Chat Completion Trace - Returning to Client:\n%s", generated_text)

        data = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt), 
                "completion_tokens": token_count, 
                "total_tokens": len(prompt) + token_count
            }
        }
        return jsonify(data)
        
    except Exception as exc:
        logger.exception("Local MLX generation failure")
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

    raw_prompt, context_id = _extract_a2a_prompt(payload)
    
    logger.info("A2A INCOMING [Req ID: %s] - Extracted Prompt:\n%s", request_id, raw_prompt)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": raw_prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if method == "message/send":
        try:
            generated_text = ""
            for response in stream_generate(model, tokenizer, prompt=formatted_prompt, max_tokens=DEFAULT_MAX_NEW_TOKENS):
                generated_text += response.text
                
            logger.info("A2A Sync Trace - Returning to Orchestrator [Req ID: %s]:\n%s", request_id, generated_text)
            
            result = {
                "kind": "message", "role": "agent", "messageId": uuid.uuid4().hex, "contextId": context_id,
                "parts": [{"type": "text", "kind": "text", "text": generated_text}], "final": True
            }
            return jsonify({"jsonrpc": "2.0", "id": request_id, "result": result})
            
        except Exception as exc:
            logger.error("A2A Sync Error: %s", str(exc))
            return jsonify({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32000, "message": str(exc)}}), 502

    def event_stream() -> Iterable[str]:
        ack = {"jsonrpc": "2.0", "id": request_id, "result": {
            "kind": "message", "role": "agent", "messageId": uuid.uuid4().hex, "contextId": context_id,
            "parts": [{"type": "text", "kind": "text", "text": ""}], "final": False
        }}
        yield f"data: {json.dumps(ack)}\n\n"

        try:
            accumulated_text = ""
            for response in stream_generate(model, tokenizer, prompt=formatted_prompt, max_tokens=DEFAULT_MAX_NEW_TOKENS):
                text_chunk = response.text
                accumulated_text += text_chunk
                
                payload = {"jsonrpc": "2.0", "id": request_id, "result": {
                    "kind": "message", "role": "agent", "messageId": uuid.uuid4().hex, "contextId": context_id,
                    "parts": [{"type": "text", "kind": "text", "text": text_chunk}], "final": False
                }}
                yield f"data: {json.dumps(payload)}\n\n"
            
            logger.info("A2A Stream Trace - Assembled text to Orchestrator [Req ID: %s]:\n%s", request_id, accumulated_text)
            
            # Send the final termination flag
            final_payload = {"jsonrpc": "2.0", "id": request_id, "result": {
                "kind": "message", "role": "agent", "messageId": uuid.uuid4().hex, "contextId": context_id,
                "parts": [{"type": "text", "kind": "text", "text": ""}], "final": True
            }}
            yield f"data: {json.dumps(final_payload)}\n\n"
            
        except Exception as exc:
            logger.error("A2A Stream Error: %s", str(exc))
            err = {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32000, "message": str(exc)}}
            yield f"data: {json.dumps(err)}\n\n"

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


def main() -> None:
    logger.info("Starting up Flask server on %s:%s using Apple MLX backend...", HOST, PORT)
    app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
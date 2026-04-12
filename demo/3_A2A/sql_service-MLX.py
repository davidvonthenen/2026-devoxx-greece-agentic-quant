#!/usr/bin/env python3
import json
import os
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from flask import Flask, Response, jsonify, request, stream_with_context

# Apple's native MLX framework tools
from mlx_lm import load, stream_generate

# --- Core Configurations ---
SYSTEM_PROMPT = "You are a database engineer. Generate valid SQL for the given schema and request."
DEFAULT_MODEL_DIR = os.environ.get("MODEL_DIR", "./qwen_sql_mlx_int4")
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", str(2 * 1024 * 1024)))

A2A_PROTOCOL_VERSION = os.environ.get("A2A_PROTOCOL_VERSION", "0.3.0")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["JSON_SORT_KEYS"] = False


# --- MLX Model Initialization ---
print(f"\n>> [INIT] Loading MLX model and tokenizer from: {DEFAULT_MODEL_DIR}")
print(">> [INIT] Mapping tensors directly to Unified Memory...")
load_start = time.perf_counter()
model, tokenizer = load(DEFAULT_MODEL_DIR)
load_time = time.perf_counter() - load_start
print(f">> [INIT] Model loaded in {load_time:.2f} seconds.\n")


# --- Utility Functions ---
def make_error(
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
        return None, make_error("Request body must be application/json.", status_code=415, param="body")
    payload = request.get_json(silent=True)
    if payload is None or not isinstance(payload, dict):
        return None, make_error("Malformed JSON body object.", status_code=400, param="body")
    return payload, None


# --- REST API Endpoints ---
@app.route("/health", methods=["GET"])
@app.route("/healthz", methods=["GET"])
def health() -> Response:
    return jsonify({"status": "ok", "model_dir": DEFAULT_MODEL_DIR, "backend": "mlx"})


@app.route("/v1/models", methods=["GET"])
def list_models() -> Response:
    return jsonify({
        "object": "list",
        "data": [{
            "id": DEFAULT_MODEL_DIR,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
            "permission": [],
            "root": DEFAULT_MODEL_DIR,
            "parent": None,
        }],
    })


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
        "name": "SQL Agent (MLX Native)",
        "description": "Generates read-only SQL from natural-language prompts using local Apple Silicon.",
        "version": "2.0.0",
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
    max_tokens = int(payload.get("max_tokens", DEFAULT_MAX_NEW_TOKENS))

    if not isinstance(schema, str) or not schema.strip():
        return make_error("`schema` must be a non-empty string.", status_code=400, param="schema")
    if not isinstance(question, str) or not question.strip():
        return make_error("`question` must be a non-empty string.", status_code=400, param="question")
    
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
        
        # Apply the chat template specific to the model loaded
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        generated_text = ""
        token_count = 0
        
        for response_chunk in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
            generated_text += response_chunk.text
            token_count += 1
            
        latency_seconds = time.perf_counter() - start_time
        
        response_payload = {
            "id": f"sqlcmpl-{uuid.uuid4().hex}",
            "object": "sql.completion",
            "created": int(time.time()),
            "model": DEFAULT_MODEL_DIR,
            "sql": generated_text,
            "latency_seconds": round(latency_seconds, 6),
            "finish_reason": "stop",
            "usage": {
                "completion_tokens": token_count
            },
        }
        
        print(f"\n[TRACE/REST] --- OUTBOUND SQL RESPONSE ---")
        print(f"{generated_text}")
        print(f"Generation Speed: {token_count / latency_seconds:.2f} TPS")
        print(f"------------------------------------------\n", flush=True)
        
        return jsonify(response_payload)
    except Exception as exc:
        err_msg = f"Local MLX generation failed: {exc}"
        print(f"[ERROR/REST] {err_msg}", flush=True)
        return make_error(err_msg, status_code=500, error_type="server_error")


@app.route("/", methods=["POST"])
def a2a_rpc() -> Response:
    payload = request.get_json(silent=True) or {}
    request_id = payload.get("id")
    method = payload.get("method")

    if payload.get("jsonrpc") != "2.0" or method not in {"message/send", "message/stream"}:
        return jsonify(_jsonrpc_error_payload(request_id, -32600, "Invalid JSON-RPC or unsupported method.")), 400

    params = payload.get("params", {})
    message = params.get("message", {})
    
    prompt_text = ""
    if isinstance(message, str):
        prompt_text = message
    elif isinstance(message, dict):
        parts = message.get("parts")
        if isinstance(parts, list):
            prompt_text = "".join(p.get("text", "") for p in parts if "text" in p)
        else:
            prompt_text = message.get("content", "")

    context_id = message.get("contextId", uuid.uuid4().hex) if isinstance(message, dict) else uuid.uuid4().hex

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text}
    ]

    print(f"\n[TRACE/A2A] --- INBOUND A2A REQUEST ({method}) ---")
    print(f"Prompt:\n{prompt_text}")
    print(f"----------------------------------------------\n", flush=True)

    # Format the payload for the MLX Engine
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if method == "message/send":
        try:
            full_text = ""
            for response_chunk in stream_generate(model, tokenizer, prompt=formatted_prompt, max_tokens=DEFAULT_MAX_NEW_TOKENS):
                full_text += response_chunk.text
            
            success_payload = _jsonrpc_success_payload(request_id, _build_a2a_message_payload(full_text, context_id=context_id))
            
            print(f"\n[TRACE/A2A-SYNC] --- OUTBOUND RESPONSE ---")
            print(f"{full_text}")
            print(f"------------------------------------------\n", flush=True)
            
            return jsonify(success_payload)
        except Exception as exc:
            err_payload = _jsonrpc_error_payload(request_id, -32000, f"Local inference error: {exc}")
            print(f"[ERROR/A2A-SYNC] Returning error to Orchestrator: {json.dumps(err_payload)}", flush=True)
            return jsonify(err_payload), 500

    # Streaming RPC implementation using MLX generator
    def event_stream() -> Iterable[str]:
        ack_payload = _jsonrpc_success_payload(request_id, _build_a2a_message_payload("", context_id=context_id, final=False))
        yield f"data: {json.dumps(ack_payload, ensure_ascii=False)}\n\n"

        full_text = ""
        try:
            for response_chunk in stream_generate(model, tokenizer, prompt=formatted_prompt, max_tokens=DEFAULT_MAX_NEW_TOKENS):
                delta = response_chunk.text
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
            err_payload = _jsonrpc_error_payload(request_id, -32000, f"Local streaming inference error: {exc}")
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
    
    print(f"Starting Local MLX SQL Agent Server routing to {DEFAULT_MODEL_DIR} on http://{host}:{port}")
    # Single-threaded execution is recommended here as MLX tensor operations lock the metal compute layer
    app.run(host=host, port=port, debug=False, threaded=False)
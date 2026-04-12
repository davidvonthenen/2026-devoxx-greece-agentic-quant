from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import random
import re
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import httpx

try:
    from flask import Flask, Response, jsonify, request, stream_with_context
    _FLASK_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # pragma: no cover - optional dependency at authoring time
    _FLASK_IMPORT_ERROR = exc

    class _StubResponse:  # pragma: no cover - fallback only
        pass

    class _StubRequest:
        def __getattr__(self, name: str) -> Any:
            raise MissingDependencyError(
                "Flask is required to use the orchestrator REST API. "
                "Install it with: pip install flask"
            ) from _FLASK_IMPORT_ERROR

    class _StubFlask:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def route(self, *_args: Any, **_kwargs: Any):
            def decorator(fn):
                return fn
            return decorator

        def run(self, *_args: Any, **_kwargs: Any) -> None:
            raise MissingDependencyError(
                "Flask is required to use the orchestrator REST API. "
                "Install it with: pip install flask"
            ) from _FLASK_IMPORT_ERROR

    def jsonify(*_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover - fallback only
        raise MissingDependencyError(
            "Flask is required to use the orchestrator REST API. "
            "Install it with: pip install flask"
        ) from _FLASK_IMPORT_ERROR

    def stream_with_context(generator):  # pragma: no cover - fallback only
        return generator

    Flask = _StubFlask  # type: ignore[assignment]
    Response = _StubResponse  # type: ignore[assignment]
    request = _StubRequest()  # type: ignore[assignment]

JsonDict = Dict[str, Any]

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
TAG_BLOCK_RE_TEMPLATE = r"<{tag}>\s*(.*?)\s*</{tag}>"
CODE_FENCE_SQL_RE = re.compile(r"```sql\s*(.*?)```", re.DOTALL | re.IGNORECASE)
CODE_FENCE_PY_RE = re.compile(r"```(?:python|py)\s*(.*?)```", re.DOTALL | re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
SEMICOLON_SPLIT_RE = re.compile(r";")
JSON_OBJECT_DECODER = json.JSONDecoder()

SQL_REQUEST_SCHEMA: JsonDict = {
    "type": "object",
    "properties": {
        "dialect": {"type": "string"},
        "schema": {"type": "string"},
        "question": {"type": "string"},
        "constraints": {"type": "string"},
    },
    "required": ["dialect", "schema", "question"],
    "additionalProperties": False,
}

DEFAULT_LOCAL_GGUF_MODEL_PATH = os.path.join(
    os.path.expanduser("~"),
    "models",
    "Nemotron-Orchestrator-8B-q4_k_m.gguf",
)

DEFAULT_SCHEMA = """
CREATE TABLE employees (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  department TEXT,
  salary NUMERIC NOT NULL
);
""".strip()

DEFAULT_TASK = (
    "Write a simple Python main-style application that queries a sqlite3 database "
    "to return the employee who has the highest salary."
)

ORCHESTRATOR_OPENAI_MODEL_NAME = os.getenv(
    "ORCHESTRATOR_OPENAI_MODEL_NAME",
    "nemotron-orchestrator-sql-python",
)

REQUEST_TIMEOUT_HEADER_MS = "X-Request-Timeout-Ms"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OrchestrationError(RuntimeError):
    """Base error for orchestration failures."""


class MissingDependencyError(OrchestrationError):
    """Raised when an optional dependency is required but unavailable."""


class ToolCallParseError(OrchestrationError):
    """Raised when a <tool_call> block cannot be parsed safely."""


class ToolValidationError(OrchestrationError):
    """Raised when a tool call fails schema validation."""


class LoopDetectedError(OrchestrationError):
    """Raised when the orchestrator appears stuck in a loop."""


class UnsafeToolInputError(OrchestrationError):
    """Raised when a tool input is unsafe."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: JsonDict
    raw_json: str = ""
    span: Tuple[int, int] = (-1, -1)


@dataclass(frozen=True)
class ParsedAssistantResponse:
    raw_text: str
    cleaned_text: str
    visible_text: str
    tool_calls: Tuple[ToolCall, ...]


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    max_new_tokens: int = 900
    seed: Optional[int] = 0
    stream_to_stdout: bool = True


@dataclass(frozen=True)
class OrchestrationConfig:
    max_turns: int = 8
    max_tool_calls: int = 8
    per_tool_timeout_s: float = 60.0
    overall_timeout_s: float = 240.0
    max_tool_output_chars: int = 12_000
    max_retries_per_tool: int = 3
    max_duplicate_assistant_messages: int = 2
    max_duplicate_tool_calls: int = 2
    route_mode: str = "hybrid"  # model | hybrid
    debug_logging: bool = False
    generation: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: JsonDict

    def as_openai_tool(self) -> JsonDict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class AgentStreamEvent:
    text: str = ""
    artifact: Optional[JsonDict] = None
    raw: Any = None


@dataclass(frozen=True)
class AgentCallResult:
    text: str
    artifacts: Tuple[JsonDict, ...] = ()
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class ToolRunResult:
    name: str
    content: str
    sql: str = ""
    code: str = ""
    sql_request: Optional[JsonDict] = None
    artifacts: Tuple[JsonDict, ...] = ()
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class TaskRequirements:
    requires_sql: bool
    requires_python: bool


@dataclass
class OrchestrationState:
    latest_sql: str = ""
    latest_code: str = ""
    latest_sql_payload: Optional[JsonDict] = None
    latest_python_payload: Optional[JsonDict] = None
    tool_history: List[str] = field(default_factory=list)
    sql_history: List[str] = field(default_factory=list)
    python_history: List[str] = field(default_factory=list)
    sql_revision: int = 0
    python_revision: int = 0
    python_sql_revision: int = 0
    pending_sql_request: Optional[JsonDict] = None
    latest_sql_fingerprint: str = ""
    latest_code_fingerprint: str = ""

    def apply_tool_result(self, call: ToolCall, result: ToolRunResult) -> None:
        self.tool_history.append(result.name)

        if result.name == "sql_agent":
            previous_sql_fingerprint = self.latest_sql_fingerprint
            had_pending_sql_request = self.pending_sql_request is not None
            self.latest_sql_payload = safe_json_loads(result.content)

            new_sql = (result.sql or "").strip()
            sql_changed = False
            if new_sql:
                normalized_sql = normalize_sql_for_comparison(new_sql)
                pretty_sql = normalize_line_endings(new_sql).strip().rstrip(";").strip()
                if normalized_sql != self.latest_sql_fingerprint:
                    self.latest_sql = pretty_sql
                    self.latest_sql_fingerprint = normalized_sql
                    self.sql_history.append(pretty_sql)
                    self.sql_revision += 1
                    sql_changed = True
                elif not self.sql_history:
                    self.latest_sql = pretty_sql
                    self.latest_sql_fingerprint = normalized_sql
                    self.sql_history.append(pretty_sql)
                    self.sql_revision = max(self.sql_revision, 1)
                    sql_changed = True

            if not had_pending_sql_request:
                self.pending_sql_request = None
            elif sql_changed or (self.latest_sql_fingerprint and self.latest_sql_fingerprint != previous_sql_fingerprint):
                self.pending_sql_request = None
            return

        if result.name == "python_agent":
            self.latest_python_payload = safe_json_loads(result.content)
            if result.sql_request is not None:
                self.pending_sql_request = result.sql_request
                return

            new_code = (result.code or "").strip()
            if new_code:
                normalized_code = normalize_code_for_comparison(new_code)
                pretty_code = normalize_line_endings(new_code).strip()
                if normalized_code != self.latest_code_fingerprint:
                    self.latest_code = pretty_code
                    self.latest_code_fingerprint = normalized_code
                    self.python_history.append(pretty_code)
                    self.python_revision += 1
                elif not self.python_history:
                    self.latest_code = pretty_code
                    self.latest_code_fingerprint = normalized_code
                    self.python_history.append(pretty_code)
                    self.python_revision = max(self.python_revision, 1)
                self.python_sql_revision = self.sql_revision
            self.pending_sql_request = None

    def python_is_stale(self) -> bool:
        if not self.latest_code:
            return False
        return self.python_sql_revision != self.sql_revision

    def recent_tools(self, limit: int = 6) -> str:
        recent = self.tool_history[-limit:]
        if not recent:
            return "(none)"
        return ", ".join(recent)

    def latest_sql_status(self) -> str:
        if self.latest_sql_payload is None:
            return "none"
        return "ok" if bool(self.latest_sql_payload.get("ok")) else "error"

    def latest_python_status(self) -> str:
        if self.latest_python_payload is None:
            return "none"
        return "ok" if bool(self.latest_python_payload.get("ok")) else "error"

    def latest_sql_error(self) -> str:
        if not self.latest_sql_payload or self.latest_sql_payload.get("ok", True):
            return ""
        return str(self.latest_sql_payload.get("error") or "").strip()

    def latest_python_error(self) -> str:
        if not self.latest_python_payload or self.latest_python_payload.get("ok", True):
            return ""
        return str(self.latest_python_payload.get("error") or "").strip()

    def progress_fingerprint(self) -> str:
        return canonical_json(
            {
                "sql_revision": self.sql_revision,
                "python_revision": self.python_revision,
                "python_sql_revision": self.python_sql_revision,
                "latest_sql": short_text_hash(self.latest_sql_fingerprint or self.latest_sql),
                "latest_code": short_text_hash(self.latest_code_fingerprint or self.latest_code),
                "pending_sql_request": self.pending_sql_request or {},
                "latest_sql_status": self.latest_sql_status(),
                "latest_python_status": self.latest_python_status(),
            }
        )

    def has_material_output(self) -> bool:
        return bool(self.latest_sql or self.latest_code)


@dataclass
class LoopTracker:
    max_duplicate_assistant_messages: int = 2
    max_duplicate_tool_calls: int = 2
    _last_assistant_signature: str = ""
    _duplicate_assistant_count: int = 0
    _last_tool_signature: str = ""
    _duplicate_tool_count: int = 0
    _tool_signature_counts: Counter[str] = field(default_factory=Counter)

    def observe_assistant(self, text: str, *, state_fingerprint: str = "") -> None:
        fingerprint = WHITESPACE_RE.sub(" ", text).strip()
        if not fingerprint:
            return
        signature = canonical_json({"state": state_fingerprint, "assistant": fingerprint})
        if signature == self._last_assistant_signature:
            self._duplicate_assistant_count += 1
        else:
            self._last_assistant_signature = signature
            self._duplicate_assistant_count = 1
        if self._duplicate_assistant_count > self.max_duplicate_assistant_messages:
            raise LoopDetectedError(
                "Assistant emitted the same response repeatedly without state progress; aborting to avoid an infinite loop."
            )

    def observe_tool_call(self, call: ToolCall, *, state_fingerprint: str = "") -> None:
        signature = canonical_json(
            {
                "state": state_fingerprint,
                "name": call.name,
                "arguments": call.arguments,
            }
        )
        self._tool_signature_counts[signature] += 1
        if signature == self._last_tool_signature:
            self._duplicate_tool_count += 1
        else:
            self._last_tool_signature = signature
            self._duplicate_tool_count = 1
        if self._duplicate_tool_count > self.max_duplicate_tool_calls:
            raise LoopDetectedError(f"Tool call repeated consecutively without progress: {call.name}")
        if self._tool_signature_counts[signature] > self.max_duplicate_tool_calls:
            raise LoopDetectedError(f"Tool call repeated too many times in the same state: {call.name}")


@dataclass(frozen=True)
class ServiceRuntimeConfig:
    local_model_path: str = DEFAULT_LOCAL_GGUF_MODEL_PATH
    local_n_ctx: int = 16384
    local_n_gpu_layers: int = -1
    local_n_threads: int = 0

    sql_agent_url: str = "http://localhost:8001"
    python_agent_url: str = "http://localhost:8002"

    default_task: str = DEFAULT_TASK
    default_schema: str = DEFAULT_SCHEMA
    default_dialect: str = "sqlite3"

    route_mode: str = "hybrid"
    max_turns: int = 8
    max_tool_calls: int = 8
    per_tool_timeout_s: float = 60.0
    overall_timeout_s: float = 240.0
    max_tool_output_chars: int = 12_000

    flask_host: str = "0.0.0.0"
    flask_port: int = 8000
    flask_debug: bool = False
    debug_logging: bool = False


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _debug_log(title: str, content: Any, enabled: bool) -> None:
    if not enabled:
        return
    top_line = f"\n{'='*20} DEBUG: {title} {'='*20}"
    print(top_line)
    if isinstance(content, (dict, list)):
        print(json.dumps(content, indent=2))
    elif isinstance(content, str):
        print(content)
    else:
        print(repr(content))
    print("=" * (len(top_line) - 1) + "\n")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def safe_json_loads(text: str) -> Optional[JsonDict]:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def print_stream(text: str, *, enabled: bool) -> None:
    if enabled and text:
        print(text, end="", flush=True)


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def clamp_text(text: str, max_chars: int, *, tail_chars: int = 800) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if tail_chars >= max_chars:
        tail_chars = max_chars // 4
    head_chars = max_chars - tail_chars - 64
    if head_chars < 0:
        head_chars = max_chars // 2
        tail_chars = max_chars - head_chars
    truncated = len(text) - (head_chars + tail_chars)
    head = text[:head_chars]
    tail = text[-tail_chars:] if tail_chars else ""
    marker = f"\n...[truncated {truncated} chars]...\n"
    return head + marker + tail


def diff_stream_text(previous: str, current: str) -> str:
    if not current:
        return ""
    if previous and current.startswith(previous):
        return current[len(previous) :]
    if current == previous:
        return ""
    return current


def read_text_arg(arg_value: str, file_path: str) -> str:
    if file_path:
        with open(file_path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    return arg_value.strip()


def normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_sql_for_comparison(sql: str) -> str:
    normalized = normalize_line_endings(sql).strip()
    normalized = normalized.rstrip(";").strip()
    return WHITESPACE_RE.sub(" ", normalized)


def normalize_code_for_comparison(code: str) -> str:
    return normalize_line_endings(code).strip()


def short_text_hash(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]



def default_db_connection_hint(dialect: str) -> str:
    return (
        "Use Python's built-in sqlite3 module. Connect to a local SQLite database file path from "
        "SQLITE_DB_PATH or DB_PATH if present. "
        "Assume the user will handle the connection via connection string. "
    )


def default_sql_constraints(dialect: str) -> str:
    return (
        "Return exactly one SQL query."
        "Target SQLite / sqlite3 syntax only."
    )


def default_python_constraints(dialect: str) -> str:
    return (
        "Return Python code only. "
        "Assume the schema already exists. "
        "Execute the provided SQL exactly as given. "
        "Write a compact main-style application that runs the query and prints the requested result with minimal extra text."
    )


async def retry_async(
    fn,
    *,
    tries: int = 3,
    base_delay_s: float = 0.5,
    max_delay_s: float = 5.0,
    jitter: float = 0.2,
) -> Any:
    last_err: Optional[BaseException] = None
    for attempt in range(1, tries + 1):
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                return await result
            return result
        except Exception as exc:
            last_err = exc
            if attempt == tries:
                break
            delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
            delay = delay * (1.0 + random.random() * jitter)
            await asyncio.sleep(delay)
    raise OrchestrationError(f"Retry failed after {tries} attempts: {last_err}") from last_err


async def with_timeout(awaitable, timeout_s: float, label: str) -> Any:
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise OrchestrationError(f"Timeout in {label} after {timeout_s:.1f}s") from exc


def _extract_tagged_blocks(text: str, tag: str) -> List[Tuple[int, int, str]]:
    open_count = text.count(f"<{tag}>")
    close_count = text.count(f"</{tag}>")
    pattern = re.compile(TAG_BLOCK_RE_TEMPLATE.format(tag=re.escape(tag)), re.DOTALL | re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if open_count != len(matches) or close_count != len(matches):
        raise ToolCallParseError(f"Malformed <{tag}> block detected.")
    return [(match.start(), match.end(), match.group(1)) for match in matches]


def _remove_spans(text: str, spans: Sequence[Tuple[int, int]]) -> str:
    if not spans:
        return text
    pieces: List[str] = []
    cursor = 0
    for start, end in sorted(spans):
        if start > cursor:
            pieces.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        pieces.append(text[cursor:])
    return "".join(pieces)


def _load_plain_json_object_strict(raw_json: str, *, tag: str) -> JsonDict:
    stripped = raw_json.strip()
    if not stripped:
        raise ToolCallParseError(f"<{tag}> block was empty.")
    try:
        payload, end = JSON_OBJECT_DECODER.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise ToolCallParseError(f"Invalid JSON inside <{tag}>: {exc}") from exc
    trailing = stripped[end:].strip()
    if trailing:
        raise ToolCallParseError(f"Unexpected trailing content inside <{tag}>.")
    if not isinstance(payload, dict):
        raise ToolCallParseError(f"<{tag}> must contain a JSON object.")
    return payload


def _load_json_object_strict(raw_json: str) -> JsonDict:
    payload = _load_plain_json_object_strict(raw_json, tag="tool_call")
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ToolCallParseError("<tool_call>.name must be a non-empty string.")
    arguments = payload.get("arguments", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ToolCallParseError("<tool_call>.arguments must be a JSON object.")
    return {"name": name.strip(), "arguments": arguments}


def parse_tool_calls(text: str) -> List[ToolCall]:
    cleaned = strip_think(text)
    calls: List[ToolCall] = []
    for start, end, inner in _extract_tagged_blocks(cleaned, "tool_call"):
        payload = _load_json_object_strict(inner)
        calls.append(
            ToolCall(
                name=payload["name"],
                arguments=payload["arguments"],
                raw_json=inner.strip(),
                span=(start, end),
            )
        )
    return calls


def parse_assistant_response(text: str) -> ParsedAssistantResponse:
    cleaned = strip_think(text)
    tool_calls = tuple(parse_tool_calls(cleaned))
    visible_text = _remove_spans(cleaned, [call.span for call in tool_calls]).strip()
    return ParsedAssistantResponse(
        raw_text=text,
        cleaned_text=cleaned,
        visible_text=visible_text,
        tool_calls=tool_calls,
    )


def parse_sql_request_turn(text: str) -> Optional[JsonDict]:
    cleaned = strip_think(text)
    blocks = _extract_tagged_blocks(cleaned, "sql_request")
    if not blocks:
        return None
    if len(blocks) != 1:
        raise ToolCallParseError("Python agent emitted multiple <sql_request> blocks in one turn.")
    start, end, inner = blocks[0]
    payload = _load_plain_json_object_strict(inner, tag="sql_request")
    _validate_json_schema(payload, SQL_REQUEST_SCHEMA)
    visible_text = _remove_spans(cleaned, [(start, end)]).strip()
    if visible_text:
        raise ToolCallParseError("Python agent emitted <sql_request> together with other text or code.")
    return payload


def make_assistant_message(
    parsed: ParsedAssistantResponse,
    *,
    message_index: int = 0,
) -> JsonDict:
    if parsed.tool_calls:
        message: JsonDict = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call_{message_index}_{idx}",
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for idx, call in enumerate(parsed.tool_calls)
            ],
        }
        if parsed.visible_text:
            message["content"] = parsed.visible_text
        return message
    return {"role": "assistant", "content": parsed.visible_text or parsed.cleaned_text}


def make_tool_message(tool_name: str, content: str, *, tool_call_id: Optional[str] = None) -> JsonDict:
    message: JsonDict = {
        "role": "tool",
        "name": tool_name,
        "content": content,
    }
    if tool_call_id:
        message["tool_call_id"] = tool_call_id
    return message


def append_assistant_message(messages: List[JsonDict], parsed: ParsedAssistantResponse) -> JsonDict:
    message = make_assistant_message(parsed, message_index=len(messages))
    messages.append(message)
    return message


def append_tool_message(
    messages: List[JsonDict],
    tool_name: str,
    content: str,
    *,
    tool_call_id: Optional[str] = None,
) -> JsonDict:
    message = make_tool_message(tool_name, content, tool_call_id=tool_call_id)
    messages.append(message)
    return message


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_json_schema(value: Any, schema: JsonDict, *, path: str = "$") -> None:
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            raise ToolValidationError(f"{path} must be a JSON object.")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ToolValidationError(f"{path}.{key} is required.")
        for key, item in value.items():
            if key in properties:
                _validate_json_schema(item, properties[key], path=f"{path}.{key}")
                continue
            if schema.get("additionalProperties", True) is False:
                raise ToolValidationError(f"{path}.{key} is not allowed.")
        return

    if expected_type == "array":
        if not isinstance(value, list):
            raise ToolValidationError(f"{path} must be an array.")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                _validate_json_schema(item, item_schema, path=f"{path}[{index}]")
        return

    if expected_type == "string":
        if not isinstance(value, str):
            raise ToolValidationError(f"{path} must be a string.")
        return

    if expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ToolValidationError(f"{path} must be an integer.")
        return

    if expected_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ToolValidationError(f"{path} must be a number.")
        return

    if expected_type == "boolean":
        if not isinstance(value, bool):
            raise ToolValidationError(f"{path} must be a boolean.")
        return


# ---------------------------------------------------------------------------
# SQL safety
# ---------------------------------------------------------------------------


def _strip_sql_comments_and_literals(sql: str) -> str:
    result: List[str] = []
    index = 0
    length = len(sql)

    while index < length:
        ch = sql[index]
        nxt = sql[index + 1] if index + 1 < length else ""

        if ch == "-" and nxt == "-":
            result.append(" ")
            index += 2
            while index < length and sql[index] not in "\r\n":
                index += 1
            continue

        if ch == "/" and nxt == "*":
            result.append(" ")
            index += 2
            while index + 1 < length and not (sql[index] == "*" and sql[index + 1] == "/"):
                index += 1
            index = min(index + 2, length)
            continue

        if ch == "'":
            result.append(" ")
            index += 1
            while index < length:
                if sql[index] == "'":
                    if index + 1 < length and sql[index + 1] == "'":
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            continue

        if ch == '"':
            result.append(" ")
            index += 1
            while index < length:
                if sql[index] == '"':
                    if index + 1 < length and sql[index + 1] == '"':
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            continue

        if ch == "$":
            match = re.match(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$", sql[index:])
            if match:
                delimiter = match.group(0)
                close_at = sql.find(delimiter, index + len(delimiter))
                if close_at != -1:
                    result.append(" ")
                    index = close_at + len(delimiter)
                    continue

        result.append(ch)
        index += 1

    return "".join(result)


def extract_sql_snippet(text: str) -> str:
    cleaned = strip_think(text)
    match = CODE_FENCE_SQL_RE.search(cleaned)
    if match:
        return match.group(1).strip()

    stripped = cleaned.strip()
    keyword_match = re.search(r"\b(select|with|explain)\b", stripped, re.IGNORECASE)
    if keyword_match:
        candidate = stripped[keyword_match.start() :].strip()
        if candidate:
            return candidate
    return stripped


def extract_python_code(text: str) -> str:
    cleaned = strip_think(text)
    match = CODE_FENCE_PY_RE.search(cleaned)
    if match:
        return match.group(1).strip()

    generic_fence = re.search(r"```\s*(.*?)```", cleaned, re.DOTALL)
    if generic_fence:
        return generic_fence.group(1).strip()
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_sql_agent_prompt(args: JsonDict) -> str:
    dialect = str(args.get("dialect", "")).strip()

    return (
        "You are a SQL generation agent in a bounded orchestration loop.\n"
        "Generate exactly SQL query for the requested task and replace `insert SQL statement` with the actual query in the return format.\n\n"
        f"Dialect: {dialect}\n\n"
        f"Schema:\n{args.get('schema', '')}\n\n"
        f"Question:\n{args.get('question', '')}\n\n"
        "Return format:\n"
        "```sql\n"
        "<insert SQL statement>\n"
        "```\n"
    )


def build_python_agent_prompt(args: JsonDict) -> str:
    dialect = str(args.get("dialect", "")).strip()
    db_connection_hint = str(args.get("db_connection_hint") or default_db_connection_hint(dialect))

    return (
        "You are a Python code-generation agent for database tasks in a bounded orchestration loop.\n"
        "The code must execute the supplied SQL and print or return the requested answer.\n"
        "Prefer a compact main-style script.\n"
        "Use the provided DB connection hint exactly.\n"
        f"Goal:\n{args.get('goal', '')}\n\n"
        f"SQL:\n{args.get('sql', '')}\n\n"
        f"Schema:\n{args.get('schema', '')}\n\n"
        f"Dialect: {dialect}\n\n"
        f"DB connection hint: {db_connection_hint}\n\n"
    )


# ---------------------------------------------------------------------------
# Orchestrator backend
# ---------------------------------------------------------------------------


class OrchestratorBackend(Protocol):
    def generate(
        self,
        *,
        messages: List[JsonDict],
        tools: List[JsonDict],
        generation: GenerationConfig,
    ) -> str:
        """Return full assistant text, possibly including <tool_call> blocks."""


class LocalTransformersOrchestrator:
    """
    Despite the name, this backend uses llama_cpp.Llama against a local GGUF model.
    """

    def __init__(
        self,
        *,
        model_path: str = DEFAULT_LOCAL_GGUF_MODEL_PATH,
        n_ctx: int = 16384,
        n_gpu_layers: int = -1,
        n_threads: int = 0,
        verbose: bool = False,
    ) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:
            raise MissingDependencyError(
                "llama-cpp-python is required to run the local orchestrator backend."
            ) from exc

        self._lock = threading.RLock()
        self._llama = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads or None,
            verbose=verbose,
        )

    def _normalize_messages(self, messages: Sequence[JsonDict]) -> List[JsonDict]:
        normalized: List[JsonDict] = []
        for message in messages:
            msg = dict(message)
            if msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list):
                tool_calls = []
                for item in msg["tool_calls"]:
                    if not isinstance(item, dict):
                        continue
                    block = dict(item)
                    function = dict(block.get("function") or {})
                    arguments = function.get("arguments")
                    if isinstance(arguments, dict):
                        function["arguments"] = canonical_json(arguments)
                    block["function"] = function
                    tool_calls.append(block)
                msg["tool_calls"] = tool_calls
            normalized.append(msg)
        return normalized

    def generate(
        self,
        *,
        messages: List[JsonDict],
        tools: List[JsonDict],
        generation: GenerationConfig,
    ) -> str:
        normalized_messages = self._normalize_messages(messages)

        visible_parts: List[str] = []
        tool_call_buffers: Dict[int, Dict[str, Any]] = {}

        with self._lock:
            stream = self._llama.create_chat_completion(
                messages=normalized_messages,
                tools=tools,
                tool_choice="auto",
                temperature=generation.temperature,
                top_p=generation.top_p,
                max_tokens=generation.max_new_tokens,
                seed=generation.seed,
                stream=True,
            )

            for chunk in stream:
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0] or {}).get("delta") or {}

                content = delta.get("content")
                if isinstance(content, str) and content:
                    visible_parts.append(content)
                    print_stream(content, enabled=generation.stream_to_stdout)

                raw_tool_calls = delta.get("tool_calls") or []
                if isinstance(raw_tool_calls, list):
                    for item in raw_tool_calls:
                        if not isinstance(item, dict):
                            continue
                        index = item.get("index", 0)
                        buf = tool_call_buffers.setdefault(
                            int(index),
                            {
                                "id": item.get("id") or f"tool_{index}",
                                "name": "",
                                "arguments": "",
                            },
                        )
                        function = item.get("function") or {}
                        name_fragment = function.get("name")
                        args_fragment = function.get("arguments")
                        if isinstance(name_fragment, str):
                            buf["name"] += name_fragment
                        if isinstance(args_fragment, str):
                            buf["arguments"] += args_fragment

        if generation.stream_to_stdout and visible_parts:
            print()

        rendered_tool_calls: List[str] = []
        for index in sorted(tool_call_buffers):
            buf = tool_call_buffers[index]
            raw_arguments = (buf.get("arguments") or "").strip()
            try:
                arguments = json.loads(raw_arguments) if raw_arguments else {}
            except Exception:
                arguments = {}
            payload = {
                "name": str(buf.get("name") or "").strip(),
                "arguments": arguments if isinstance(arguments, dict) else {},
            }
            rendered_tool_calls.append(f"<tool_call>{canonical_json(payload)}</tool_call>")

        visible_text = "".join(visible_parts)
        if rendered_tool_calls:
            if visible_text.strip():
                return visible_text.rstrip() + "\n" + "\n".join(rendered_tool_calls)
            return "\n".join(rendered_tool_calls)
        return visible_text


# ---------------------------------------------------------------------------
# A2A helpers
# ---------------------------------------------------------------------------


def _build_a2a_text_part(text: str) -> Dict[str, Any]:
    return {"type": "text", "kind": "text", "text": text}


def _build_a2a_request_payload(prompt: str, *, method: str, context_id: Optional[str] = None) -> Dict[str, Any]:
    context = context_id or uuid.uuid4().hex
    return {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": method,
        "params": {
            "message": {
                "kind": "message",
                "role": "user",
                "messageId": uuid.uuid4().hex,
                "contextId": context,
                "parts": [_build_a2a_text_part(prompt)],
                "metadata": {},
            }
        },
    }


def _extract_text_from_part(part: Any) -> List[str]:
    if part is None:
        return []
    if isinstance(part, str):
        return [part]
    if isinstance(part, dict):
        texts: List[str] = []
        if isinstance(part.get("text"), str):
            texts.append(part["text"])
        if isinstance(part.get("content"), str):
            texts.append(part["content"])
        root = part.get("root")
        if isinstance(root, dict):
            texts.extend(_extract_text_from_part(root))
        parts = part.get("parts")
        if isinstance(parts, list):
            for nested in parts:
                texts.extend(_extract_text_from_part(nested))
        return texts
    text_value = getattr(part, "text", None)
    if isinstance(text_value, str):
        return [text_value]
    root = getattr(part, "root", None)
    if root is not None:
        return _extract_text_from_part(root)
    nested_parts = getattr(part, "parts", None)
    if nested_parts is not None:
        texts: List[str] = []
        for nested in nested_parts:
            texts.extend(_extract_text_from_part(nested))
        return texts
    return []


def _extract_text_fragments(obj: Any) -> List[str]:
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        fragments: List[str] = []
        if isinstance(obj.get("content"), str):
            fragments.append(obj["content"])
        if isinstance(obj.get("text"), str):
            fragments.append(obj["text"])
        if isinstance(obj.get("parts"), list):
            for part in obj["parts"]:
                fragments.extend(_extract_text_from_part(part))
        if obj.get("message") is not None:
            fragments.extend(_extract_text_fragments(obj.get("message")))
        if obj.get("status") is not None:
            fragments.extend(_extract_text_fragments(obj.get("status")))
        if obj.get("result") is not None:
            fragments.extend(_extract_text_fragments(obj.get("result")))
        if obj.get("update") is not None:
            fragments.extend(_extract_text_fragments(obj.get("update")))
        return fragments
    content_value = getattr(obj, "content", None)
    if isinstance(content_value, str):
        return [content_value]
    parts = getattr(obj, "parts", None)
    if parts is not None:
        fragments: List[str] = []
        for part in parts:
            fragments.extend(_extract_text_from_part(part))
        if fragments:
            return fragments
    message = getattr(obj, "message", None)
    if message is not None:
        nested = _extract_text_fragments(message)
        if nested:
            return nested
    status = getattr(obj, "status", None)
    if status is not None:
        nested = _extract_text_fragments(status)
        if nested:
            return nested
    result = getattr(obj, "result", None)
    if result is not None:
        nested = _extract_text_fragments(result)
        if nested:
            return nested
    return []


def _extract_artifacts(obj: Any) -> List[JsonDict]:
    if obj is None:
        return []
    artifacts: List[JsonDict] = []
    if isinstance(obj, dict):
        artifact = obj.get("artifact")
        if artifact is not None:
            payload = artifact if isinstance(artifact, dict) else {"value": artifact}
            artifacts.append(payload)
        if isinstance(obj.get("artifacts"), list):
            for item in obj["artifacts"]:
                payload = item if isinstance(item, dict) else {"value": item}
                artifacts.append(payload)
        if obj.get("update") is not None:
            artifacts.extend(_extract_artifacts(obj.get("update")))
        if obj.get("result") is not None:
            artifacts.extend(_extract_artifacts(obj.get("result")))
        return artifacts
    artifact = getattr(obj, "artifact", None)
    if artifact is not None:
        payload = artifact if isinstance(artifact, dict) else {"value": artifact}
        artifacts.append(payload)
    artifacts_list = getattr(obj, "artifacts", None)
    if isinstance(artifacts_list, list):
        for item in artifacts_list:
            payload = item if isinstance(item, dict) else {"value": item}
            artifacts.append(payload)
    update = getattr(obj, "update", None)
    if update is not None:
        artifacts.extend(_extract_artifacts(update))
    result = getattr(obj, "result", None)
    if result is not None:
        artifacts.extend(_extract_artifacts(result))
    return artifacts


def _extract_text_from_artifact_payload(artifact: JsonDict) -> List[str]:
    texts: List[str] = []
    if isinstance(artifact.get("text"), str):
        texts.append(artifact["text"])
    parts = artifact.get("parts")
    if isinstance(parts, list):
        for part in parts:
            texts.extend(_extract_text_from_part(part))
    return texts


def _normalize_a2a_event(event: Any, previous_snapshot: str) -> Tuple[List[AgentStreamEvent], str]:
    chunks: List[AgentStreamEvent] = []
    sources = list(event) if isinstance(event, tuple) else [event]
    snapshot = previous_snapshot
    for source in sources:
        texts = _extract_text_fragments(source)
        if texts:
            combined = "".join(texts)
            delta = diff_stream_text(snapshot, combined)
            if delta:
                chunks.append(AgentStreamEvent(text=delta, raw=event))
            if combined:
                snapshot = combined
        for artifact in _extract_artifacts(source):
            artifact_text = "".join(_extract_text_from_artifact_payload(artifact))
            if artifact_text:
                delta = diff_stream_text(snapshot, artifact_text)
                if delta:
                    chunks.append(AgentStreamEvent(text=delta, raw=event))
                snapshot = artifact_text
            chunks.append(AgentStreamEvent(artifact=artifact, raw=event))
    return chunks, snapshot


class BaseAgentClient(Protocol):
    name: str

    async def call(self, prompt: str, *, timeout_s: float) -> AgentCallResult:
        ...

    async def close(self) -> None:
        ...


class A2AStreamingAgentClient:
    def __init__(
        self,
        agent_url: str,
        *,
        name: str,
        http_timeout_s: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.agent_url = agent_url.rstrip("/")
        self.name = name
        self.http_timeout_s = http_timeout_s
        self.headers = dict(headers or {})
        self._client = httpx.AsyncClient(timeout=http_timeout_s)

    @property
    def rpc_url(self) -> str:
        return self.agent_url + "/"

    def _request_headers(self, *, timeout_s: float, accept_sse: bool) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            REQUEST_TIMEOUT_HEADER_MS: str(int(max(timeout_s, 0.001) * 1000)),
        }
        if accept_sse:
            headers["Accept"] = "text/event-stream"
            headers["Cache-Control"] = "no-cache"
        headers.update(self.headers)
        return headers

    async def stream_call(self, prompt: str, *, timeout_s: float) -> AsyncIterator[AgentStreamEvent]:
        payload = _build_a2a_request_payload(prompt, method="message/stream")
        snapshot = ""

        async with self._client.stream(
            "POST",
            self.rpc_url,
            json=payload,
            headers=self._request_headers(timeout_s=timeout_s, accept_sse=True),
            timeout=timeout_s,
        ) as response:
            response.raise_for_status()
            data_lines: List[str] = []

            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line:
                    if not data_lines:
                        continue
                    raw_data = "".join(data_lines)
                    data_lines = []
                    if raw_data == "[DONE]":
                        break
                    try:
                        event = json.loads(raw_data)
                    except json.JSONDecodeError:
                        continue
                    normalized, snapshot = _normalize_a2a_event(event, snapshot)
                    for item in normalized:
                        yield item
                    continue

                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())

            if data_lines:
                raw_data = "".join(data_lines)
                if raw_data and raw_data != "[DONE]":
                    with contextlib.suppress(json.JSONDecodeError):
                        event = json.loads(raw_data)
                        normalized, snapshot = _normalize_a2a_event(event, snapshot)
                        for item in normalized:
                            yield item

    async def _send_nonstream(self, prompt: str, *, timeout_s: float) -> AgentCallResult:
        start = time.perf_counter()
        payload = _build_a2a_request_payload(prompt, method="message/send")
        response = await self._client.post(
            self.rpc_url,
            json=payload,
            headers=self._request_headers(timeout_s=timeout_s, accept_sse=False),
            timeout=timeout_s,
        )
        response.raise_for_status()
        body = response.json()
        result = body.get("result") or {}
        text = "".join(_extract_text_fragments(result)).strip()
        artifacts = tuple(_extract_artifacts(result))
        return AgentCallResult(
            text=text,
            artifacts=artifacts,
            elapsed_ms=round((time.perf_counter() - start) * 1000.0, 2),
        )

    async def call(self, prompt: str, *, timeout_s: float) -> AgentCallResult:
        start = time.perf_counter()
        text_parts: List[str] = []
        artifacts: List[JsonDict] = []
        try:
            async for event in self.stream_call(prompt, timeout_s=timeout_s):
                if event.text:
                    text_parts.append(event.text)
                if event.artifact is not None:
                    artifacts.append(event.artifact)
        except Exception:
            return await self._send_nonstream(prompt, timeout_s=timeout_s)

        return AgentCallResult(
            text="".join(text_parts).strip(),
            artifacts=tuple(artifacts),
            elapsed_ms=round((time.perf_counter() - start) * 1000.0, 2),
        )

    async def close(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Tool router
# ---------------------------------------------------------------------------


class ToolRouter:
    def __init__(
        self,
        sql_agent: BaseAgentClient,
        python_agent: BaseAgentClient,
        *,
        max_tool_output_chars: int = 12_000,
    ) -> None:
        self.sql_agent = sql_agent
        self.python_agent = python_agent
        self.max_tool_output_chars = max_tool_output_chars
        self._specs = {spec.name: spec for spec in self._build_specs()}

    def _build_specs(self) -> List[ToolSpec]:
        return [
            ToolSpec(
                name="sql_agent",
                description=(
                    "Generate a single read-only SQL query for the provided database dialect, "
                    "schema, and question."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "dialect": {"type": "string"},
                        "schema": {"type": "string"},
                        "question": {"type": "string"},
                        "constraints": {"type": "string"},
                    },
                    "required": ["dialect", "schema", "question"],
                    "additionalProperties": False,
                },
            ),
            ToolSpec(
                name="python_agent",
                description=(
                    "Generate a Python application that executes the provided SQL and returns or prints the result."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string"},
                        "sql": {"type": "string"},
                        "schema": {"type": "string"},
                        "dialect": {"type": "string"},
                        "db_connection_hint": {"type": "string"},
                        "constraints": {"type": "string"},
                    },
                    "required": ["goal", "sql"],
                    "additionalProperties": False,
                },
            ),
        ]

    def tool_schemas(self) -> List[JsonDict]:
        return [spec.as_openai_tool() for spec in self._specs.values()]

    def validate_tool_call(self, call: ToolCall) -> None:
        spec = self._specs.get(call.name)
        if spec is None:
            raise ToolValidationError(f"Unknown tool: {call.name}")
        _validate_json_schema(call.arguments, spec.parameters)

    async def run_tool(self, call: ToolCall, *, timeout_s: float, debug_logging: bool = False) -> ToolRunResult:
        self.validate_tool_call(call)
        if call.name == "sql_agent":
            return await self._run_sql_agent(call, timeout_s=timeout_s, debug_logging=debug_logging)
        if call.name == "python_agent":
            return await self._run_python_agent(call, timeout_s=timeout_s, debug_logging=debug_logging)
        raise ToolValidationError(f"Unknown tool: {call.name}")

    async def _run_sql_agent(self, call: ToolCall, *, timeout_s: float, debug_logging: bool = False) -> ToolRunResult:
        args = call.arguments
        prompt = build_sql_agent_prompt(args)
        
        _debug_log("SQL Agent - Input Prompt", prompt, debug_logging)
        result = await self.sql_agent.call(prompt, timeout_s=timeout_s)
        _debug_log("SQL Agent - Output Response", result.text, debug_logging)

        sql = extract_sql_snippet(result.text)
        error_message = ""

        payload = {
            "ok": not error_message,
            "tool_name": "sql_agent",
            "text": clamp_text(result.text, self.max_tool_output_chars),
            "sql": clamp_text(sql, max(512, self.max_tool_output_chars // 3)) if sql else "",
            "error": error_message,
            "artifacts": list(result.artifacts),
            "elapsed_ms": round(result.elapsed_ms, 2),
        }
        content = canonical_json(payload)
        return ToolRunResult(
            name="sql_agent",
            content=content,
            sql=sql,
            artifacts=result.artifacts,
            elapsed_ms=result.elapsed_ms,
        )

    async def _run_python_agent(self, call: ToolCall, *, timeout_s: float, debug_logging: bool = False) -> ToolRunResult:
        args = call.arguments

        prompt = build_python_agent_prompt(args)
        
        _debug_log("Python Agent - Input Prompt", prompt, debug_logging)
        result = await self.python_agent.call(prompt, timeout_s=timeout_s)
        _debug_log("Python Agent - Output Response", result.text, debug_logging)

        sql_request_payload = parse_sql_request_turn(result.text)
        if sql_request_payload is not None:
            payload = {
                "ok": True,
                "tool_name": "python_agent",
                "text": clamp_text(result.text, self.max_tool_output_chars),
                "sql_request": sql_request_payload,
                "artifacts": list(result.artifacts),
                "elapsed_ms": round(result.elapsed_ms, 2),
            }
            return ToolRunResult(
                name="python_agent",
                content=canonical_json(payload),
                sql_request=sql_request_payload,
                artifacts=result.artifacts,
                elapsed_ms=result.elapsed_ms,
            )

        code = normalize_line_endings(extract_python_code(result.text)).strip()
        error_message = ""
        if not code:
            error_message = "Python agent returned no code and no <sql_request> block."

        payload = {
            "ok": not error_message,
            "tool_name": "python_agent",
            "text": clamp_text(result.text, self.max_tool_output_chars),
            "code": clamp_text(code, max(1024, self.max_tool_output_chars)) if code else "",
            "error": error_message,
            "artifacts": list(result.artifacts),
            "elapsed_ms": round(result.elapsed_ms, 2),
        }
        content = canonical_json(payload)
        return ToolRunResult(
            name="python_agent",
            content=content,
            code=code if not error_message else "",
            artifacts=result.artifacts,
            elapsed_ms=result.elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Deterministic routing helpers
# ---------------------------------------------------------------------------


def infer_task_requirements(user_task: str, schema: str) -> TaskRequirements:
    task_lower = user_task.lower()
    schema_lower = schema.lower()
    sql_terms = (
        "sql",
        "sqlite3",
        "database",
        "query",
        "table",
        "schema",
        "salary",
        "employee",
    )
    python_terms = (
        "python",
        "script",
        "application",
        "app",
        "program",
        "main-style",
        "main style",
        "main()",
        "main ",
        "code",
    )
    schema_markers = ("create table", "primary key", "foreign key", " fk", "(")
    requires_sql = any(term in task_lower for term in sql_terms) or any(
        marker in schema_lower for marker in schema_markers
    )
    requires_python = any(term in task_lower for term in python_terms)
    return TaskRequirements(requires_sql=requires_sql, requires_python=requires_python)


class RoutePlanner:
    def __init__(
        self,
        *,
        user_task: str,
        schema: str,
        dialect: str,
        requirements: TaskRequirements,
        route_mode: str,
    ) -> None:
        self.user_task = user_task
        self.schema = schema
        self.dialect = dialect
        self.requirements = requirements
        self.route_mode = route_mode

    def build_sql_call(
        self,
        *,
        question: Optional[str] = None,
        constraints: Optional[str] = None,
        dialect: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> ToolCall:
        target_dialect = dialect or self.dialect
        return ToolCall(
            name="sql_agent",
            arguments={
                "dialect": target_dialect,
                "schema": schema or self.schema,
                "question": question or self.user_task,
                "constraints": constraints or default_sql_constraints(target_dialect),
            },
        )

    def build_python_call(
        self,
        state: OrchestrationState,
        *,
        goal: Optional[str] = None,
        constraints: Optional[str] = None,
    ) -> Optional[ToolCall]:
        if self.requirements.requires_sql and not state.latest_sql:
            return None
        return ToolCall(
            name="python_agent",
            arguments={
                "goal": goal or self.user_task,
                "sql": state.latest_sql,
                "schema": self.schema,
                "dialect": self.dialect,
                "db_connection_hint": default_db_connection_hint(self.dialect),
                "constraints": constraints or default_python_constraints(self.dialect),
            },
        )

    def final_requirements_met(self, state: OrchestrationState) -> bool:
        if state.pending_sql_request is not None:
            return False
        if self.requirements.requires_sql and not state.latest_sql:
            return False
        if self.requirements.requires_python:
            if not state.latest_code:
                return False
            if state.python_is_stale():
                return False
        return True

    def next_required_call(self, state: OrchestrationState) -> Optional[ToolCall]:
        if self.final_requirements_met(state):
            return None

        if state.pending_sql_request is not None:
            request_payload = state.pending_sql_request
            return self.build_sql_call(
                question=request_payload.get("question") or self.user_task,
                constraints=request_payload.get("constraints")
                or (
                    "Refine the SQL so the downstream Python agent can complete the task. "
                    f"{default_sql_constraints(request_payload.get('dialect') or self.dialect)}"
                ),
                dialect=request_payload.get("dialect") or self.dialect,
                schema=request_payload.get("schema") or self.schema,
            )

        if self.requirements.requires_sql and not state.latest_sql:
            return self.build_sql_call()

        if self.requirements.requires_python:
            if not state.latest_code:
                return self.build_python_call(state)
            if state.python_is_stale():
                return self.build_python_call(
                    state,
                    constraints=(
                        "The SQL changed after the previous Python draft. Regenerate the Python "
                        "against the latest SQL. "
                        f"{default_python_constraints(self.dialect)}"
                    ),
                )
        return None

    def prepare_call(self, call: ToolCall, state: OrchestrationState) -> ToolCall:
        if call.name == "sql_agent":
            args = dict(call.arguments)
            args.setdefault("dialect", self.dialect)
            args.setdefault("schema", self.schema)
            args.setdefault("question", self.user_task)
            if not args.get("constraints"):
                args["constraints"] = default_sql_constraints(str(args.get("dialect") or self.dialect))
            return ToolCall(name=call.name, arguments=args, raw_json=call.raw_json, span=call.span)

        if call.name == "python_agent":
            args = dict(call.arguments)
            args.setdefault("goal", self.user_task)
            args.setdefault("schema", self.schema)
            args.setdefault("dialect", self.dialect)
            if not args.get("db_connection_hint"):
                args["db_connection_hint"] = default_db_connection_hint(str(args.get("dialect") or self.dialect))
            if state.latest_sql and (self.route_mode != "model" or not args.get("sql")):
                args["sql"] = state.latest_sql
            if not args.get("constraints"):
                args["constraints"] = default_python_constraints(str(args.get("dialect") or self.dialect))
            return ToolCall(name=call.name, arguments=args, raw_json=call.raw_json, span=call.span)

        return call

    def build_turn_status(self, state: OrchestrationState) -> str:
        latest_sql_preview = clamp_text(state.latest_sql, 240, tail_chars=0).replace("\n", " ").strip() or "(none)"
        latest_python_preview = clamp_text(state.latest_code, 240, tail_chars=0).replace("\n", " ").strip() or "(none)"
        status_lines = [
            "Ralph-style loop status:",
            f"- latest_sql_available: {'yes' if state.latest_sql else 'no'}",
            f"- latest_python_available: {'yes' if state.latest_code else 'no'}",
            f"- latest_sql_status: {state.latest_sql_status()}",
            f"- latest_python_status: {state.latest_python_status()}",
            f"- pending_sql_request: {'yes' if state.pending_sql_request is not None else 'no'}",
            f"- sql_revision: {state.sql_revision}",
            f"- python_revision: {state.python_revision}",
            f"- python_stale_vs_latest_sql: {'yes' if state.python_is_stale() else 'no'}",
            f"- final_requirements_met: {'yes' if self.final_requirements_met(state) else 'no'}",
            f"- recent_tools: {state.recent_tools()}",
            f"- latest_sql_preview: {latest_sql_preview}",
            f"- latest_python_preview: {latest_python_preview}",
        ]
        if state.latest_sql_error():
            status_lines.append(f"- latest_sql_error: {clamp_text(state.latest_sql_error(), 240, tail_chars=0)}")
        if state.latest_python_error():
            status_lines.append(f"- latest_python_error: {clamp_text(state.latest_python_error(), 240, tail_chars=0)}")
        if state.pending_sql_request is not None:
            pending_question = str(state.pending_sql_request.get("question") or "").strip()
            if pending_question:
                status_lines.append(
                    f"- pending_sql_question: {clamp_text(pending_question, 240, tail_chars=0).replace(chr(10), ' ')}"
                )
        status_lines.extend(
            [
                "If final_requirements_met=yes and pending_sql_request=no, do not call any more tools. Return the final answer immediately.",
                "Only call a tool when a required artifact is missing, stale, or rejected.",
            ]
        )
        return "\n".join(status_lines)

    def _merge_calls(self, forced: ToolCall, model_call: ToolCall) -> ToolCall:
        merged = dict(forced.arguments)
        for key, value in model_call.arguments.items():
            if value not in (None, ""):
                merged[key] = value
        return ToolCall(name=forced.name, arguments=merged, raw_json=model_call.raw_json, span=model_call.span)

    def choose_calls(self, parsed_calls: Sequence[ToolCall], state: OrchestrationState) -> List[ToolCall]:
        if self.final_requirements_met(state):
            return []

        if self.route_mode == "model":
            return [self.prepare_call(call, state) for call in parsed_calls]

        forced = self.next_required_call(state)
        if forced is None:
            return [self.prepare_call(call, state) for call in parsed_calls]
        for call in parsed_calls:
            if call.name == forced.name:
                return [self.prepare_call(self._merge_calls(forced, call), state)]
        return [self.prepare_call(forced, state)]


def build_system_prompt(config: OrchestrationConfig, requirements: TaskRequirements) -> str:
    lines = [
        "You are an orchestration model.",
        "Operate as a bounded Ralph-style loop: inspect the latest state, call tools, inspect results, and iterate until the task is complete.",
        "Decide when to call tools and when to return the final answer.",
        f"Hard limits: max_turns={config.max_turns}, max_tool_calls={config.max_tool_calls}.",
        "Use deterministic routing. Temperature is zero and sampling is disabled unless explicitly overridden by the caller.",
        "When you need a tool, emit <tool_call>{...}</tool_call> blocks only.",
        "You may call sql_agent and python_agent multiple times across turns when revisions are needed.",
        "After receiving tool responses, continue until the task is complete.",
        "Do not invent SQL if the SQL tool has not been called for a database task.",
        "Do not write Python that runs database queries until SQL is available.",
        "If a later SQL revision changes the query semantics, regenerate Python against the latest SQL before finalizing.",
        "Once the latest required outputs are available and aligned, do not call more tools. Return the final answer immediately.",
        "For sqlite3 tasks, preserve SQLite semantics end-to-end.",
    ]
    if requirements.requires_sql and requirements.requires_python:
        lines.append(
            "For tasks that require both SQL and Python, obtain SQL before the first Python pass, then keep iterating until the SQL and Python outputs agree with each other and the task is fully answered."
        )
    elif requirements.requires_sql:
        lines.append("This task requires SQL. Use sql_agent before finalizing.")
    elif requirements.requires_python:
        lines.append("This task requires Python. Use python_agent before finalizing.")
    return "\n".join(lines)


def compose_best_effort_final(state: OrchestrationState, model_text: str = "") -> str:
    parts: List[str] = []
    if model_text.strip():
        parts.append(model_text.strip())
    if state.latest_sql:
        parts.append("SQL\n```sql\n" + state.latest_sql.strip() + "\n```")
    if state.latest_code:
        parts.append("Python\n```python\n" + state.latest_code.strip() + "\n```")
    if state.latest_sql_payload and state.latest_sql_payload.get("artifacts"):
        parts.append("SQL artifacts: " + canonical_json(state.latest_sql_payload["artifacts"]))
    if state.latest_python_payload and state.latest_python_payload.get("artifacts"):
        parts.append("Python artifacts: " + canonical_json(state.latest_python_payload["artifacts"]))
    return "\n\n".join(part for part in parts if part).strip()


# ---------------------------------------------------------------------------
# Orchestration loop
# ---------------------------------------------------------------------------


async def orchestrate(
    orchestrator: OrchestratorBackend,
    router: ToolRouter,
    *,
    user_task: str,
    schema: str,
    dialect: str = "sqlite3",
    config: Optional[OrchestrationConfig] = None,
) -> str:
    config = config or OrchestrationConfig()
    state = OrchestrationState()
    requirements = infer_task_requirements(user_task, schema)
    planner = RoutePlanner(
        user_task=user_task,
        schema=schema,
        dialect=dialect,
        requirements=requirements,
        route_mode=config.route_mode,
    )
    loop_tracker = LoopTracker(
        max_duplicate_assistant_messages=config.max_duplicate_assistant_messages,
        max_duplicate_tool_calls=config.max_duplicate_tool_calls,
    )
    tools = router.tool_schemas()

    messages: List[JsonDict] = [
        {"role": "system", "content": build_system_prompt(config, requirements)},
        {
            "role": "user",
            "content": f"Task:\n{user_task}\n\nDialect: {dialect}\n\nSchema:\n{schema}\n",
        },
    ]

    async def _run() -> str:
        tool_calls_used = 0
        try:
            for turn in range(1, config.max_turns + 1):
                if planner.final_requirements_met(state):
                    return compose_best_effort_final(state)

                turn_messages = list(messages)
                turn_messages.append({"role": "system", "content": planner.build_turn_status(state)})

                state_fingerprint = state.progress_fingerprint()
                
                _debug_log(f"Turn {turn} - Model Input Messages", turn_messages, config.debug_logging)
                raw = orchestrator.generate(messages=turn_messages, tools=tools, generation=config.generation)
                _debug_log(f"Turn {turn} - Model Output Raw", raw, config.debug_logging)
                
                parsed = parse_assistant_response(raw)
                _debug_log(f"Turn {turn} - Parsed Tool Calls (Attempted by Model)", [c.name for c in parsed.tool_calls], config.debug_logging)
                
                loop_tracker.observe_assistant(
                    parsed.visible_text or parsed.cleaned_text or raw,
                    state_fingerprint=state_fingerprint,
                )

                assistant_message = append_assistant_message(messages, parsed)
                chosen_calls = planner.choose_calls(parsed.tool_calls, state)
                _debug_log(f"Turn {turn} - Chosen Routing Calls (Post-Planner)", [c.name for c in chosen_calls], config.debug_logging)

                if chosen_calls != list(parsed.tool_calls):
                    forced_parsed = ParsedAssistantResponse(
                        raw_text=parsed.raw_text,
                        cleaned_text=parsed.cleaned_text,
                        visible_text=parsed.visible_text,
                        tool_calls=tuple(chosen_calls),
                    )
                    messages[-1] = make_assistant_message(forced_parsed, message_index=len(messages) - 1)
                    assistant_message = messages[-1]

                if not chosen_calls:
                    if config.route_mode != "model":
                        forced = planner.next_required_call(state)
                        if forced is not None:
                            forced_parsed = ParsedAssistantResponse(
                                raw_text="",
                                cleaned_text="",
                                visible_text="",
                                tool_calls=(forced,),
                            )
                            assistant_message = append_assistant_message(messages, forced_parsed)
                            chosen_calls = [forced]
                            _debug_log(f"Turn {turn} - Forced Fallback Call", [c.name for c in chosen_calls], config.debug_logging)
                        elif planner.final_requirements_met(state):
                            return compose_best_effort_final(state, parsed.visible_text or parsed.cleaned_text)
                        elif state.has_material_output():
                            return compose_best_effort_final(
                                state,
                                parsed.visible_text or parsed.cleaned_text or "Returning best-effort output.",
                            )
                        else:
                            return parsed.visible_text or parsed.cleaned_text
                    else:
                        if planner.final_requirements_met(state):
                            return compose_best_effort_final(state, parsed.visible_text or parsed.cleaned_text)
                        if state.has_material_output():
                            return compose_best_effort_final(
                                state,
                                parsed.visible_text or parsed.cleaned_text or "Returning best-effort output.",
                            )
                        return parsed.visible_text or parsed.cleaned_text

                tool_call_ids = [tc.get("id") for tc in assistant_message.get("tool_calls", [])]
                assistant_tool_calls = assistant_message.get("tool_calls", [])
                for index, call in enumerate(chosen_calls):
                    current_call = planner.prepare_call(call, state)
                    if index < len(assistant_tool_calls):
                        function_block = assistant_tool_calls[index].setdefault("function", {})
                        function_block["name"] = current_call.name
                        function_block["arguments"] = current_call.arguments

                    tool_calls_used += 1
                    if tool_calls_used > config.max_tool_calls:
                        return compose_best_effort_final(
                            state,
                            "Tool-call budget exceeded. Returning best-effort output.",
                        )

                    pre_call_state_fingerprint = state.progress_fingerprint()
                    loop_tracker.observe_tool_call(current_call, state_fingerprint=pre_call_state_fingerprint)

                    tool_result: ToolRunResult = await with_timeout(
                        retry_async(
                            lambda prepared_call=current_call: router.run_tool(
                                prepared_call,
                                timeout_s=config.per_tool_timeout_s,
                                debug_logging=config.debug_logging,
                            ),
                            tries=config.max_retries_per_tool,
                        ),
                        timeout_s=config.per_tool_timeout_s,
                        label=f"tool:{current_call.name}",
                    )
                    state.apply_tool_result(current_call, tool_result)
                    
                    _debug_log(f"Turn {turn} - Tool Msg Inserted ({current_call.name})", tool_result.content, config.debug_logging)
                    
                    tool_call_id = tool_call_ids[index] if index < len(tool_call_ids) else None
                    append_tool_message(messages, current_call.name, tool_result.content, tool_call_id=tool_call_id)

                    if planner.final_requirements_met(state):
                        return compose_best_effort_final(state)

            return compose_best_effort_final(state, "Max turns reached. Returning best-effort output.")
        except LoopDetectedError as exc:
            if state.has_material_output():
                return compose_best_effort_final(
                    state,
                    f"Loop detected ({exc}). Returning best-effort output.",
                )
            raise

    return await with_timeout(_run(), timeout_s=config.overall_timeout_s, label="orchestration loop")


# ---------------------------------------------------------------------------
# Request / response helpers for the Flask OpenAI-compatible API
# ---------------------------------------------------------------------------

def extract_openai_message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
                continue
            text_value = item.get("text")
            if isinstance(text_value, str):
                parts.append(text_value)
                continue
            input_text = item.get("input_text")
            if isinstance(input_text, str):
                parts.append(input_text)
        return "".join(parts).strip()
    return str(content).strip()


def normalize_openai_messages(raw_messages: Any) -> List[JsonDict]:
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty array.")
    messages: List[JsonDict] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            raise ValueError("each message must be an object.")
        role = str(item.get("role") or "").strip()
        if not role:
            raise ValueError("each message must include a role.")
        content = extract_openai_message_text(item.get("content"))
        messages.append({"role": role, "content": content})
    return messages


def compose_user_task_from_messages(messages: Sequence[JsonDict]) -> str:
    if len(messages) == 1 and messages[0].get("role") == "user":
        return str(messages[0].get("content") or "").strip()

    sections: List[str] = []
    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        if role in {"system", "developer"}:
            sections.append(f"{role.upper()} INSTRUCTIONS:\n{content}")
        else:
            sections.append(f"{role.upper()}:\n{content}")
    return "\n\n".join(sections).strip()


def coerce_int(value: Any, default: int, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def coerce_float(value: Any, default: float, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def build_chat_completion_response(
    *,
    completion_id: str,
    created: int,
    model: str,
    text: str,
    finish_reason: str = "stop",
) -> Dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def chunk_text(text: str, chunk_size: int = 256) -> Iterable[str]:
    if not text:
        yield ""
        return
    for index in range(0, len(text), chunk_size):
        yield text[index : index + chunk_size]


def build_streaming_chat_completion_response(
    *,
    completion_id: str,
    created: int,
    model: str,
    text: str,
) -> Response:
    def event_stream() -> Iterable[str]:
        for piece in chunk_text(text, chunk_size=256):
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": piece},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        done = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def make_openai_error(
    message: str,
    *,
    status_code: int = 400,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
) -> Response:
    error: Dict[str, Any] = {"message": message, "type": error_type}
    if param is not None:
        error["param"] = param
    return jsonify({"error": error}), status_code


def run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    container: Dict[str, Any] = {}
    error_container: Dict[str, BaseException] = {}

    def runner() -> None:
        try:
            container["result"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive
            error_container["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in error_container:
        raise error_container["error"]
    return container.get("result")


# ---------------------------------------------------------------------------
# Flask app state
# ---------------------------------------------------------------------------


class OrchestratorServiceState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.runtime_config = ServiceRuntimeConfig()
        self._backend: Optional[LocalTransformersOrchestrator] = None

    def configure(self, config: ServiceRuntimeConfig) -> None:
        with self._lock:
            self.runtime_config = config
            self._backend = None

    def get_backend(self) -> LocalTransformersOrchestrator:
        with self._lock:
            if self._backend is None:
                self._backend = LocalTransformersOrchestrator(
                    model_path=self.runtime_config.local_model_path,
                    n_ctx=self.runtime_config.local_n_ctx,
                    n_gpu_layers=self.runtime_config.local_n_gpu_layers,
                    n_threads=self.runtime_config.local_n_threads,
                )
            return self._backend

    def warm_start_backend(self) -> LocalTransformersOrchestrator:
        return self.get_backend()


SERVICE_STATE = OrchestratorServiceState()
app = Flask(__name__)


def _resolve_request_schema(payload: Dict[str, Any], runtime: ServiceRuntimeConfig) -> str:
    metadata = payload.get("metadata") or {}
    schema = payload.get("schema")
    if isinstance(schema, str) and schema.strip():
        return schema.strip()
    metadata_schema = metadata.get("schema")
    if isinstance(metadata_schema, str) and metadata_schema.strip():
        return metadata_schema.strip()
    return runtime.default_schema


def _resolve_request_dialect(payload: Dict[str, Any], runtime: ServiceRuntimeConfig) -> str:
    metadata = payload.get("metadata") or {}
    dialect = payload.get("dialect")
    if isinstance(dialect, str) and dialect.strip():
        return dialect.strip()
    metadata_dialect = metadata.get("dialect")
    if isinstance(metadata_dialect, str) and metadata_dialect.strip():
        return metadata_dialect.strip()
    return runtime.default_dialect


def _resolve_request_route_mode(payload: Dict[str, Any], runtime: ServiceRuntimeConfig) -> str:
    metadata = payload.get("metadata") or {}
    route_mode = payload.get("route_mode")
    if isinstance(route_mode, str) and route_mode in {"model", "hybrid"}:
        return route_mode
    metadata_route_mode = metadata.get("route_mode")
    if isinstance(metadata_route_mode, str) and metadata_route_mode in {"model", "hybrid"}:
        return metadata_route_mode
    return runtime.route_mode


def _resolve_request_generation(payload: Dict[str, Any]) -> GenerationConfig:
    max_new_tokens = payload.get("max_completion_tokens", payload.get("max_tokens", 900))
    return GenerationConfig(
        temperature=coerce_float(payload.get("temperature"), 0.0, minimum=0.0),
        top_p=coerce_float(payload.get("top_p"), 1.0, minimum=0.0, maximum=1.0),
        do_sample=coerce_float(payload.get("temperature"), 0.0) > 0.0,
        max_new_tokens=coerce_int(max_new_tokens, 900, minimum=1),
        seed=coerce_int(payload.get("seed"), 0),
        stream_to_stdout=False,
    )


def _build_request_config(payload: Dict[str, Any], runtime: ServiceRuntimeConfig) -> OrchestrationConfig:
    return OrchestrationConfig(
        max_turns=coerce_int(payload.get("max_turns"), runtime.max_turns, minimum=1),
        max_tool_calls=coerce_int(payload.get("max_tool_calls"), runtime.max_tool_calls, minimum=1),
        per_tool_timeout_s=coerce_float(
            payload.get("per_tool_timeout"),
            runtime.per_tool_timeout_s,
            minimum=1.0,
        ),
        overall_timeout_s=coerce_float(
            payload.get("overall_timeout"),
            runtime.overall_timeout_s,
            minimum=1.0,
        ),
        max_tool_output_chars=coerce_int(
            payload.get("max_tool_output_chars"),
            runtime.max_tool_output_chars,
            minimum=256,
        ),
        route_mode=_resolve_request_route_mode(payload, runtime),
        debug_logging=bool(payload.get("debug", runtime.debug_logging)),
        generation=_resolve_request_generation(payload),
    )


async def _run_orchestrator_request(payload: Dict[str, Any], runtime: ServiceRuntimeConfig) -> str:
    messages = normalize_openai_messages(payload.get("messages"))
    task = compose_user_task_from_messages(messages) or runtime.default_task
    schema = _resolve_request_schema(payload, runtime) or runtime.default_schema
    dialect = _resolve_request_dialect(payload, runtime)
    config = _build_request_config(payload, runtime)

    sql_agent = A2AStreamingAgentClient(
        runtime.sql_agent_url,
        name="sql_agent",
        http_timeout_s=max(config.per_tool_timeout_s, 60.0),
    )

    python_agent = A2AStreamingAgentClient(
        runtime.python_agent_url,
        name="python_agent",
        http_timeout_s=max(config.per_tool_timeout_s, 60.0),
    )

    router = ToolRouter(
        sql_agent=sql_agent,
        python_agent=python_agent,
        max_tool_output_chars=config.max_tool_output_chars,
    )

    try:
        return await orchestrate(
            SERVICE_STATE.get_backend(),
            router,
            user_task=task,
            schema=schema,
            dialect=dialect,
            config=config,
        )
    finally:
        await sql_agent.close()
        await python_agent.close()


@app.route("/health", methods=["GET"])
@app.route("/healthz", methods=["GET"])
def health() -> Response:
    return jsonify({"ok": True, "service": "orchestrator"})


@app.route("/v1/models", methods=["GET"])
def list_models() -> Response:
    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": ORCHESTRATOR_OPENAI_MODEL_NAME,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }
    )


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions() -> Response:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return make_openai_error("Request body must be a JSON object.", status_code=400)

    if payload.get("n", 1) != 1:
        return make_openai_error("Only n=1 is supported.", status_code=400, param="n")

    if payload.get("tools"):
        return make_openai_error(
            "Tool calling is not supported by this endpoint. The orchestrator manages its own internal tools.",
            status_code=400,
            param="tools",
        )

    try:
        runtime = SERVICE_STATE.runtime_config
        result_text = run_async(_run_orchestrator_request(payload, runtime))
    except (ValueError, ToolCallParseError, ToolValidationError, UnsafeToolInputError) as exc:
        return make_openai_error(str(exc), status_code=400)
    except MissingDependencyError as exc:
        return make_openai_error(str(exc), status_code=500, error_type="server_error")
    except Exception as exc:
        return make_openai_error(
            f"Orchestration failed: {exc}",
            status_code=500,
            error_type="server_error",
        )

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model_name = str(payload.get("model") or ORCHESTRATOR_OPENAI_MODEL_NAME)

    if bool(payload.get("stream", False)):
        return build_streaming_chat_completion_response(
            completion_id=completion_id,
            created=created,
            model=model_name,
            text=result_text,
        )

    return jsonify(
        build_chat_completion_response(
            completion_id=completion_id,
            created=created,
            model=model_name,
            text=result_text,
        )
    )


# ---------------------------------------------------------------------------
# CLI and service bootstrap
# ---------------------------------------------------------------------------


def build_service_runtime_config(args: argparse.Namespace) -> ServiceRuntimeConfig:
    return ServiceRuntimeConfig(
        local_model_path=args.local_model_path,
        local_n_ctx=args.local_n_ctx,
        local_n_gpu_layers=args.local_n_gpu_layers,
        local_n_threads=args.local_n_threads,
        sql_agent_url=args.sql_agent_url,
        python_agent_url=args.python_agent_url,
        default_task=args.task,
        default_schema=args.schema,
        default_dialect=args.dialect,
        route_mode=args.route_mode,
        max_turns=args.max_turns,
        max_tool_calls=args.max_tool_calls,
        per_tool_timeout_s=args.per_tool_timeout,
        overall_timeout_s=args.overall_timeout,
        max_tool_output_chars=args.max_tool_output_chars,
        flask_host=args.host,
        flask_port=args.port,
        flask_debug=args.flask_debug,
        debug_logging=args.debug,
    )


async def main_once(args: argparse.Namespace) -> None:
    user_task = args.task.strip()
    schema = args.schema.strip()

    generation = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
        max_new_tokens=900,
        seed=0,
        stream_to_stdout=not args.no_stream,
    )
    config = OrchestrationConfig(
        max_turns=args.max_turns,
        max_tool_calls=args.max_tool_calls,
        per_tool_timeout_s=args.per_tool_timeout,
        overall_timeout_s=args.overall_timeout,
        max_tool_output_chars=args.max_tool_output_chars,
        route_mode=args.route_mode,
        debug_logging=args.debug,
        generation=generation,
    )

    backend: OrchestratorBackend = SERVICE_STATE.get_backend()

    sql_agent = A2AStreamingAgentClient(
        args.sql_agent_url,
        name="sql_agent",
        http_timeout_s=max(args.per_tool_timeout, 60.0),
    )

    python_agent = A2AStreamingAgentClient(
        args.python_agent_url,
        name="python_agent",
        http_timeout_s=max(args.per_tool_timeout, 60.0),
    )

    router = ToolRouter(
        sql_agent=sql_agent,
        python_agent=python_agent,
        max_tool_output_chars=config.max_tool_output_chars,
    )

    try:
        final = await orchestrate(
            backend,
            router,
            user_task=user_task,
            schema=schema,
            dialect=args.dialect,
            config=config,
        )
    finally:
        await sql_agent.close()
        await python_agent.close()

    print("\n=== FINAL ===\n")
    print(final)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nemotron orchestrator for bounded SQL + Python workflows using A2A agents."
    )
    parser.add_argument("--serve", action="store_true", help="Run the Flask OpenAI-compatible REST API.")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument(
        "--flask-debug",
        action="store_true",
        default=os.getenv("FLASK_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"},
    )

    parser.add_argument(
        "--local-model-path",
        default=os.getenv("ORCH_LOCAL_MODEL_PATH", DEFAULT_LOCAL_GGUF_MODEL_PATH),
    )
    parser.add_argument("--local-n-ctx", type=int, default=int(os.getenv("ORCH_LOCAL_N_CTX", "16384")))
    parser.add_argument(
        "--local-n-gpu-layers",
        type=int,
        default=int(os.getenv("ORCH_LOCAL_N_GPU_LAYERS", "-1")),
    )
    parser.add_argument("--local-n-threads", type=int, default=int(os.getenv("ORCH_LOCAL_N_THREADS", "0")))

    parser.add_argument("--sql-agent-url", default=os.getenv("SQL_AGENT_URL", "http://localhost:8001"))
    parser.add_argument("--python-agent-url", default=os.getenv("PYTHON_AGENT_URL", "http://localhost:8002"))

    parser.add_argument("--task", default=os.getenv("ORCH_TASK", DEFAULT_TASK))
    parser.add_argument("--schema", default=os.getenv("ORCH_SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--dialect", default=os.getenv("ORCH_DIALECT", "sqlite3"))

    parser.add_argument(
        "--route-mode",
        choices=["model", "hybrid"],
        default=os.getenv("ORCH_ROUTE_MODE", "hybrid"),
    )
    parser.add_argument("--max-turns", type=int, default=int(os.getenv("ORCH_MAX_TURNS", "8")))
    parser.add_argument("--max-tool-calls", type=int, default=int(os.getenv("ORCH_MAX_TOOL_CALLS", "8")))
    parser.add_argument("--per-tool-timeout", type=float, default=float(os.getenv("ORCH_PER_TOOL_TIMEOUT", "60")))
    parser.add_argument("--overall-timeout", type=float, default=float(os.getenv("ORCH_OVERALL_TIMEOUT", "240")))
    parser.add_argument(
        "--max-tool-output-chars",
        type=int,
        default=int(os.getenv("ORCH_MAX_TOOL_OUTPUT_CHARS", "12000")),
    )
    parser.add_argument("--allow-destructive-sql", action="store_true")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging for inputs/outputs in the orchestration loop.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    SERVICE_STATE.configure(build_service_runtime_config(args))
    SERVICE_STATE.warm_start_backend()

    if args.serve:
        app.run(host=args.host, port=args.port, debug=args.flask_debug, threaded=True)
        return

    asyncio.run(main_once(args))


if __name__ == "__main__":
    main()

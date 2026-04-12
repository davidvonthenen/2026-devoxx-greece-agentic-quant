#!/usr/bin/env python3
"""
Run local inference against a GGUF model with llama-cpp-python.

This script replaces the older Transformers + Optimum-Quanto runner with a
GGUF-native runtime. It is designed for the merged and quantized Qwen Python
assistant produced by the GGUF quantization script, for example:

    ./qwen_python_gguf/merged_qwen_python.Q4_K_M.gguf

Features
--------
- Accepts either a direct .gguf file or a directory containing GGUF files.
- If a directory is provided, prefers a quantized GGUF over an
  *.unquantized.gguf file.
- Preserves the same system prompt used by the fine-tune and prior inference
  script for consistent behavior.
- Supports bundled examples, a one-shot prompt, or an interactive REPL.
- Defaults to CPU-only inference (n_gpu_layers=0).

Install
-------
    pip install llama-cpp-python

If you later want Metal acceleration on Apple Silicon, rebuild/install
llama-cpp-python with the appropriate Metal flags and set --n_gpu_layers > 0.
"""

from __future__ import annotations

import argparse
import inspect
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

try:
    from llama_cpp import Llama
except ImportError as exc:  # pragma: no cover - import failure is runtime-only
    raise SystemExit(
        "llama-cpp-python is required for GGUF inference. "
        "Install it with: pip install llama-cpp-python"
    ) from exc


SYSTEM_PROMPT = (
    "You are a senior Python engineer and code assistant. "
    "Answer the user's question accurately. When helpful, include Python code."
)


EXAMPLE_PROMPTS: list[tuple[int, str]] = [
    (
        1,
        "Write a Python function `slugify(text: str) -> str` that:\n"
        "- lowercases\n"
        "- replaces whitespace runs with '-'\n"
        "- removes non-alphanumeric characters except '-'\n"
        "- strips leading/trailing '-'\n"
        "Include a few quick examples.",
    ),
    (
        2,
        "I have this code:\n"
        "```python\n"
        "def f(items=[]):\n"
        "    items.append(1)\n"
        "    return items\n"
        "```\n"
        "Why does it keep growing across calls? Fix it and explain the change.",
    ),
    (
        3,
        "Given a list of dicts like:\n"
        "```python\n"
        "rows = [\n"
        "  {'user':'a','score':10},\n"
        "  {'user':'b','score':7},\n"
        "  {'user':'a','score':3},\n"
        "]\n"
        "```\n"
        "Write code to aggregate total score per user, returning a dict mapping user->total.\n"
        "Show two approaches: plain loop and using collections.",
    ),
]


def default_threads() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GGUF inference for the Qwen Python assistant using llama-cpp-python."
    )

    parser.add_argument(
        "--model",
        "--model_path",
        dest="model",
        default="./qwen_python_gguf",
        help=(
            "Path to a GGUF file or a directory containing GGUF files. "
            "If a directory is given, the script prefers a quantized file over an unquantized one."
        ),
    )
    parser.add_argument(
        "--chat_format",
        type=str,
        default="chatml",
        help=(
            "Chat format passed to llama-cpp-python. Use 'chatml' for Qwen-style chat prompts, "
            "or 'auto' to rely on GGUF metadata/template detection."
        ),
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=SYSTEM_PROMPT,
        help="System prompt prepended to each request.",
    )

    parser.add_argument("--n_ctx", type=int, default=4096, help="Context window to allocate.")
    parser.add_argument(
        "--n_batch",
        type=int,
        default=512,
        help="Prompt processing batch size for llama.cpp.",
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=default_threads(),
        help="CPU threads used for token generation.",
    )
    parser.add_argument(
        "--n_threads_batch",
        type=int,
        default=0,
        help="CPU threads for prompt processing. 0 means use n_threads or library defaults.",
    )
    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=0,
        help="Layers to offload to GPU. 0 keeps inference CPU-only.",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Sampling seed.")

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate per answer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0.0 is deterministic/greedy-like behavior.",
    )
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p nucleus sampling value.")
    parser.add_argument(
        "--repeat_penalty",
        type=float,
        default=1.0,
        help="Repeat penalty passed to llama.cpp.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Run a single prompt and exit.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Read a single prompt from a text file and run it once.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start a simple interactive chat loop.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable llama-cpp-python verbose logging.",
    )

    return parser.parse_args()


def _supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return only keyword args accepted by the callable when introspection works."""
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

    def sort_key(p: Path) -> tuple[int, float, str]:
        name = p.name.lower()
        quant_hint = 0 if re.search(r"(?:^|[._-])q\d", name) else 1
        return (quant_hint, -p.stat().st_mtime, name)

    chosen = sorted(pool, key=sort_key)[0]
    return chosen.resolve()


def read_prompt_from_file(path_str: str) -> str:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_llm(args: argparse.Namespace, model_path: Path) -> Llama:
    init_kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": args.n_ctx,
        "n_batch": args.n_batch,
        "n_threads": args.n_threads,
        "n_threads_batch": args.n_threads_batch if args.n_threads_batch > 0 else None,
        "n_gpu_layers": args.n_gpu_layers,
        "seed": args.seed,
        "verbose": args.verbose,
    }
    if args.chat_format.lower() != "auto":
        init_kwargs["chat_format"] = args.chat_format

    filtered_kwargs = _supported_kwargs(Llama.__init__, init_kwargs)
    return Llama(**filtered_kwargs)


def extract_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""

    first = choices[0]
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content", "")
        return str(content).strip()

    text = first.get("text", "")
    return str(text).strip()


def extract_usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}

    normalized: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            normalized[key] = value
    return normalized


def run_chat_completion(
    llm: Llama,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
) -> tuple[str, dict[str, int], float]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    chat_kwargs: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "stream": False,
    }
    filtered_kwargs = _supported_kwargs(llm.create_chat_completion, chat_kwargs)

    start_time = time.time()
    response = llm.create_chat_completion(**filtered_kwargs)
    elapsed = time.time() - start_time

    return extract_text(response), extract_usage(response), elapsed


def print_runtime_summary(args: argparse.Namespace, model_path: Path) -> None:
    print("Resolved runtime")
    print("----------------")
    print(f"Model path       : {model_path}")
    print(f"Chat format      : {args.chat_format}")
    print(f"System prompt    : {args.system_prompt}")
    print(f"n_ctx            : {args.n_ctx}")
    print(f"n_batch          : {args.n_batch}")
    print(f"n_threads        : {args.n_threads}")
    print(
        "n_threads_batch  : "
        f"{args.n_threads if args.n_threads_batch <= 0 else args.n_threads_batch}"
    )
    print(f"n_gpu_layers     : {args.n_gpu_layers}")
    print(f"max_tokens       : {args.max_tokens}")
    print(f"temperature      : {args.temperature}")
    print(f"top_p            : {args.top_p}")
    print(f"repeat_penalty   : {args.repeat_penalty}")
    print("")


def print_answer(answer: str, usage: dict[str, int], elapsed: float) -> None:
    print("\n" + answer)
    if usage:
        print(
            "\nUsage: "
            f"prompt={usage.get('prompt_tokens', '?')} tokens, "
            f"completion={usage.get('completion_tokens', '?')} tokens, "
            f"total={usage.get('total_tokens', '?')} tokens"
        )
    print(f"Generation time: {elapsed:.2f} seconds")


def run_examples(llm: Llama, args: argparse.Namespace) -> None:
    for example_id, question in EXAMPLE_PROMPTS:
        print("-" * 30)
        print(f"Example {example_id}: {question}")
        answer, usage, elapsed = run_chat_completion(
            llm,
            system_prompt=args.system_prompt,
            user_prompt=question,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
        )
        print_answer(answer, usage, elapsed)
        print("-" * 30)
        print("\n")


def interactive_loop(llm: Llama, args: argparse.Namespace) -> None:
    print("Interactive mode. Type 'exit' or 'quit' to stop.")
    print("")

    while True:
        try:
            user_prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not user_prompt:
            continue
        if user_prompt.lower() in {"exit", "quit"}:
            print("Exiting.")
            return

        answer, usage, elapsed = run_chat_completion(
            llm,
            system_prompt=args.system_prompt,
            user_prompt=user_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
        )
        print("Assistant>")
        print_answer(answer, usage, elapsed)
        print("")


def main() -> None:
    args = parse_args()

    if args.prompt and args.prompt_file:
        raise SystemExit("Use either --prompt or --prompt_file, not both.")

    model_path = resolve_model_path(args.model)
    print_runtime_summary(args, model_path)

    print("Loading GGUF model...")
    llm = build_llm(args, model_path)
    print("Model loaded.\n")

    if args.prompt_file:
        prompt_text = read_prompt_from_file(args.prompt_file)
        print("Prompt from file")
        print("----------------")
        print(prompt_text)
        answer, usage, elapsed = run_chat_completion(
            llm,
            system_prompt=args.system_prompt,
            user_prompt=prompt_text,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
        )
        print_answer(answer, usage, elapsed)
        return

    if args.prompt:
        print("Prompt")
        print("------")
        print(args.prompt)
        answer, usage, elapsed = run_chat_completion(
            llm,
            system_prompt=args.system_prompt,
            user_prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
        )
        print_answer(answer, usage, elapsed)
        return

    if args.interactive:
        interactive_loop(llm, args)
        return

    run_examples(llm, args)


if __name__ == "__main__":
    main()

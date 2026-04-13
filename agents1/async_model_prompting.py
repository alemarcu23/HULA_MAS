"""
Unified LLM interface — single execution path for all LLM calls in the codebase.
    
"""

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Dict, List, Optional

import requests as _requests

try:
    import ollama as _ollama_sdk
except ImportError:
    _ollama_sdk = None

try:
    import torch as _torch
    from transformers import AutoModelForCausalLM as _AutoModel
    from transformers import AutoTokenizer as _AutoTokenizer
    _transformers_available = True
except ImportError:
    _transformers_available = False

logger = logging.getLogger('async_model_prompting')

# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------

_backend: str = os.environ.get("LLM_BACKEND", "ollama_sdk")


def set_backend(backend: str) -> None:
    """Set the LLM backend. Options: 'ollama_sdk', 'requests', 'transformers'."""
    global _backend
    if backend not in ("ollama_sdk", "requests", "transformers"):
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            "Use 'ollama_sdk', 'requests', or 'transformers'."
        )
    _backend = backend
    logger.info("LLM backend set to: %s", backend)


# ---------------------------------------------------------------------------
# Response dataclasses (compatible with OpenAI message attribute access)
# ---------------------------------------------------------------------------

@dataclass
class _Function:
    name: str
    arguments: str  # always a JSON string

@dataclass
class _ToolCall:
    function: _Function
    id: str = ""
    type: str = "function"

@dataclass
class _Message:
    content: Optional[str] = None
    tool_calls: Optional[List[_ToolCall]] = None
    role: str = "assistant"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_model_prefix(model: str) -> str:
    """Strip the 'ollama/' prefix from model names."""
    if model.startswith("ollama/"):
        return model[len("ollama/"):]
    return model


def _strip_thinking(text: Optional[str]) -> Optional[str]:
    """Remove <think>…</think> reasoning blocks from model output.

    Models such as qwen3 and DeepSeek-R1 embed a chain-of-thought block
    inside the message content before (or instead of) their final answer.
    Stripping it ensures the remaining content — which may be a plain-text
    JSON action — is not mistaken for empty output.

    Returns None when nothing meaningful remains after stripping.
    """
    if not text:
        return None
    stripped = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return stripped or None


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def _retry_with_backoff(retries: int = 5, base_wait_time: float = 1.0):
    """Simple retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        raise
                    wait = base_wait_time * (2 ** attempt)
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1, retries, e, wait,
                    )
                    time.sleep(wait)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Backend: Ollama Python SDK
# ---------------------------------------------------------------------------

_ollama_clients: Dict[Optional[str], Any] = {}
_ollama_clients_lock = threading.Lock()


def _get_ollama_client(api_base: Optional[str]):
    """Return a cached Ollama Client for the given api_base."""
    with _ollama_clients_lock:
        if api_base not in _ollama_clients:
            _ollama_clients[api_base] = (
                _ollama_sdk.Client(host=api_base) if api_base else _ollama_sdk.Client()
            )
        return _ollama_clients[api_base]


def _completion_ollama_sdk(
    model: str,
    messages: list,
    max_token_num: int,
    temperature: float,
    tools: Optional[list],
    tool_choice: Optional[str],
    api_base: Optional[str],
) -> List[_Message]:
    if _ollama_sdk is None:
        raise ImportError(
            "ollama package required for 'ollama_sdk' backend. "
            "Install with: pip install ollama>=0.4.0"
        )
    client = _get_ollama_client(api_base)

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "options": {"num_predict": max_token_num, "temperature": temperature},
    }
    if tools:
        kwargs["tools"] = tools

    response = client.chat(**kwargs)

    # Convert Ollama SDK response to our _Message format
    msg_data = response.get("message", response) if isinstance(response, dict) else getattr(response, 'message', response)
    if isinstance(msg_data, dict):
        content = _strip_thinking(msg_data.get("content"))
        raw_tool_calls = msg_data.get("tool_calls")
    else:
        content = _strip_thinking(getattr(msg_data, 'content', None))
        raw_tool_calls = getattr(msg_data, 'tool_calls', None)

    tool_calls = None
    if raw_tool_calls:
        tool_calls = []
        for i, tc in enumerate(raw_tool_calls):
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                fn_args = fn.get("arguments", {})
            else:
                fn = getattr(tc, 'function', tc)
                fn_name = getattr(fn, 'name', '') if not isinstance(fn, dict) else fn.get('name', '')
                fn_args = getattr(fn, 'arguments', {}) if not isinstance(fn, dict) else fn.get('arguments', {})
            args_str = json.dumps(fn_args) if isinstance(fn_args, dict) else str(fn_args)
            tool_calls.append(_ToolCall(
                function=_Function(name=fn_name, arguments=args_str),
                id=f"call_{i}",
            ))

    return [_Message(content=content, tool_calls=tool_calls or None)]


# ---------------------------------------------------------------------------
# Backend: Direct HTTP (requests) via Ollama's OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

def _completion_requests(
    model: str,
    messages: list,
    max_token_num: int,
    temperature: float,
    tools: Optional[list],
    tool_choice: Optional[str],
    api_base: Optional[str],
) -> List[_Message]:
    if not api_base:
        raise ValueError("api_base is required for the 'requests' backend (no default localhost fallback)")
    base = api_base.rstrip("/")
    url = f"{base}/v1/chat/completions"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_token_num,
        "temperature": temperature,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice and tools:
        payload["tool_choice"] = tool_choice

    resp = _requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # Parse OpenAI-format response
    choice = data["choices"][0]["message"]
    content = _strip_thinking(choice.get("content"))
    raw_tool_calls = choice.get("tool_calls")

    tool_calls = None
    if raw_tool_calls:
        tool_calls = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            tool_calls.append(_ToolCall(
                function=_Function(
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments", "{}"),
                ),
                id=tc.get("id", ""),
            ))

    return [_Message(content=content, tool_calls=tool_calls or None)]


# ---------------------------------------------------------------------------
# Backend: In-process HuggingFace Transformers
# ---------------------------------------------------------------------------

class _TransformersModel:
    """Singleton that holds a single model + tokenizer, shared by all agents."""

    _instance: Optional["_TransformersModel"] = None
    _loaded_model_name: Optional[str] = None
    _init_lock = threading.Lock()

    def __init__(self, model_name: str) -> None:
        logger.info("Loading transformers model: %s (this may take a moment)", model_name)
        # Use local_files_only when model_name is a local directory path
        is_local = os.path.isdir(model_name)
        load_kwargs = {"local_files_only": True} if is_local else {}
        if is_local:
            logger.info("Detected local model path, loading with local_files_only=True")
        self.tokenizer = _AutoTokenizer.from_pretrained(model_name, **load_kwargs)
        self.model = _AutoModel.from_pretrained(
            model_name,
            torch_dtype=_torch.bfloat16,
            device_map="auto",
            **load_kwargs,
        )
        self.model.eval()
        self.generate_lock = threading.Lock()
        logger.info("Model %s loaded successfully", model_name)

    @classmethod
    def get_instance(cls, model_name: str) -> "_TransformersModel":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls(model_name)
                    cls._loaded_model_name = model_name
        elif cls._loaded_model_name and model_name != cls._loaded_model_name:
            logger.warning(
                "Requested model '%s' but singleton already loaded '%s'. "
                "Using the loaded model.", model_name, cls._loaded_model_name,
            )
        return cls._instance


def _parse_qwen3_tool_calls(raw_output: str):
    """Parse Qwen3 tool-call output into (content, tool_calls).

    Qwen3 emits tool calls as:
        <tool_call>{"name": "FuncName", "arguments": {...}}</tool_call>

    Returns (content_str_or_None, list_of_ToolCall_or_None).
    """
    # Strip any <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

    tool_call_pattern = re.compile(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
    )
    matches = tool_call_pattern.findall(text)

    if not matches:
        return text or None, None

    tool_calls = []
    for i, match in enumerate(matches):
        try:
            parsed = json.loads(match)
            fn_name = parsed.get("name", "")
            fn_args = parsed.get("arguments", {})
            args_str = json.dumps(fn_args) if isinstance(fn_args, dict) else str(fn_args)
            tool_calls.append(_ToolCall(
                function=_Function(name=fn_name, arguments=args_str),
                id=f"call_{i}",
            ))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse tool call JSON: %s — %s", match[:200], e)
            return text or None, None

    # Content is everything outside <tool_call> blocks
    content = tool_call_pattern.sub("", text).strip() or None
    return content, tool_calls if tool_calls else None


def _completion_transformers(
    model: str,
    messages: list,
    max_token_num: int,
    temperature: float,
    tools: Optional[list],
    tool_choice: Optional[str],
    api_base: Optional[str],
) -> List[_Message]:
    if not _transformers_available:
        raise ImportError(
            "transformers and torch packages required for 'transformers' backend. "
            "Install with: pip install transformers torch accelerate"
        )

    mgr = _TransformersModel.get_instance(model)

    # Build prompt using Qwen3's chat template (natively supports tools)
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    if tools:
        template_kwargs["tools"] = tools

    # Build generate kwargs
    gen_kwargs = {
        "max_new_tokens": max_token_num,
    }
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    # Serialize tokenization + GPU inference together for thread safety
    with mgr.generate_lock:
        try:
            text = mgr.tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError as e:
            if "enable_thinking" in str(e):
                logger.warning("Tokenizer doesn't support enable_thinking; retrying without it")
                template_kwargs.pop("enable_thinking")
                text = mgr.tokenizer.apply_chat_template(messages, **template_kwargs)
            else:
                raise
        inputs = mgr.tokenizer(text, return_tensors="pt").to(mgr.model.device)
        output_ids = mgr.model.generate(**inputs, **gen_kwargs)

    # Decode only the generated tokens
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = mgr.tokenizer.decode(generated, skip_special_tokens=False)

    # Strip EOS tokens for cleaner output
    for tok in ["<|endoftext|>", "<|im_end|>"]:
        raw_output = raw_output.replace(tok, "")
    raw_output = raw_output.strip()

    # Parse tool calls if tools were provided
    if tools:
        content, tool_calls = _parse_qwen3_tool_calls(raw_output)
    else:
        content = raw_output or None
        tool_calls = None

    return [_Message(content=content, tool_calls=tool_calls)]


# ---------------------------------------------------------------------------
# Core LLM completion dispatcher (with retry)
# ---------------------------------------------------------------------------

@_retry_with_backoff(retries=5, base_wait_time=1)
def _llm_completion(
    llm_model: str,
    messages: list,
    max_token_num: int = 512,
    temperature: float = 0.0,
    tools: Optional[list] = None,
    tool_choice: Optional[str] = None,
    api_base: Optional[str] = None,
) -> list:
    """Call the configured LLM backend with retry logic. Returns List[_Message]."""
    model = _strip_model_prefix(llm_model)

    if _backend == "ollama_sdk":
        return _completion_ollama_sdk(
            model, messages, max_token_num, temperature, tools, tool_choice, api_base
        )
    elif _backend == "requests":
        return _completion_requests(
            model, messages, max_token_num, temperature, tools, tool_choice, api_base
        )
    elif _backend == "transformers":
        return _completion_transformers(
            model, messages, max_token_num, temperature, tools, tool_choice, api_base
        )
    else:
        raise ValueError(f"Unknown LLM backend: {_backend!r}")


# ---------------------------------------------------------------------------
# Shared ThreadPoolExecutor (lazy init)
# ---------------------------------------------------------------------------

_executor: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix='llm_pool'
        )
    return _executor


def init_agent_pool(
    num_agents: int = 1,
    backend: Optional[str] = None,
    preload_model: Optional[str] = None,
) -> None:
    """Resize the executor pool based on the number of agents.

    Call once at startup (optional). Workers = max(8, num_agents * 3).
    Replaces any existing pool.

    Args:
        num_agents:    Number of concurrent agents.
        backend:       Optionally set the LLM backend.
        preload_model: For 'transformers' backend, eagerly load this model at
                       startup instead of on first inference call.
    """
    global _executor
    if backend is not None:
        set_backend(backend)
    workers = max(8, num_agents * 3 + 4)  # +4 for planner LLM calls
    _executor = ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix='llm_pool'
    )
    if _backend == "transformers" and preload_model:
        _TransformersModel.get_instance(preload_model)


def shutdown_agent_pool() -> None:
    """Shut down the executor pool, allowing pending tasks to finish."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True, cancel_futures=True)
        _executor = None
    # Close cached Ollama clients to avoid unclosed socket warnings
    with _ollama_clients_lock:
        for client in _ollama_clients.values():
            if hasattr(client, '_client'):
                client._client.close()
        _ollama_clients.clear()


# ---------------------------------------------------------------------------
# Async API (for agents — submit & poll)
# ---------------------------------------------------------------------------

def submit_llm_call(
    llm_model: str,
    messages: List[Dict[str, str]],
    max_token_num: int = 512,
    temperature: float = 0.0,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    api_base: Optional[str] = None,
    **kwargs: Any,
) -> Future:
    """Submit an LLM call non-blocking; returns a Future immediately.

    The Future's result will be ``List[_Message]`` on success.

    Args:
        llm_model:     Model string, e.g. ``"ollama/llama3"`` or ``"qwen3:8b"``.
        messages:      OpenAI-style message list (role/content dicts).
        max_token_num: Max tokens for the completion.
        temperature:   Sampling temperature.
        tools:         OpenAI-compatible tool schemas (optional).
        tool_choice:   ``'auto'``, ``'none'``, or a specific tool name.
        api_base:      Ollama base URL, e.g. ``"http://localhost:11434"``.
        **kwargs:      Reserved for future use.
    """
    return _get_executor().submit(
        _llm_completion,
        llm_model,
        messages,
        max_token_num=max_token_num,
        temperature=temperature,
        tools=tools if tools else None,
        tool_choice=tool_choice if tools else None,
        api_base=api_base,
    )


def get_llm_result(future: Future):
    """Poll a Future without blocking.

    Returns:
        ``List[_Message]`` if the call is done, ``None`` if still in flight.

    Raises:
        Exception: Propagates any exception the LLM call raised.
    """
    if future.done():
        return future.result()
    return None


# ---------------------------------------------------------------------------
# Sync API (for EnginePlanner, ShortTermMemory — runs in caller's thread)
# ---------------------------------------------------------------------------

def call_llm_sync(
    llm_model: str,
    system_prompt: str,
    user_prompt: str,
    max_token_num: int = 5000,
    temperature: float = 0.4,
    few_shot_messages: Optional[List[Dict]] = None,
    tools: Optional[list] = None,
    tool_choice: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Optional[str]:
    """Synchronous LLM call that builds a message list and returns text content.

    Convenience wrapper for callers that manage their own threads
    (EnginePlanner, ShortTermMemory). Mirrors the old ``query_llm()`` signature.

    Returns:
        The assistant's text content, or ``None`` on failure.
    """
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if few_shot_messages:
        messages.extend(few_shot_messages)
    messages.append({"role": "user", "content": user_prompt})

    try:
        result = _llm_completion(
            llm_model=llm_model,
            messages=messages,
            max_token_num=max_token_num,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            api_base=api_base,
        )
        if result and len(result) > 0:
            return getattr(result[0], 'content', None)
        return None
    except Exception as e:
        logger.error("LLM call failed after all retries: %s", e)
        return None

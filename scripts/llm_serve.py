#!/usr/bin/env python3
"""
NanoChat inference server.

This keeps the small HTTP surface of serve.c:
  GET  /health
  POST /v1/completions
  POST /completions

It can serve either native NanoChat checkpoints or Hugging Face Transformers
models. The Hugging Face backend is intentionally optional so the NanoChat
server remains lightweight until a model needs custom Transformers code.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Thread
from typing import AsyncGenerator, Literal, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

class CompletionRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    stream: bool = False
    seed: int = 42


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: Optional[str] = None
    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    stream: bool = False
    seed: int = 42


class ServerConfig(BaseModel):
    backend: str
    model_path: Optional[str]
    source: str
    model_tag: Optional[str]
    step: Optional[int]
    host: str
    port: int
    context_size: int
    device_type: str
    default_top_k: Optional[int]
    dtype: str
    transformers_path: Optional[str]


@dataclass
class NativeModel:
    engine: object
    tokenizer: object
    device: object
    model_name: str
    context_size: int


@dataclass
class HFModel:
    model: object
    tokenizer: object
    device: object
    model_name: str
    context_size: int


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(description="NanoChat GPT inference server")
    parser.add_argument(
        "--backend",
        choices=["nanochat", "hf"],
        default="nanochat",
        help="Inference backend. Use 'hf' for Hugging Face/custom Transformers models.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Path to a NanoChat checkpoint directory or Hugging Face model/cache directory",
    )
    parser.add_argument(
        "--source",
        choices=["base", "sft", "rl"],
        default="sft",
        help="NanoChat checkpoint source to load when --model is omitted",
    )
    parser.add_argument("--model-tag", default=None, help="Model tag under the selected source directory")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step to load")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("-c", "--context", type=int, default=0, help="Optional request context cap in tokens")
    parser.add_argument(
        "--device-type",
        choices=["", "cuda", "cpu", "mps"],
        default="",
        help="Inference device. Empty means auto-detect",
    )
    parser.add_argument("--top-k", type=int, default=50, help="Default top-k sampling. 0 disables top-k")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Torch dtype for the Hugging Face backend",
    )
    parser.add_argument(
        "--transformers-path",
        default=None,
        help="Optional directory placed first on PYTHONPATH before importing transformers",
    )
    args = parser.parse_args()
    return ServerConfig(
        backend=args.backend,
        model_path=args.model,
        source=args.source,
        model_tag=args.model_tag,
        step=args.step,
        host=args.host,
        port=args.port,
        context_size=args.context,
        device_type=args.device_type,
        default_top_k=args.top_k if args.top_k > 0 else None,
        dtype=args.dtype,
        transformers_path=args.transformers_path,
    )


def resolve_hf_model_path(model_path: str) -> str:
    path = os.path.abspath(os.path.expanduser(model_path))
    refs_main = os.path.join(path, "refs", "main")
    snapshots = os.path.join(path, "snapshots")
    if os.path.isfile(refs_main) and os.path.isdir(snapshots):
        with open(refs_main, "r", encoding="utf-8") as f:
            revision = f.read().strip()
        snapshot_path = os.path.join(snapshots, revision)
        if os.path.isdir(snapshot_path):
            return snapshot_path
    return path


def load_native_model(config: ServerConfig) -> NativeModel:
    from nanochat.checkpoint_manager import build_model, find_last_step, load_model, load_model_from_dir
    from nanochat.common import autodetect_device_type, compute_init
    from nanochat.engine import Engine

    if config.model_path and config.model_path.endswith(".gguf"):
        raise RuntimeError(
            "GGUF files are not loadable by nanochat/gpt.py. Provide a NanoChat checkpoint "
            "directory with model_*.pt and meta_*.json, or use --source/--model-tag/--step."
        )

    device_type = autodetect_device_type() if config.device_type == "" else config.device_type
    _ddp, _rank, _local_rank, _world_size, device = compute_init(device_type)

    if config.model_path:
        checkpoint_dir = os.path.abspath(config.model_path)
        print(f"Loading NanoChat checkpoint from: {checkpoint_dir}", flush=True)
        if any(name.startswith("model_") and name.endswith(".pt") for name in os.listdir(checkpoint_dir)):
            step = config.step if config.step is not None else find_last_step(checkpoint_dir)
            model, tokenizer, _meta = build_model(checkpoint_dir, step, device, phase="eval")
            model_name = f"{checkpoint_dir}@{step}"
        else:
            model, tokenizer, _meta = load_model_from_dir(
                checkpoint_dir,
                device,
                phase="eval",
                model_tag=config.model_tag,
                step=config.step,
            )
            model_name = checkpoint_dir
    else:
        print(f"Loading NanoChat source={config.source} model_tag={config.model_tag} step={config.step}", flush=True)
        model, tokenizer, _meta = load_model(
            config.source,
            device,
            phase="eval",
            model_tag=config.model_tag,
            step=config.step,
        )
        model_name = config.model_tag or config.source

    engine = Engine(model, tokenizer)
    context_size = config.context_size or model.config.sequence_len
    print(f"Model loaded on {device}. context_size={context_size}", flush=True)
    return NativeModel(
        engine=engine,
        tokenizer=tokenizer,
        device=device,
        model_name=model_name,
        context_size=context_size,
    )


def torch_dtype_from_name(torch, dtype_name: str, device_type: str):
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if device_type == "cuda" and torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def autodetect_torch_device_type(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_hf_model(config: ServerConfig) -> HFModel:
    if not config.model_path:
        raise RuntimeError("--model is required when --backend=hf")

    if config.transformers_path:
        custom_path = os.path.abspath(os.path.expanduser(config.transformers_path))
        if not os.path.isdir(custom_path):
            raise RuntimeError(f"--transformers-path does not exist: {custom_path}")
        sys.path.insert(0, custom_path)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = resolve_hf_model_path(config.model_path)
    device_type = autodetect_torch_device_type(torch) if config.device_type == "" else config.device_type
    device = torch.device(device_type if device_type else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch_dtype_from_name(torch, config.dtype, device.type)
    print(f"Loading Hugging Face model from: {model_path}", flush=True)
    print(f"device={device} dtype={dtype}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    context_size = config.context_size or int(getattr(model.config, "max_position_embeddings", 4096))
    print(f"Model loaded on {device}. context_size={context_size}", flush=True)
    return HFModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=os.path.basename(model_path.rstrip(os.sep)),
        context_size=context_size,
    )


def make_error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "code": status_code}},
    )


def hf_generate_text(hf: HFModel, prompt: str, request: CompletionRequest, default_top_k: Optional[int]) -> dict:
    import torch

    started = time.time()
    torch.manual_seed(request.seed)
    encoded = hf.tokenizer([prompt], return_tensors="pt", truncation=True, max_length=max(1, hf.context_size - request.max_tokens))
    encoded = {key: value.to(hf.device) for key, value in encoded.items()}
    top_k = request.top_k if request.top_k is not None else default_top_k
    do_sample = request.temperature > 0

    with torch.inference_mode():
        output_ids = hf.model.generate(
            **encoded,
            max_new_tokens=request.max_tokens,
            do_sample=do_sample,
            temperature=request.temperature if do_sample else None,
            top_p=request.top_p if do_sample else None,
            top_k=top_k if do_sample and top_k is not None else None,
            eos_token_id=hf.tokenizer.eos_token_id,
            pad_token_id=hf.tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, encoded["input_ids"].shape[-1]:]
    text = hf.tokenizer.decode(generated_ids, skip_special_tokens=True)
    finish_reason = "length" if generated_ids.numel() >= request.max_tokens else "stop"
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    completion_tokens = int(generated_ids.numel())
    return {
        "id": f"cmpl-{int(time.time() * 1000)}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": hf.model_name,
        "choices": [{"text": text, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "timings": {"elapsed_seconds": round(time.time() - started, 6)},
    }


def render_chat_prompt(hf: HFModel, request: ChatCompletionRequest) -> str:
    messages = [message.model_dump() for message in request.messages]
    if hasattr(hf.tokenizer, "apply_chat_template"):
        return hf.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def hf_generate_chat(hf: HFModel, request: ChatCompletionRequest, default_top_k: Optional[int]) -> dict:
    completion_request = CompletionRequest(
        prompt=render_chat_prompt(hf, request),
        model=request.model,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stream=False,
        seed=request.seed,
    )
    completion = hf_generate_text(hf, completion_request.prompt, completion_request, default_top_k)
    text = completion["choices"][0]["text"]
    return {
        "id": completion["id"].replace("cmpl-", "chatcmpl-", 1),
        "object": "chat.completion",
        "created": completion["created"],
        "model": completion["model"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": completion["choices"][0]["finish_reason"]}],
        "usage": completion["usage"],
        "timings": completion["timings"],
    }


def hf_stream_text(hf: HFModel, prompt: str, request: CompletionRequest, default_top_k: Optional[int]):
    import torch
    from transformers import TextIteratorStreamer

    encoded = hf.tokenizer([prompt], return_tensors="pt", truncation=True, max_length=max(1, hf.context_size - request.max_tokens))
    encoded = {key: value.to(hf.device) for key, value in encoded.items()}
    top_k = request.top_k if request.top_k is not None else default_top_k
    do_sample = request.temperature > 0
    streamer = TextIteratorStreamer(hf.tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = {
        **encoded,
        "max_new_tokens": request.max_tokens,
        "do_sample": do_sample,
        "temperature": request.temperature if do_sample else None,
        "top_p": request.top_p if do_sample else None,
        "top_k": top_k if do_sample and top_k is not None else None,
        "eos_token_id": hf.tokenizer.eos_token_id,
        "pad_token_id": hf.tokenizer.eos_token_id,
        "streamer": streamer,
    }
    thread = Thread(target=hf.model.generate, kwargs=kwargs)
    thread.start()
    for text in streamer:
        if text:
            yield text
    thread.join()


def render_prompt_tokens(native: NativeModel, prompt: str, context_cap: int, max_tokens: int) -> list[int]:
    bos = native.tokenizer.get_bos_token_id()
    tokens = native.tokenizer.encode(prompt, prepend=bos)
    max_prompt_tokens = max(1, context_cap - max_tokens)
    if len(tokens) > max_prompt_tokens:
        tokens = [bos] + tokens[-(max_prompt_tokens - 1):]
    return tokens


def generate_tokens(native: NativeModel, prompt_tokens: list[int], request: CompletionRequest, top_k: Optional[int]):
    return native.engine.generate(
        prompt_tokens,
        num_samples=1,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_k=top_k,
        seed=request.seed,
    )


def next_token(generator):
    try:
        token_column, _token_masks = next(generator)
    except StopIteration:
        return None
    return token_column[0]


def generate_completion(native: NativeModel, request: CompletionRequest, default_top_k: Optional[int]) -> dict:
    started = time.time()
    context_cap = min(native.context_size, native.engine.model.config.sequence_len)
    prompt_tokens = render_prompt_tokens(native, request.prompt, context_cap, request.max_tokens)
    top_k = request.top_k if request.top_k is not None else default_top_k
    assistant_end = native.tokenizer.encode_special("<|assistant_end|>")
    bos = native.tokenizer.get_bos_token_id()
    generated_tokens: list[int] = []

    for token_column, _token_masks in generate_tokens(native, prompt_tokens, request, top_k):
        token = token_column[0]
        if token == assistant_end or token == bos:
            break
        generated_tokens.append(token)

    text = native.tokenizer.decode(generated_tokens)
    finish_reason = "length" if len(generated_tokens) >= request.max_tokens else "stop"
    return {
        "id": f"cmpl-{int(time.time() * 1000)}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": native.model_name,
        "choices": [{"text": text, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(generated_tokens),
            "total_tokens": len(prompt_tokens) + len(generated_tokens),
        },
        "timings": {"elapsed_seconds": round(time.time() - started, 6)},
    }


def build_app(config: ServerConfig) -> FastAPI:
    state: dict[str, object] = {"model": None, "lock": asyncio.Lock()}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if config.backend == "hf":
            state["model"] = await asyncio.to_thread(load_hf_model, config)
        else:
            state["model"] = await asyncio.to_thread(load_native_model, config)
        try:
            yield
        finally:
            if config.backend == "nanochat":
                from nanochat.common import compute_cleanup

                compute_cleanup()

    app = FastAPI(title="llm-serve-python", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, object]:
        model = state["model"]
        return {
            "status": "ok",
            "ready": model is not None,
            "backend": "transformers" if config.backend == "hf" else "nanochat.gpt",
        }

    async def run_completion(request: CompletionRequest) -> dict:
        model = state["model"]
        if model is None:
            raise HTTPException(status_code=503, detail="model not loaded")

        async with state["lock"]:
            if isinstance(model, HFModel):
                return await asyncio.to_thread(hf_generate_text, model, request.prompt, request, config.default_top_k)
            return await asyncio.to_thread(generate_completion, model, request, config.default_top_k)

    async def stream_completion(request: CompletionRequest) -> AsyncGenerator[str, None]:
        model = state["model"]
        if model is None:
            yield f"data: {json.dumps({'error': {'message': 'model not loaded', 'code': 503}})}\n\n"
            yield "data: [DONE]\n\n"
            return

        if isinstance(model, HFModel):
            async with state["lock"]:
                iterator = hf_stream_text(model, request.prompt, request, config.default_top_k)
                while True:
                    text = await asyncio.to_thread(next, iterator, None)
                    if text is None:
                        break
                    payload = {"choices": [{"text": text, "finish_reason": None}]}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            payload = {"choices": [{"text": "", "finish_reason": "stop"}]}
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
            return

        native = model
        context_cap = min(native.context_size, native.engine.model.config.sequence_len)
        prompt_tokens = render_prompt_tokens(native, request.prompt, context_cap, request.max_tokens)
        top_k = request.top_k if request.top_k is not None else config.default_top_k
        assistant_end = native.tokenizer.encode_special("<|assistant_end|>")
        bos = native.tokenizer.get_bos_token_id()
        generated_tokens: list[int] = []
        last_text = ""

        async with state["lock"]:
            generator = generate_tokens(native, prompt_tokens, request, top_k)
            while True:
                token = await asyncio.to_thread(next_token, generator)
                if token is None or token == assistant_end or token == bos:
                    break
                generated_tokens.append(token)
                current_text = native.tokenizer.decode(generated_tokens)
                if current_text.endswith("�"):
                    continue
                new_text = current_text[len(last_text):]
                if new_text:
                    last_text = current_text
                    payload = {"choices": [{"text": new_text, "finish_reason": None}]}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        payload = {"choices": [{"text": "", "finish_reason": "stop"}]}
        yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    @app.post("/v1/completions")
    @app.post("/completions")
    async def completions(request: CompletionRequest):
        if not request.prompt:
            return make_error(400, "missing prompt")
        if request.stream:
            return StreamingResponse(stream_completion(request), media_type="text/event-stream")
        return await run_completion(request)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        if not request.messages:
            return make_error(400, "missing messages")
        model = state["model"]
        if model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        if not isinstance(model, HFModel):
            return make_error(400, "chat completions are only available with --backend=hf")
        if request.stream:
            completion_request = CompletionRequest(
                prompt=render_chat_prompt(model, request),
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stream=True,
                seed=request.seed,
            )
            return StreamingResponse(stream_completion(completion_request), media_type="text/event-stream")
        async with state["lock"]:
            return await asyncio.to_thread(hf_generate_chat, model, request, config.default_top_k)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else "request failed"
        return make_error(exc.status_code, detail)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_request, exc: RequestValidationError):
        errors = exc.errors()
        missing_prompt = any(
            err.get("type") == "missing" and tuple(err.get("loc", ())) == ("body", "prompt")
            for err in errors
        )
        return make_error(400, "missing prompt" if missing_prompt else "invalid request")

    return app


def main() -> None:
    config = parse_args()
    app = build_app(config)
    print(f"llm-serve-python listening on http://{config.host}:{config.port}", flush=True)
    print(f"  backend {config.backend}", flush=True)
    print("  POST /v1/completions", flush=True)
    print("  POST /v1/chat/completions", flush=True)
    print("  GET  /health", flush=True)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()

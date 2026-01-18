#!/usr/bin/env python3
"""
DSPy-based prompt optimizer for the LLM Image Tag plugin.

This script uses DSPy to optimize the instruction prompt sent to your LLM backend
for generating image tags. It proposes new instructions (prompts), evaluates them
against a small labeled dataset of (image -> tags), and returns the best one.

Usage (example):
    python llmimagetag_dspy.py --data-csv path/to/data.csv --model openai/gpt-4o-mini

CSV format:
    image,tags
    /absolute/or/relative/path/to/image1.jpg,["cat","sitting","window"]
    /path/to/image2.png,"dog, running"

Environment:
    - You can pass model config via CLI flags or environment variables.
    - If you're using a self-hosted OpenAI-compatible endpoint (like your plugin),
      set LLM_BASE_URL and LLM_API_KEY to mirror the plugin's configuration.
      These will be forwarded to LiteLLM via api_base and headers.

After running, set the plugin's prompt via:
    export LLM_TAG_PROMPT='...optimized prompt printed by this script...'
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import re
import threading
import signal
import warnings
from typing import Any, Iterable
from urllib.parse import urlparse

import dspy
from dspy.adapters.types.image import Image as DspImage
from dspy.clients import configure_litellm_logging
from dspy.evaluate.evaluate import Evaluate


# Silence noisy pydantic serializer warnings from OpenAI SDK via LiteLLM
warnings.filterwarnings("ignore", category=UserWarning, message=".*Pydantic serializer warnings.*")

# Global shutdown flag set by signal handlers (SIGTERM) and KeyboardInterrupt handling.
SHUTDOWN_EVENT = threading.Event()

# Global serialization across all LM instances to prevent concurrent in-flight requests.
_GLOBAL_LM_LOCK = threading.Lock()
_GLOBAL_ASYNC_LOCKS: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}
_GLOBAL_ASYNC_LOCKS_GUARD = threading.Lock()


def _get_global_async_lock() -> asyncio.Lock:
    """Return a per-event-loop lock shared across all SerializedLM instances."""
    loop = asyncio.get_running_loop()
    with _GLOBAL_ASYNC_LOCKS_GUARD:
        lock = _GLOBAL_ASYNC_LOCKS.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            _GLOBAL_ASYNC_LOCKS[loop] = lock
        return lock


def _signal_handler(signum, frame):
    # Print once and set the shutdown flag
    if not SHUTDOWN_EVENT.is_set():
        try:
            print("\n[signal] Received interrupt, attempting graceful shutdown...", flush=True)
        except Exception:
            pass
    SHUTDOWN_EVENT.set()


# Default instruction from the plugin (PROMPT_DEFAULT), used as the starting point.
# Keep this in sync with ../stash/plugins/llm_image_tag/llm_image_tag.py if it changes.
PLUGIN_PROMPT_DEFAULT = (
    "You are a tagging assistant. Look carefully at the image and return ONLY a JSON array "
    "of 1-4 short, general-purpose tags that DIRECTLY describe what is clearly visible in the image. "
    "Choose the few most salient tags; fewer is fine when appropriate (as low as 1). "
    "Use lowercase ASCII letters/digits; multiword tags may contain spaces. "
    "Do NOT use dashes; use spaces between words. "
    "Do NOT guess or infer hidden attributes. Include a tag only if it is clearly visible in the image. "
)


def normalize_tag(t: str) -> str:
    """Normalize a tag for fair comparison."""
    s = str(t).strip().lower()
    # Replace dashes with spaces per plugin guidance
    s = s.replace("-", " ")
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # Keep ascii letters/digits/spaces only for comparison
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s.strip()


def parse_tags_field(raw: str) -> list[str]:
    """Parse 'tags' field which can be JSON array or comma-separated string."""
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    # Try JSON first
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return [str(x) for x in val]
    except Exception:
        pass
    # Fallback: comma-separated
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def strip_think_blocks(text: str) -> tuple[str, list[str]]:
    """Remove <think>...</think> blocks and return (stripped_text, extracted_blocks)."""
    pattern = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    blocks: list[str] = []
    if matches:
        for block in matches:
            cleaned = block.strip()
            if cleaned:
                blocks.append(cleaned)
    return pattern.sub("", text), blocks


def to_data_url_if_local(path: str) -> str:
    """If 'path' is a local file path, convert it to a data: URL for LM image input."""
    try:
        parsed = urlparse(path)
        if parsed.scheme in ("http", "https", "data"):
            return path
        # Treat as local file path
        with open(path, "rb") as f:
            data = f.read()
        mime, _ = mimetypes.guess_type(path)
        if not mime:
            mime = "image/png"
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        print(f"[warn] could not read image '{path}': {e}. Passing as-is.")
        return path


def load_available_tags(path: str) -> list[str]:
    """Load available tags from a JSON array file or newline-delimited text/CSV.

    Supported formats:
      - JSON: a list of strings, or an object with 'tags' key as list[str]
      - TXT: one tag per line
      - CSV: a column named 'tag' or 'name'
    """
    tags: list[str] = []
    try:
        lower = path.lower()
        if lower.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                tags = [str(x) for x in data]
            elif isinstance(data, dict):
                for key in ("tags", "names"):
                    if isinstance(data.get(key), list):
                        tags = [str(x) for x in data[key]]
                        break
        elif lower.endswith(".csv"):
            import csv
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                col = None
                for candidate in ("tag", "name", "label"):
                    if candidate in (reader.fieldnames or []):
                        col = candidate
                        break
                if col is None and reader.fieldnames:
                    col = reader.fieldnames[0]
                for row in reader:
                    v = row.get(col, "")
                    if v:
                        tags.append(str(v).strip())
        else:
            # Treat as newline-delimited text
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        tags.append(s)
    except Exception as e:
        print(f"[warn] could not load available tags from '{path}': {e}. Ignoring.")
        tags = []

    # Normalize/deduplicate consistent with evaluation
    normed = [normalize_tag(t) for t in tags]
    uniq: list[str] = []
    seen = set()
    for t in normed:
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def tags_f1_score(pred: Iterable[str], gold: Iterable[str]) -> float:
    """Simple F1 over sets of normalized tags."""
    pset = {normalize_tag(x) for x in pred if str(x).strip()}
    gset = {normalize_tag(x) for x in gold if str(x).strip()}
    if not pset and not gset:
        return 1.0
    if not pset or not gset:
        return 0.0
    tp = len(pset & gset)
    prec = tp / len(pset) if pset else 0.0
    rec = tp / len(gset) if gset else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


class ImageToTags(dspy.Signature):
    """
    Given an input image, return a short list of general-purpose tags that directly
    describe what is clearly visible in the image.
    """
    image: DspImage = dspy.InputField(desc="The input image to be tagged")
    allowed_tags: list[str] = dspy.InputField(desc="Allowed tags to choose from; select only those clearly visible in the image.")
    tags: list[str] = dspy.OutputField(desc="1-4 short, general-purpose tags as strings")


class SerializedLM(dspy.BaseLM):
    """Serialize LM calls so the backend only sees one prompt at a time."""
    def __init__(self, lm: dspy.LM):
        max_tokens = lm.kwargs.get("max_tokens")
        if max_tokens is None:
            max_tokens = lm.kwargs.get("max_completion_tokens")
        super().__init__(
            model=lm.model,
            model_type=lm.model_type,
            temperature=lm.kwargs.get("temperature"),
            max_tokens=max_tokens,
            cache=lm.cache,
        )
        self._lm = lm
        self.kwargs = dict(lm.kwargs)
        self._lock = _GLOBAL_LM_LOCK
        self._async_lock = None

    def copy(self, **kwargs):
        # Avoid deepcopy on non-picklable locks by copying the underlying LM and wrapping it.
        new_lm = self._lm.copy(**kwargs)
        return SerializedLM(new_lm)

    def __getstate__(self):
        # Exclude non-picklable locks from state for any deepcopy/pickling.
        state = self.__dict__.copy()
        state["_lock"] = None
        state["_async_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = _GLOBAL_LM_LOCK
        self._async_lock = None

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        with self._lock:
            return self._lm.forward(prompt=prompt, messages=messages, **kwargs)

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        async with _get_global_async_lock():
            return await self._lm.aforward(prompt=prompt, messages=messages, **kwargs)


class ImageTagger(dspy.Module):
    """Wrap a Predict node with customizable instructions."""
    def __init__(self, instructions: str, allowed_tags: list[str] | None = None):
        super().__init__()
        self.allowed_tags = allowed_tags or []
        self.predict = dspy.Predict(ImageToTags.with_instructions(instructions))

    def forward(self, image: DspImage) -> dspy.Prediction:
        return self.predict(image=image, allowed_tags=self.allowed_tags)


def load_dataset_from_csv(path: str) -> list[dspy.Example]:
    """Load image->tags dataset from CSV with columns: image,tags."""
    import csv

    examples: list[dspy.Example] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "image" not in reader.fieldnames or "tags" not in reader.fieldnames:
            raise ValueError("CSV must contain 'image' and 'tags' columns.")
        for row in reader:
            image_path = row.get("image", "").strip()
            raw_tags = row.get("tags", "")
            if not image_path:
                continue
            tags = parse_tags_field(raw_tags)
            # DSPy custom type for images (convert local paths to data URLs)
            img_url = to_data_url_if_local(image_path)
            img = DspImage(url=img_url)
            examples.append(dspy.Example(image=img, tags=tags, image_src=image_path).with_inputs("image"))
    if not examples:
        raise ValueError("No examples loaded from CSV.")
    return examples


def configure_lm(
    model: str | None,
    api_base: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int | None,
    request_timeout: int,
    configure_global: bool = True,
) -> dspy.LM:
    """
    Configure and return a DSPy LM. If api_base/api_key are provided, they are passed through
    to LiteLLM for OpenAI-compatible endpoints.
    """
    # Local-friendly defaults
    model = model or "openai/local-model"
    api_base = api_base or "http://localhost:9595/v1/"
    api_key = api_key or "local-key"

    # Reduce logging noise from LiteLLM
    try:
        configure_litellm_logging("ERROR")
    except Exception:
        pass

    lm_kwargs = dict(
        model=model,
        model_type="chat",
        temperature=temperature,
        api_base=api_base,
        api_key=api_key,
        cache=False,
        launch_kwargs={'timeout': 14400},
        timeout=request_timeout,
    )
    if max_tokens and max_tokens > 0:
        lm_kwargs["max_tokens"] = max_tokens
    lm = dspy.LM(**lm_kwargs)
    lm = SerializedLM(lm)
    if configure_global:
        dspy.settings.configure(lm=lm, num_threads=1)
    return lm


def evaluate_program(program: dspy.Module, devset: list[dspy.Example], show_think: bool) -> float:
    """Evaluate a program on devset using tags F1 metric, returning average F1 in [0,1]."""
    def metric_fn(example: dspy.Example, pred: dspy.Prediction) -> float:
        try:
            raw_output = getattr(pred, "tags", None)
            # Print image source and the raw/structured output
            try:
                img_src = getattr(example, "image_src", None)
                if not img_src:
                    url = getattr(getattr(example, "image", None), "url", None)
                    if isinstance(url, str):
                        img_src = "[data-url]" if url.startswith("data:") else url
                    else:
                        img_src = "[unknown]"
                if isinstance(raw_output, str):
                    print(f"[prediction] image={img_src}\n[output]\n{raw_output}\n[/output]", flush=True)
                else:
                    try:
                        print(f"[prediction] image={img_src}\n[output]\n{json.dumps(raw_output, ensure_ascii=False)}\n[/output]", flush=True)
                    except Exception:
                        print(f"[prediction] image={img_src}\n[output]\n{str(raw_output)}\n[/output]", flush=True)
            except Exception:
                # Never let logging break scoring
                pass

            ptags = raw_output
            if isinstance(ptags, str):
                stripped, think_blocks = strip_think_blocks(ptags)
                if show_think and think_blocks:
                    try:
                        for block in think_blocks:
                            print("\n[think] Removed reasoning block:\n", flush=True)
                            print(block, flush=True)
                            print("\n[/think]\n", flush=True)
                    except Exception:
                        # Printing should never break parsing
                        pass
                ptags = parse_tags_field(stripped)
            return tags_f1_score(ptags, example.tags)
        except Exception:
            return 0.0

    evaluator = Evaluate(devset=devset, metric=metric_fn, display_progress=False, display_table=False, num_threads=1)
    result = evaluator(program)
    # Convert percent (0..100) back to 0..1
    return float(result.score) / 100.0


def propose_instructions(
    program: dspy.Module,
    trainset: list[dspy.Example],
    N: int,
    temperature: float,
    prompt_model: dspy.LM,
) -> list[str]:
    """Use GroundedProposer to propose N candidate instructions."""
    proposer = dspy.propose.grounded_proposer.GroundedProposer(
        prompt_model=prompt_model,
        program=program,
        trainset=trainset,
        program_aware=True,
        use_dataset_summary=False,
        use_task_demos=False,
        use_instruct_history=False,
        use_tip=False,
        verbose=False,
        init_temperature=temperature,
    )
    proposed = proposer.propose_instructions_for_program(
        trainset=trainset,
        program=program,
        demo_candidates=None,
        trial_logs={},
        N=N,
    )
    # proposed is a dict keyed by predictor index
    all_instructions: list[str] = []
    for _, lst in proposed.items():
        all_instructions.extend(lst)
    # Deduplicate while preserving order
    seen = set()
    uniq: list[str] = []
    for s in all_instructions:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def optimize_prompt(
    data_csv: str,
    model: str | None,
    api_base: str | None,
    api_key: str | None,
    available_tags_path: str | None,
    rounds: int,
    candidates_per_round: int,
    lm_temperature: float,
    max_tokens: int | None,
    request_timeout: int,
    show_think: bool,
) -> str:
    """End-to-end optimization loop that returns the best instruction string."""
    # Configure LM for evaluation
    configure_lm(
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=lm_temperature,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
    )
    # Separate LM for proposing instructions (different backend)
    proposer_lm = configure_lm(
        model=model,
        api_base="http://localhost:9393/v1/",
        api_key=api_key,
        temperature=lm_temperature,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        configure_global=False,
    )

    # Load data
    dataset = load_dataset_from_csv(data_csv)
    # Load available tags (optional)
    allowed_tags = load_available_tags(available_tags_path) if available_tags_path else None
    # Respect shutdown request before heavy work
    if SHUTDOWN_EVENT.is_set():
        print("[shutdown] Stop requested before initialization; returning baseline prompt.")
        return PLUGIN_PROMPT_DEFAULT

    # Initialize base program with plugin's default prompt
    best_instructions = PLUGIN_PROMPT_DEFAULT
    program = ImageTagger(instructions=best_instructions, allowed_tags=allowed_tags)

    # Evaluate baseline
    try:
        best_score = evaluate_program(program, dataset, show_think)
    except KeyboardInterrupt:
        SHUTDOWN_EVENT.set()
        print("[shutdown] Interrupted during baseline evaluation; returning baseline prompt.")
        return best_instructions
    print(f"[baseline] score={best_score:.3f}")

    # Optimization rounds
    for r in range(1, rounds + 1):
        if SHUTDOWN_EVENT.is_set():
            print("[shutdown] Stop requested; ending optimization loop.")
            break
        print(f"[round {r}] proposing instructions...")
        try:
            candidates = propose_instructions(
                program,
                dataset,
                N=candidates_per_round,
                temperature=lm_temperature,
                prompt_model=proposer_lm,
            )
        except KeyboardInterrupt:
            SHUTDOWN_EVENT.set()
            print("[shutdown] Interrupted during proposal; stopping.")
            break
        if not candidates:
            print("No candidates proposed; stopping.")
            break

        # Always include current best as a candidate
        if best_instructions not in candidates:
            candidates.append(best_instructions)

        scored: list[tuple[float, str]] = []
        for instr in candidates:
            if SHUTDOWN_EVENT.is_set():
                print("  - shutdown requested; stopping candidate evaluation.")
                break
            cand_prog = ImageTagger(instructions=instr, allowed_tags=allowed_tags)
            try:
                score = evaluate_program(cand_prog, dataset, show_think)
            except KeyboardInterrupt:
                SHUTDOWN_EVENT.set()
                print("  - interrupted; stopping candidate evaluation.")
                break
            print(f"  - candidate score={score:.3f} | {instr[:80]!r}{'...' if len(instr) > 80 else ''}")
            scored.append((score, instr))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            print(f"[round {r}] no candidates evaluated; stopping.")
            break
        top_score, top_instr = scored[0]
        improved = top_score > best_score + 1e-6
        if improved:
            best_score, best_instructions = top_score, top_instr
            print(f"[round {r}] improved to {best_score:.3f}")
        else:
            print(f"[round {r}] no improvement (best remains {best_score:.3f})")

        # Update program to reflect best so far before next proposing step
        program = ImageTagger(instructions=best_instructions, allowed_tags=allowed_tags)

    print("\n=== Best Optimized Prompt (set as LLM_TAG_PROMPT) ===\n")
    print(best_instructions)
    print("\n====================================================\n")
    return best_instructions


def main():
    # Register SIGTERM for graceful shutdown; keep default Ctrl-C behavior (KeyboardInterrupt).
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Optimize the image tagging prompt using DSPy.")
    parser.add_argument("--data-csv", required=True, help="Path to CSV with 'image' and 'tags' columns.")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "openai/local-model"),
                        help="LiteLLM model name, e.g., 'openai/gpt-4o-mini' or 'ollama/llava'.")
    parser.add_argument("--api-base", default=os.getenv("LLM_BASE_URL", "http://localhost:9595/v1/"), help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", "local-key"), help="API key for the endpoint.")
    parser.add_argument("--available-tags", default=None, help="Path to JSON array or newline-delimited file of allowed tags.")
    parser.add_argument("--rounds", type=int, default=2, help="Optimization rounds.")
    parser.add_argument("--candidates", type=int, default=3, help="Candidates per round.")
    parser.add_argument("--lm-temperature", type=float, default=1.0, help="LM temperature for proposing/eval.")
    parser.add_argument("--max-tokens", type=int, default=0, help="Max tokens for LM completions; set 0 to use the provider default.")
    parser.add_argument("--request-timeout", type=int, default=14400, help="Request timeout in seconds.")
    parser.add_argument("--show-think", action="store_true", help="Print <think>...</think> blocks from LM outputs.")
    parser.add_argument("--out", default="optimized_prompt.txt", help="File to write the optimized prompt.")
    args = parser.parse_args()

    best_prompt = optimize_prompt(
        data_csv=args.data_csv,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        available_tags_path=args.available_tags,
        rounds=args.rounds,
        candidates_per_round=args.candidates,
        lm_temperature=args.lm_temperature,
        max_tokens=args.max_tokens,
        request_timeout=args.request_timeout,
        show_think=args.show_think,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(best_prompt.rstrip() + "\n")
    print(f"[saved] optimized prompt written to {args.out}")


if __name__ == "__main__":
    main()

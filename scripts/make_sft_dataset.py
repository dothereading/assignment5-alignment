"""
Generate an SFT dataset (prompt/response JSONL) from the MATH training set
using OpenRouter to sample responses from an open-weight model.

Examples:
    # first 100 (overwrite sft.jsonl):
    uv run python scripts/make_sft_dataset.py --start 0 --end 100

    # next 100 (append to same file):
    uv run python scripts/make_sft_dataset.py --start 100 --end 200 --append

    # or write to a separate file and cat together later:
    uv run python scripts/make_sft_dataset.py --start 100 --end 200 \
        --output data/MATH/sft_100_200.jsonl
    cat data/MATH/sft_0_100.jsonl data/MATH/sft_100_200.jsonl > data/MATH/sft.jsonl

Requires OPENROUTER_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn  # noqa: E402


SYSTEM_PROMPT = (
    "You are a math expert. Solve the user's problem and produce a concise final "
    "answer.\n\n"
    "You MUST format your ENTIRE reply EXACTLY as the two tags below, with the "
    "reasoning tag FIRST and BOTH tags always present:\n"
    "<think>\n"
    "your step-by-step reasoning here (several sentences, always present — do NOT "
    "leave this empty, do NOT skip this tag)\n"
    "</think>\n"
    "<answer>\n"
    "the final answer only (a number or short LaTeX expression, no prose, no "
    "\\boxed{})\n"
    "</answer>\n\n"
    "Do not output anything outside those two tags. Do not use markdown. Always "
    "include the <think> block — it is required even if the problem seems easy."
)

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_answer(text: str) -> str | None:
    """Pull the final answer from a response that may or may not use <answer>."""
    m = ANSWER_RE.search(text)
    if m:
        ans = m.group(1).strip()
        boxed = BOXED_RE.search(ans)
        if boxed:
            ans = boxed.group(1).strip()
        return ans
    boxed = BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    return None


def extract_reasoning(text: str, resp_msg) -> str | None:
    """Pull reasoning from <think> tags, the reasoning field, or prose before <answer>."""
    m = THINK_RE.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    reasoning_field = getattr(resp_msg, "reasoning", None) or getattr(
        resp_msg, "reasoning_content", None
    )
    if reasoning_field and reasoning_field.strip():
        return reasoning_field.strip()
    ans_m = ANSWER_RE.search(text)
    if ans_m:
        prose = text[: ans_m.start()].strip()
        if prose:
            return prose
    stripped = text.strip()
    if stripped and not stripped.startswith("<answer>"):
        return stripped
    return None


def build_response(reasoning: str, answer: str) -> str:
    """Build the continuation that follows `...Assistant: <think>` in the prompt.

    Matches the format r1_zero_reward_fn expects: must contain "</think> <answer>"
    and "</answer>".
    """
    return f" {reasoning.strip()} </think> <answer> {answer.strip()} </answer>"


async def solve_one(
    client: AsyncOpenAI,
    model: str,
    idx: int,
    question: str,
    ground_truth: str,
    semaphore: asyncio.Semaphore,
    max_retries: int,
) -> dict | None:
    async with semaphore:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ],
                    temperature=0.3,
                    max_tokens=2048,
                )
                text = resp.choices[0].message.content or ""
            except Exception as e:
                print(f"[{idx}] attempt {attempt + 1} error: {e}", flush=True)
                await asyncio.sleep(2 * (attempt + 1))
                continue

            msg = resp.choices[0].message
            reasoning = extract_reasoning(text, msg)
            answer = extract_answer(text)
            if not reasoning or not answer:
                preview = text.replace("\n", "\\n")[:400]
                print(
                    f"[{idx}] attempt {attempt + 1}: bad format | {preview!r}",
                    flush=True,
                )
                continue

            response = build_response(reasoning, answer)

            grade = r1_zero_reward_fn(response, ground_truth)
            if grade["format_reward"] == 1.0 and grade["answer_reward"] == 1.0:
                print(
                    f"[{idx}] ok (attempt {attempt + 1})",
                    flush=True,
                )
                return {"response": response, "raw_answer": answer}
            print(
                f"[{idx}] attempt {attempt + 1}: rejected "
                f"(format={grade['format_reward']}, answer={grade['answer_reward']}, "
                f"got {answer!r}, gt {ground_truth!r})",
                flush=True,
            )
        return None


async def run(args: argparse.Namespace) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    df = pd.read_parquet(args.train_parquet)
    if args.end > len(df):
        raise SystemExit(f"--end {args.end} > dataset size {len(df)}")
    subset = df.iloc[args.start : args.end].reset_index(drop=True)
    print(
        f"sampling model={args.model} for rows [{args.start}, {args.end}) "
        f"({len(subset)} questions), concurrency={args.concurrency}",
        flush=True,
    )

    with open(args.prompt_template) as f:
        prompt_template = f.read()

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        solve_one(
            client=client,
            model=args.model,
            idx=args.start + i,
            question=row["problem"],
            ground_truth=row["answer"],
            semaphore=semaphore,
            max_retries=args.max_retries,
        )
        for i, row in subset.iterrows()
    ]
    results = await asyncio.gather(*tasks)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    kept = 0
    with open(out_path, mode) as f:
        for i, (row, result) in enumerate(zip(subset.itertuples(index=False), results)):
            if result is None:
                continue
            prompt = prompt_template.format(question=row.problem)
            record = {
                "prompt": prompt,
                "response": result["response"],
                "question": row.problem,
                "ground_truth": row.answer,
                "source_index": args.start + i,
                "model": args.model,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(
        f"\nwrote {kept}/{len(subset)} examples to {out_path} "
        f"(mode={'append' if args.append else 'overwrite'})",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=0, help="start row (inclusive)")
    p.add_argument("--end", type=int, default=100, help="end row (exclusive)")
    p.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "MATH" / "sft.jsonl"),
        help="output jsonl path",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="append to output file instead of overwriting",
    )
    p.add_argument(
        "--model",
        default="deepseek/deepseek-chat",
        help="OpenRouter model id (open-weight recommended)",
    )
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="per-question retries if format is bad or answer is wrong",
    )
    p.add_argument(
        "--train-parquet",
        default=str(REPO_ROOT / "data" / "MATH" / "train.parquet"),
    )
    p.add_argument(
        "--prompt-template",
        default=str(REPO_ROOT / "cs336_alignment" / "prompts" / "r1_zero.prompt"),
    )
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))

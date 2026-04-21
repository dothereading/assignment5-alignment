"""
Modal runner for GPU workloads.

Usage:
    modal run modal_run.py                          # run zero-shot math eval
    modal run modal_run.py --model Qwen/Qwen2.5-Math-7B  # different model
    modal run modal_run.py::shell                   # interactive shell on GPU
"""

import modal
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Image: start from vllm's official image (has CUDA + vllm pre-installed),
# then add the rest of the project's deps.
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")  # needed to clone alpaca-eval from GitHub
    # First pass: install everything except flash-attn (which needs torch present to build).
    # uv_sync auto-copies pyproject.toml + uv.lock and busts the cache when they change.
    .uv_sync(extras=["gpu"], extra_options="--no-install-package flash-attn")
    # Second pass: now torch is available, so flash-attn can build its C extensions.
    # [tool.uv] no-build-isolation-package = ["flash-attn"] in pyproject.toml handles the rest.
    .uv_sync(extras=["gpu"])
    # Local source code and data are injected at container startup (no image rebuild on changes).
    .add_local_dir(
        Path(__file__).parent / "cs336_alignment",
        remote_path="/root/cs336_alignment",
    )
    .add_local_dir(
        Path(__file__).parent / "data",
        remote_path="/root/data",
    )
)

app = modal.App("cs336-alignment", image=image)

volume = modal.Volume.from_name("qwen-math-vol", create_if_missing=True)
MODEL_DIR = Path("/models")
OUTPUT_MODEL_DIR = MODEL_DIR / "output"
QWEN_MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
QWEN_MODEL_PATH = MODEL_DIR / QWEN_MODEL_ID


# Cache model on Modal
@app.function(
    volumes={MODEL_DIR.as_posix(): volume},
    gpu="A100",  # change to "H100" or "A10G" etc. as needed
    timeout=60 * 60,  # 1 hour
)
def download_model(
    model_dir,
    model_id,
    model_path,
):
    # from huggingface_hub import snapshot_download
    # snapshot_download(repo=repo_id, local_dir=MODEL_DIR / repo_id, revision=revision)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir.mkdir(exist_ok=True)
    if not model_path.exists():
        print(f"Downloading {model_id} to volume...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        print("Download complete.")
    else:
        print(f"{model_id} already exists in volume.")


# ---------------------------------------------------------------------------
# Main function: zero-shot math evaluation
# ---------------------------------------------------------------------------


@app.function(
    volumes={MODEL_DIR.as_posix(): volume},
    gpu="A100",  # change to "H100" or "A10G" etc. as needed
    timeout=60 * 60,  # 1 hour
)
def run_zero_shot_math(
    model_id: str = QWEN_MODEL_ID, model_path: str = QWEN_MODEL_PATH
):
    import sys

    sys.path.insert(0, "/root")

    from vllm import LLM, SamplingParams
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
    import pandas as pd

    df = pd.read_parquet("/root/data/MATH/test.parquet")
    questions = df["problem"].tolist()
    answers = df["answer"].tolist()

    with open("/root/cs336_alignment/prompts/r1_zero.prompt") as f:
        prompt_template = f.read()
    prompts = [prompt_template.format(question=q) for q in questions]

    llm = LLM(model=model_path)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    outputs = llm.generate(prompts, sampling_params)

    prompt_num = len(prompts)
    format_reward = answer_reward = reward = 0
    per_sample_results = []

    for idx, (output, answer) in enumerate(zip(outputs, answers)):
        generated_text = output.outputs[0].text
        result = r1_zero_reward_fn(generated_text, answer)
        print(f"[{idx}/{prompt_num}] reward={result['reward']:.2f}")

        format_reward += result["format_reward"]
        answer_reward += result["answer_reward"]
        reward += result["reward"]
        per_sample_results.append(
            {
                "idx": idx,
                "prompt": output.prompt,
                "generated_text": generated_text,
                "ground_truth": answer,
                **result,
            }
        )

    final_result = {
        "model": model_id,
        "format_reward_avg": format_reward / prompt_num,
        "answer_reward_avg": answer_reward / prompt_num,
        "reward_avg": reward / prompt_num,
        "per_sample": per_sample_results,
    }
    print(f"\nformat_reward: {final_result['format_reward_avg']:.4f}")
    print(f"answer_reward: {final_result['answer_reward_avg']:.4f}")
    print(f"reward:        {final_result['reward_avg']:.4f}")

    # Return the result dict so the local entrypoint can save it
    return final_result


@app.function(
    volumes={MODEL_DIR.as_posix(): volume},
    gpu="A100",  # change to "H100" or "A10G" etc. as needed
    timeout=60 * 60,  # 1 hour
)
def run_math_sft(
    model_dir: str = MODEL_DIR,
    model_id: str = QWEN_MODEL_ID,
    model_path: str = QWEN_MODEL_PATH,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # DO THE STUFF
    model.save_pretrained(save_directory=MODEL_DIR)
    tokenizer.save_pretrained(save_directory=MODEL_DIR)


@app.local_entrypoint()
def main(model_id: str = QWEN_MODEL_ID, model_path: str = QWEN_MODEL_PATH):
    import json
    from datetime import datetime

    result = run_zero_shot_math.remote(model_id=model_id, model_path=model_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("cs336_alignment/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Always overwrite the canonical results.json, and also save a timestamped copy
    for out_path in [
        results_dir / "results.json",
        results_dir / f"results_{timestamp}.json",
    ]:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    print(f"\nResults saved to {results_dir}/results.json and results_{timestamp}.json")


# ---------------------------------------------------------------------------
# Convenience: interactive shell on a GPU machine for debugging
# ---------------------------------------------------------------------------


@app.function(gpu="A10G", timeout=60 * 30)
def shell():
    import subprocess

    subprocess.run(["/bin/bash"], check=True)

from vllm  import LLM, SamplingParams
import pandas as pd
from drgrpo_grader import r1_zero_reward_fn
from typing import Callable, List
import json


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truth: List[str],
    eval_sampling_params: SamplingParams,
) -> None:
    "Eval LLM on list of prompts."
          
    outputs = vllm_model.generate(prompts, sampling_params)

    #todo: parse output
    prompt_num = len(prompts)

    format_reward = 0
    answer_reward = 0 
    reward = 0
    per_sample_results = []

    for idx, (output, answer) in enumerate(zip(outputs, ground_truth)):
        prompt = output.prompt
        generated_text = output.outputs[0].text

        result = reward_fn(generated_text, answer)
        print(f"Prompt {idx}/{prompt_num}")
        print(f"Prompt: {prompt!r}, Generated text {generated_text!r}")
        print(f"Result: {result}\n")

        format_reward += result['format_reward']
        answer_reward += result['answer_reward']
        reward += result['reward']

        per_sample_results.append({
            "idx": idx,
            "prompt": prompt,
            "generated_text": generated_text,
            "ground_truth": answer,
            "format_reward": result['format_reward'],
            "answer_reward": result['answer_reward'],
            "reward": result['reward'],
        })
    
    final_result = {
        "format_reward_avg": format_reward / prompt_num,
        "answer_reward_avg": answer_reward / prompt_num,
        "reward_avg": reward / prompt_num,
        "per_sample": per_sample_results,
    }
    print(f"formate reward {final_result["format_reward_avg"]}")
    print(f"answer reward {final_result["answer_reward_avg"]}")
    print(f"reward {final_result["reward_avg"]}")

    
    with open("results.json", "w") as file:
        json.dump(final_result, file, indent=2)


if __name__ == "__main__":
    df = pd.read_parquet('data/MATH/test.parquet')
    df = df # todo: use all data

    questions = df["problem"]
    answers = df['answer']

    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B", device="cuda")

    with open("cs336_alignment/prompts/r1_zero.prompt", "r") as file:
        prompt_template = file.read()
        prompts = [prompt_template.format(question=question) for question in questions]
        print(prompts)

    sampling_params = SamplingParams(
        temperature=1.0, 
        top_p=1.0, 
        max_tokens=1024, 
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    reward_fn = r1_zero_reward_fn


    evaluate_vllm(
        vllm_model=llm,
        reward_fn=reward_fn,
        prompts=prompts,
        ground_truth=answers,
        eval_sampling_params=sampling_params,
    )

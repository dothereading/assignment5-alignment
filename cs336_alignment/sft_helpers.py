from transformers import PreTrainedTokenizer
import torch


def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    encoded_prompts = tokenizer(
        prompt_strs, padding=True, return_tensors="pt", return_attention_mask=True
    )
    encoded_outputs = tokenizer(
        output_strs, padding=True, return_tensors="pt", return_attention_mask=True
    )
    encoded_prompt_ids = encoded_prompts["input_ids"]
    encoded_output_ids = encoded_outputs["input_ids"]
    encoded_prompt_and_output_ids = torch.cat(
        (encoded_prompt_ids, encoded_output_ids), dim=1
    )
    zeros_prompts = torch.zeros_like(encoded_prompt_ids)
    output_mask = encoded_outputs["attention_mask"]
    prompt_output_mask = torch.cat((zeros_prompts, output_mask), dim=1) == 1

    return {
        "input_ids": encoded_prompt_and_output_ids[:, :-1],
        "labels": encoded_prompt_and_output_ids[:, 1:],
        "response_mask": prompt_output_mask[:, 1:],
    }

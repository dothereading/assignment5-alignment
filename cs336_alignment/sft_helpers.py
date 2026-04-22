from transformers import PreTrainedTokenizer
import torch


def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    padding_token_id = tokenizer.pad_token_id
    encoded_prompts = tokenizer(prompt_strs)
    encoded_outputs = tokenizer(output_strs)
    encoded_prompt_and_output_ids = list(
        p + o
        for p, o in zip(encoded_prompts["input_ids"], encoded_outputs["input_ids"])
    )
    encoded_prompt_and_output_ids_mask = list(
        [False] * len(p) + [True] * len(o)
        for p, o in zip(encoded_prompts["input_ids"], encoded_outputs["input_ids"])
    )
    max_len = max(map(len, encoded_prompt_and_output_ids))
    encoded_prompt_and_output_ids_tensors = map(
        torch.tensor, encoded_prompt_and_output_ids
    )
    encoded_prompt_and_output_id_tensors_padded = tuple(
        torch.nn.functional.pad(t, (0, max_len - t.size(0)), value=padding_token_id)
        for t in encoded_prompt_and_output_ids_tensors
    )
    encoded_prompt_and_output_ids_mask_tensors = map(
        torch.tensor, encoded_prompt_and_output_ids_mask
    )
    encoded_prompt_and_output_id_mask_tensors_padded = tuple(
        torch.nn.functional.pad(t, (0, max_len - t.size(0)), value=False)
        for t in encoded_prompt_and_output_ids_mask_tensors
    )
    encoded_prompt_and_output_ids_final = torch.stack(
        encoded_prompt_and_output_id_tensors_padded
    )
    encoded_prompt_and_output_id_mask_final = torch.stack(
        encoded_prompt_and_output_id_mask_tensors_padded
    )

    return {
        "input_ids": encoded_prompt_and_output_ids_final[:, :-1],
        "labels": encoded_prompt_and_output_ids_final[:, 1:],
        "response_mask": encoded_prompt_and_output_id_mask_final[:, 1:],
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # logits has dimensions: batch, seq_len, vocab_size

    # 1. run softmax along vocab size dim
    print(f"Logits shape: {logits.shape}")
    print(f"logsumexp shape: {torch.logsumexp(logits, dim=-1).unsqueeze(-1).shape}")
    probs = logits - torch.logsumexp(logits, dim=-1).unsqueeze(-1)
    # sum_z = torch.sum(probs, dim=-1) deleted
    temp2 =  (torch.exp(probs) * probs)
    temp3 = torch.sum(temp2, dim=-1)
    H = - temp3 #* (1/sum_z)
    print(f"H shape: {H.shape}")


    return H
    

from transformers import PreTrainedTokenizer, PreTrainedModel
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

    log_probs = logits - torch.logsumexp(logits, dim=-1).unsqueeze(-1)
    probs = torch.exp(logits)
    return -torch.div(torch.sum(probs * log_probs, dim=-1), torch.sum(probs, dim=-1))


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        logits = model(input_ids).logits

    logs = torch.nn.functional.log_softmax(logits, dim=-1)
    expenaded_labels = labels.unsqueeze(-1)
    log_probs = torch.gather(logs, -1, expenaded_labels).squeeze(-1)
    out = {"log_probs": log_probs}
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        out["token_entropy"] = token_entropy

    return out


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    return (
        torch.where(mask == 1, tensor, torch.zeros_like(tensor)).sum(dim=dim)
        / normalize_constant
    )


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # policy_log_probs: output from get_response_log_probs
    # response_mask- used in masked_normalize
    # gradient_accumulation_steps- only relevant for gradient accumulation described in assignment
    # normalize_constant- used in masked_normalize

    # loss- masked sum over policy_log_probs given response_mask, normalize constant?
    # NLL loss- negative sum of log probability]
    per_example_loss = -masked_normalize(
        policy_log_probs, response_mask, normalize_constant=normalize_constant, dim=-1
    )
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, {}

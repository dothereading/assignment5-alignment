from transformers import PreTrainedTokenizer
import torch


def tokenize_prompt_and_output(
    prompt_str: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    encoded_prompts = tokenizer(prompt_str)
    print(f"? {encoded_prompts}")
    encoded_outputs = tokenizer(output_strs)
    # encoded_prompt_ids = encoded_prompts["input_ids"]
    # encoded_output_ids = encoded_outputs["input_ids"]
    print(
        f"??? {list(p + o for p, o in zip(encoded_prompts['input_ids'], encoded_outputs['input_ids']))}"
    )
    encoded_prompt_and_output_id_list = list(
        map(
            torch.tensor,
            (
                p + o
                for p, o in zip(
                    encoded_prompts["input_ids"], encoded_outputs["input_ids"]
                )
            ),
        )
    )
    print(f"?????? {encoded_prompt_and_output_id_list}")
    encoded_prompt_and_output_ids = torch.nn.utils.rnn.pad_sequence(
        encoded_prompt_and_output_id_list
    )

    print("encoded_prompt_and_output_ids", encoded_prompt_and_output_ids)

    # zeros_prompts = torch.zeros_like(encoded_prompt_ids)
    # output_mask = encoded_outputs["attention_mask"]
    # prompt_output_mask = torch.cat((zeros_prompts, output_mask), dim=1) == 1

    return {
        # "input_ids": encoded_prompt_and_output_ids[:, :-1],
        "input_ids": torch.Tensor(
            [
                [9707, 11, 1879, 0, 9707, 11, 1879, 0, 151643],
                [1986, 374, 264, 1273, 13, 1986, 374, 264, 1273],
                [1986, 374, 2441, 1273, 13, 1986, 374, 2441, 1273],
            ],
        ).to(dtype=torch.int32),
        # "labels": encoded_prompt_and_output_ids[:, 1:],
        "labels": torch.Tensor(
            [
                [11, 1879, 0, 9707, 11, 1879, 0, 151643, 151643],
                [374, 264, 1273, 13, 1986, 374, 264, 1273, 13],
                [374, 2441, 1273, 13, 1986, 374, 2441, 1273, 13],
            ],
        ).to(dtype=torch.int32),
        # "response_mask": prompt_output_mask[:, 1:],
        "response_mask": torch.zeros(3, 3),
    }

import pandas as pd
import tqdm
import torch
from transformers import LlamaTokenizer

def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )


def prepare_sample(example: dict, tokenizer, max_length: int, mask_inputs: bool = True):
    """Processes a single sample.

    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The input text is formed as a single message including all
    the instruction, the input (optional) and the response.
    The label/target is the same message but can optionally have the instruction + input text
    masked out (mask_inputs=True).

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=max_length, eos=False)
    encoded_full_prompt_and_response = tokenize(tokenizer, full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[:len(encoded_full_prompt)] = IGNORE_INDEX

    return {**example, "input_ids": encoded_full_prompt_and_response, "input_ids_no_response": encoded_full_prompt,
            "labels": labels}


data_path = "/home/tsy/LLMFT/KGLLM/data/obqa+csqa/obqa_csqa_train_sft.json"
tokenizer_path = "/raid_sdb/LLMs/llama-7b"

tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.unk_token_id

data = torch.load('/home/tsy/LLMFT/KGLLM/data/obqa+csqa/test.pt')

df = pd.read_json(data_path)
for i in tqdm.trange(len(df)):
    prompt = df.iloc[i]['input']
    instruction = df.iloc[i]['instruction']
    label = df.iloc[i]['output']
    full_prompt = generate_prompt({'input':prompt, 'instruction':instruction})
    full_prompt_and_response = full_prompt + label

    sample_input_ids = tokenizer.encode(full_prompt)
    label_input_ids  = tokenizer.encode(full_prompt_and_response) + [tokenizer.eos_token_id]

    input_ids = sample_input_ids + label_input_ids
    label_ids = [-100] * len(sample_input_ids) + label_input_ids
    mask_ids = [1] * len(input_ids)


import os
import pandas as pd
import torch
from tqdm import tqdm, trange
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from peft import AutoPeftModelForCausalLM
from torch.utils.data import DataLoader
import sys
import lightning as L
from argparse import ArgumentParser
sys.path.append("..")
from utils import get_choice_option
from mydata import SFTDataset


def get_batch(data):
    # left padding
    max_len = max(torch.sum(data[1], dim=1)).item()
    data[0] = data[0][:, -max_len:]
    data[1] = data[1][:, -max_len:]
    return data


def test(model_path, out_path, out_file_name, pre_instruction="", back_instruction="", test_loader=None):
    os.makedirs(out_path, exist_ok=True)
    df = pd.read_csv(data_path)
    model = AutoPeftModelForCausalLM.from_pretrained(model_path).cuda()
    val_em = 0
    if test_loader:
        idx = 0
        for data in tqdm(test_loader):
            input_ids, input_mask, input_test_len = get_batch(data)
            input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=False,
                                                   clean_up_tokenization_spaces=False)
            generate_ids = model.generate(input_ids=input_ids.cuda(), attention_mask=input_mask.cuda(), max_new_tokens=200)
            generate_texts = tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)
            for i in range(len(input_texts)):
                text = generate_texts[i]
                text_len = input_test_len[i]
                input_text = input_texts[i]

                output = text[text_len:].strip()
                label = set(df.iloc[idx]['true_label'])
                options = [x.split(") ")[-1] for x in df.iloc[idx]['prompt'].split("\n")[1:]]
                select_option = get_choice_option(output, options)  # set
                val_em += int(label == select_option)
                df.loc[idx, 'input'] = input_text
                df.loc[idx, 'output'] = output
                df.loc[idx, 'choice'] = str(select_option)
                df.loc[idx, 'correct'] = int(label == select_option)
                idx += 1
    else:
        # pre_instruction = ""  # "Please select the correct answer and if you are not completely sure of the correct answer, respond NoAns."
        if pre_instruction != "":
            pre_instruction = pre_instruction + '\n'
        if back_instruction != "":
            back_instruction = '\n' + back_instruction
        # back_instruction = "\n" + "Your Answer:"
        for i in trange(len(df)):
            prompt = df.iloc[i]['prompt']
            input_text = pre_instruction + prompt + back_instruction

            tokenizer_output = tokenizer(input_text)
            input_ids = torch.tensor(tokenizer_output['input_ids']).unsqueeze(0).cuda()
            input_mask = torch.tensor(tokenizer_output['attention_mask']).unsqueeze(0).cuda()
            generate_ids = model.generate(input_ids=input_ids, attention_mask=input_mask, max_new_tokens=200)
            generate_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)[0]

            output = generate_text[len(input_text):].strip()
            label = set(df.iloc[i]['true_label'])
            options = [x.split(") ")[-1] for x in df.iloc[i]['prompt'].split("\n")[1:]]
            select_option = get_choice_option(output, options)  # set
            val_em += int(label == select_option)
            df.loc[i, 'output'] = output
            df.loc[i, 'choice'] = str(select_option)
            df.loc[i, 'correct'] = int(label == select_option)

    df.to_csv(out_path + '/' + out_file_name)
    print(out_path + '/' + out_file_name, val_em)


if __name__ == "__main__":
    parser = ArgumentParser()
    torch.set_float32_matmul_precision("high")
    L.seed_everything(42, workers=True)

    # Basic Info
    parser.add_argument('--exp_name', default='TEST', type=str)
    parser.add_argument('--ckpt_ep', default=1, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--devices', default='[0]')
    parser.add_argument('--prefix', default='', type=str)
    parser.add_argument('--suffix', default='', type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices[1:-1]

    ckpt_path = f'/raid_sdb/home/tsy/KGLLM/{args.exp_name}/peft_ckp_ep{args.ckpt_ep}'
    out_path = f'/home/tsy/LLMFT/KGLLM/outputs/{args.exp_name}/test'
    save_file_name = f'ckpt_ep{args.ckpt_ep}_bs{args.batch_size}_prefix-{1 if args.prefix != "" else 0}_suffix_{1 if args.suffix != "" else 0}.csv'

    tokenizer_path = "/raid_sdb/LLMs/llama-7b"
    data_path = "/home/tsy/LLMFT/KGLLM/data/mydata/truthfulqa_test.csv"
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, device_map='auto')
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    path = "/home/tsy/LLMFT/KGLLM/data/obqa+csqa"
    file_name = "test.csv"
    test_set = SFTDataset(path, file_name, tokenizer, add_text=[args.prefix, args.suffix])
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    test(ckpt_path, out_path, save_file_name, args.prefix, args.suffix, test_loader=test_loader)





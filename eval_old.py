import argparse
import os
import torch
import lightning as L
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from myeval import llm_eval

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--pretrained_path", default="")
    parser.add_argument("--test_set", default="")
    parser.add_argument("--test_data_version", default="")      # not use
    parser.add_argument("--eval_data_version", default="")
    parser.add_argument('--data_path', default='/raid_sdb/home/tsy/KG_data', type=str)
    parser.add_argument('--micro_batch_size', default=2, type=int)
    parser.add_argument('--out_dir', default='/raid_sdb/home/tsy/outputs/', type=str)
    parser.add_argument('--exp_name', default='eval_baselines', type=str)


    return parser.parse_args()


def main():
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_path,
                                                 low_cpu_mem_usage=True,
                                                 torch_dtype=torch.bfloat16).to("cuda:2")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)
    args.model_config = AutoConfig.from_pretrained(args.pretrained_path)
    args.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else 0
    llm_eval(model, args=args, tokenizer=tokenizer, epoch="test")


if __name__ == "__main__":
    main()
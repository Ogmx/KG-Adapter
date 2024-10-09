import argparse
import json
import logging
import os

import torch
from transformers import LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append("..")
from myeval import llm_eval
from utils import get_edge2id
from model.llama import LlamaKgAdapterForCausalLM

#python eval_lm.py --model hf --model_args pretrained="/raid_sdb/LLMs/llama-7b",dtype="float16" --tasks truthfulqa_mc --batch_size 1 --device cuda:0

ckpt_path = "/raid_sdb/home/tsy/KGLLM/kg-adapter_lr5e-4_wu1_DS2_BiCA/peft_ckpt_epoch=1-step=458.bin"
pretrained_path = "/raid_sdb/home/tsy/models/kg-adapter-llama_base_model_llama-7b_p_num_6767430962"

if __name__ == "__main__":

    model = LlamaKgAdapterForCausalLM.from_pretrained(pretrained_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    state_dict = torch.load(ckpt_path)
    if len(state_dict.keys()) == len(model.state_dict().keys()):
        diff_parm_name = set(model.state_dict().keys()) ^ set(state_dict.keys())
        if len(diff_parm_name) > 0:
            print(diff_parm_name)
            print("These parameters not match, please check the ckpt again!")
            # return
        print("load all parameters")
    else:
        trainable_param_name = []
        for n,p in model.named_parameters():
            if "kg_adapter" in n:
                trainable_param_name.append(n)
        diff_parm_name = set(trainable_param_name) ^ set(state_dict.keys())
        if len(diff_parm_name) > 0:
            print(diff_parm_name)
            print("These parameters not match, please check the ckpt again!")
            # return
        print("only load adapter parameters")
    model.load_state_dict(state_dict, strict=False)
    model.to("cuda:3")
    result = llm_eval(model, tokenizer=tokenizer)
    print(result)

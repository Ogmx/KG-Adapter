import sys
sys.path.append("..")
import os
import shutil
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from peft import AdaptionPromptConfig, LoraConfig, IA3Config, PrefixTuningConfig, PromptTuningConfig, TaskType, get_peft_model
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from peft.utils.save_and_load import get_peft_model_state_dict

# convert a deepspeed kg-adapter checkpoint to .pt format that only keep trainable params
ckpt_path = "/raid_sdb/home/tsy/KGLLM/kg-adapter_lr1e-4_wu1_DS2"


path_lst = os.listdir(ckpt_path)
for file_name in path_lst:
    if '.ckpt' in file_name:
        print("now processing ckpt :", file_name)
        try:
            state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_path + '/' + file_name)
            tmp_dict = {}
            for k, p in state_dict.items():
                name = k.replace("_forward_module.model.", "")
                if "kg_adapter" in name:
                    tmp_dict[name] = p
            save_path = ckpt_path + "/peft_ckpt_" + str(file_name.replace(".ckpt", ".bin"))
            torch.save(tmp_dict, save_path)
            shutil.rmtree(ckpt_path + '/' + file_name)
        except:
            print("fail to convert checkpoint, maybe not have enough memery and will try again in next epoch")

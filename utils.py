# Copyright 2021 Zhongyang Zhang
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
import time
import codecs
import json
import torch
import pickle
import pandas as pd
import logging

from transformers import AutoTokenizer, BartTokenizer, GPT2Tokenizer, T5Tokenizer, BlenderbotTokenizer

model_name2path = {"BART": '/home/tsy/CRDG/pretrained_models/BART/',
                   "GPT2": '/home/tsy/CRDG/pretrained_models/GPT-2/',
                   "DialogGPT": '/home/tsy/CRDG/pretrained_models/DialogGPT-small/',
                   "T5": '/home/tsy/CRDG/pretrained_models/T5-small/',
                   "BlenderBot": '/home/tsy/CRDG/pretrained_models/BlenderBot/'}


def load_model_path(root=None, version=None, v_num=None, best=False):
    """ When best = True, return the best model's path in a directory 
        by selecting the best model with largest epoch. If not, return
        the last model saved. You must provide at least one of the 
        first three args.
    Args: 
        root: The root directory of checkpoints. It can also be a
            model ckpt file. Then the function will return it.
        version: The name of the version you are going to load.
        v_num: The version's number that you are going to load.
        best: Whether return the best model.
    """

    def sort_by_epoch(path):
        name = path.stem
        epoch = int(name.split('-')[1].split('=')[1])
        return epoch

    def generate_root():
        if root is not None:
            return root
        elif version is not None:
            return str(Path('lightning_logs', version, 'checkpoints'))
        else:
            return str(Path('lightning_logs', f'version_{v_num}', 'checkpoints'))

    if root == version == v_num == None:
        return None

    root = generate_root()
    if Path(root).is_file():
        return root
    if best:
        files = [i for i in list(Path(root).iterdir()) if i.stem.startswith('best')]
        files.sort(key=sort_by_epoch, reverse=True)
        res = str(files[0])
    else:
        res = str(Path(root) / 'last.ckpt')
    return res


def load_model_path_by_args(args):
    return load_model_path(root=args.load_dir, version=args.load_ver, v_num=args.load_v_num)


def load_tokenizer(model_name):
    if "BART" in model_name:
        tokenizer = BartTokenizer.from_pretrained(model_name2path["BART"])
    elif "GPT2" in model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name2path["GPT2"], padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
    elif "DialogGPT" in model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name2path["DialogGPT"], padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
    elif "T5" in model_name:
        tokenizer = T5Tokenizer.from_pretrained(model_name2path["T5"])
    elif "BlenderBot" in model_name:
        tokenizer = BlenderbotTokenizer.from_pretrained(model_name2path["BlenderBot"])
    else:
        tokenizer = None
    return tokenizer


def gpu_info(gpu_index):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('\n')[gpu_index].split('|')
    power = int(gpu_status[1].split()[-3][:-1])
    memory = int(gpu_status[2].split('/')[0].strip()[:-3])
    return power, memory


def waiting_gpu(interval=5, least_memory=2000):
    id = [0, 1, 2]
    flag = True
    while (flag):
        for gpu_id in id:
            gpu_power, gpu_memory = gpu_info(gpu_id)
            gpu = 'gpu id:%d' % gpu_id
            gpu_power_str = 'gpu power:%d W |' % gpu_power
            gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
            sys.stdout.write('\r' + gpu + ' ' + gpu_memory_str + ' ' + gpu_power_str)
            sys.stdout.flush()
            time.sleep(interval)
            if gpu_memory < least_memory:
                flag = False
                break
    # cmd = "CUDA_VISIBLE_DEVICES=%d python RE_model_CLS_hidden.py --data_dir ../data/dd/ --experiment_type 'all+cls' --data_set 'dd_label' --do_train --output_dir ../trained_models/dd/RE_bart_hidden_CLS/ --log_file_path ../trained_models/dd/RE_bart_hidden_CLS/log.txt --model_file_path ../trained_models/dd/RE_bart_CL_hidden_2/CL2/checkpoint-190000/all_model.pt --source_max_len 512 --target_max_len 128  --learning_rate 5e-5 --train_batch_size 8 --gradient_accumulation_steps 1 --validation_timing 10000 --num_train_epochs 50" % gpu_id
    print("\n")
    print(time.ctime(time.time()), "\n")
    print("find available GPU at ", gpu_id)
    return gpu_id


def load_node2id():
    f = codecs.open("/home/tsy/CRDG/CRDG/KE/node2id.json", "r", "utf-8")
    a = f.read()
    f.close()
    node2id = eval(str(a))
    id2node = {}
    for k, v in node2id.items():
        id2node[v] = k
    print("Test node2id, apple id is: ", node2id["/c/en/apple"])
    print("Test id2node, id ", node2id["/c/en/apple"], 'is :', id2node[node2id["/c/en/apple"]])
    del a
    return node2id, id2node


def load_kg_emb(emb_name):
    kg_emb = None
    if emb_name == "TransE":
        kg_emb = torch.load("/home/tsy/CRDG/KG/KG_emb/CSKG_TransE_emb.pt")
    elif emb_name == "TransE_2ExtDim":
        kg_emb = torch.load("/home/tsy/CRDG/KG/KG_emb/CSKG_TransE_2ExtDim_emb.pt")
    elif emb_name == "ComplEx":
        kg_emb = torch.load("/home/tsy/CRDG/KG/KG_emb/CSKG_ComplEx_emb.pt")
    elif emb_name == "DistMult":
        kg_emb = torch.load("/home/tsy/CRDG/KG/KG_emb/CSKG_DistMult_emb.pt")
    elif emb_name == "RESCAL":
        kg_emb = torch.load("/home/tsy/CRDG/KG/KG_emb/CSKG_RESCAL_emb.pt")

    return kg_emb


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


####################### New ##################################

def save_args(args):
    with open(args.out_dir + args.exp_name + '/args.json', 'wt') as f:
        json.dump(vars(args), f, indent=4)  # indent意思就是json格式缩进4个space，便于肉眼查看
        # dump()方法的第一个参数是dict，第二个参数是打开的文件句柄，第三个参数是缩进的位数
    return


def check_filename_available(filename):
    n = [0]

    def check_meta(file_name):
        file_name_new = file_name
        if os.path.isfile(file_name):
            file_name_new = file_name[:file_name.rfind('.')] + '_' + str(n[0]) + file_name[file_name.rfind('.'):]
            n[0] += 1
        if os.path.isfile(file_name_new):
            file_name_new = check_meta(file_name)
        return file_name_new

    return_name = check_meta(filename)
    return return_name


def get_peft_config(args):
    from peft import AdaptionPromptConfig, LoraConfig, IA3Config, PrefixTuningConfig, PromptTuningConfig, TaskType, \
        get_peft_model

    if args.peft_type.lower() == "llama-adapter":
        peft_config = AdaptionPromptConfig(task_type=TaskType.CAUSAL_LM,
                                           inference_mode=False,
                                           adapter_len=10,
                                           adapter_layers=30,
                                           )
    elif args.peft_type.lower() == "lora":
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                 inference_mode=False,
                                 r=8,
                                 lora_alpha=16,
                                 lora_dropout=0.05)
    elif args.peft_type.lower() == "ia3":
        peft_config = IA3Config(task_type=TaskType.CAUSAL_LM,
                                inference_mode=False)
    else:
        peft_config = None
        assert "unavailable peft-type"
    return peft_config


def load_peft_weights(args, path):
    from peft import AutoPeftModelForCausalLM
    model = AutoPeftModelForCausalLM.from_pretrained(path)
    return model


def convert_deepspeed_checkpoint_to_peft(file_path, file_name, model):
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    from peft.utils.save_and_load import get_peft_model_state_dict
    if os.path.isfile(file_path + '/' + file_name):
        state_dict = torch.load(file_path + '/' + file_name, map_location='cpu')['state_dict']
        tmp_dict = {}
        if "kgadapter" in str(type(model.model)).lower():
            save_p_lst = []
            for n, p in model.model.named_parameters():
                if p.requires_grad:
                    save_p_lst.append(n)

            for k, p in state_dict.items():
                name = k.replace("model.model.", "model.")
                if name in save_p_lst:
                    tmp_dict[name] = p
            assert len(save_p_lst) == len(tmp_dict)
            state_dict = tmp_dict

    elif os.path.isdir(file_path + '/' + file_name):
        state_dict = get_fp32_state_dict_from_zero_checkpoint(file_path + '/' + file_name)
        tmp_dict = {}
        if "kgadapter" in str(type(model.model)).lower():  # for kg-adapter model
            save_p_lst = []
            for n, p in model.model.named_parameters():
                if p.requires_grad:
                    save_p_lst.append(n)

            for k, p in state_dict.items():
                name = k.replace("_forward_module.model.", "")
                if name in save_p_lst:
                    tmp_dict[name] = p
            assert len(save_p_lst) == len(tmp_dict)
            state_dict = tmp_dict
        else:  # for huggingface PEFT model
            for k, p in state_dict.items():
                if not k.startswith("base_model."):
                    tmp_dict["base_model." + k.split("base_model.")[1]] = p
                else:
                    tmp_dict[k] = p
            state_dict = get_peft_model_state_dict(model.model, state_dict=tmp_dict)

    save_path = file_path + "/peft_ckpt_" + str(file_name.replace(".ckpt", ".bin"))
    torch.save(state_dict, save_path)
    return save_path


def get_edge2id():
    edge2id = {'/r/Antonym': 0,
               '/r/AtLocation': 1,
               '/r/CapableOf': 2,
               '/r/Causes': 3,
               '/r/CausesDesire': 4,
               '/r/CreatedBy': 5,
               '/r/DefinedAs': 6,
               '/r/DerivedFrom': 7,
               '/r/Desires': 8,
               '/r/DistinctFrom': 9,
               '/r/Entails': 10,
               '/r/EtymologicallyDerivedFrom': 11,
               '/r/EtymologicallyRelatedTo': 12,
               '/r/FormOf': 13,
               '/r/HasA': 14,
               '/r/HasContext': 15,
               '/r/HasFirstSubevent': 16,
               '/r/HasLastSubevent': 17,
               '/r/HasPrerequisite': 18,
               '/r/HasProperty': 19,
               '/r/HasSubevent': 20,
               '/r/InstanceOf': 21,
               '/r/IsA': 22,
               '/r/LocatedNear': 23,
               '/r/MadeOf': 24,
               '/r/MannerOf': 25,
               '/r/MotivatedByGoal': 26,
               '/r/NotCapableOf': 27,
               '/r/NotDesires': 28,
               '/r/NotHasProperty': 29,
               '/r/PartOf': 30,
               '/r/ReceivesAction': 31,
               '/r/RelatedTo': 32,
               '/r/SimilarTo': 33,
               '/r/SymbolOf': 34,
               '/r/Synonym': 35,
               '/r/UsedFor': 36,
               '/r/dbpedia/capital': 37,
               '/r/dbpedia/field': 38,
               '/r/dbpedia/genre': 39,
               '/r/dbpedia/genus': 40,
               '/r/dbpedia/influencedBy': 41,
               '/r/dbpedia/knownFor': 42,
               '/r/dbpedia/language': 43,
               '/r/dbpedia/leader': 44,
               '/r/dbpedia/occupation': 45,
               '/r/dbpedia/product': 46,
               'at:oEffect': 47,
               'at:oReact': 48,
               'at:oWant': 49,
               'at:xAttr': 50,
               'at:xEffect': 51,
               'at:xIntent': 52,
               'at:xNeed': 53,
               'at:xReact': 54,
               'at:xWant': 55,
               'fn:HasLexicalUnit': 56,
               'mw:MayHaveProperty': 57,
               'InSameSentence': 58,
               'InContextSentence': 59,
               'SelfLoop': 60,
               'NoEdge': 61}
    return edge2id


def eval_llm(model, args, ckpt_path=None):
    # llm_eval method from "https://github.com/EleutherAI/lm-evaluation-harness"
    # reference this file: from lm_eval.models.huggingface import _loglikelihood_tokens
    from lm_eval import tasks, evaluator, utils
    from transformers import AutoModelForCausalLM
    logging.getLogger("openai").setLevel(logging.WARNING)
    model = model.model
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
        for n, p in model.named_parameters():
            if "kg_adapter" in n:
                trainable_param_name.append(n)
        diff_parm_name = set(trainable_param_name) ^ set(state_dict.keys())
        if len(diff_parm_name) > 0:
            print(diff_parm_name)
            print("These parameters not match, please check the ckpt again!")
            # return
        print("only load adapter parameters")
    model.load_state_dict(state_dict, strict=False)
    output_path = args.out_dir + args.exp_name + "/test_results"
    results = evaluator.simple_evaluate(
        model=model,
        # model_args="dtype='float16'",
        tasks=['truthfulqa_mc'],
        num_fewshot=0,
        batch_size='auto',
        max_batch_size=8,
        device=f"cuda:{args.devices[0]}",
        write_out=True,
        output_base_path=output_path,
    )
    dumped = json.dumps(results, indent=2)
    print(dumped)

    if output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(evaluator.make_table(results))

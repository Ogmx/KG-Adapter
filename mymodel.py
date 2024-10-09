import torch
from torch import nn
import os
from typing import Optional, Any
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
import hashlib
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
import lightning as L
from transformers import get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, MistralForCausalLM, MistralConfig, \
    AutoTokenizer, AutoModelForCausalLM
from accelerate import infer_auto_device_map, init_empty_weights, init_on_device
import lightning.fabric.strategies as fbs
#from lit_llama.adapter import LLaMA, LLaMAConfig, mark_only_adapter_as_trainable, adapter_state_from_state_dict
#from lit_llama.tokenizer import Tokenizer
from peft import AdaptionPromptConfig, LoraConfig, IA3Config, PrefixTuningConfig, PromptTuningConfig, TaskType, \
    get_peft_model
from peft.peft_model import PeftModelForCausalLM
from utils import get_peft_config, get_edge2id, check_filename_available
from eval.utils import get_choice_option, get_true_or_false_option
from model.llama_v3 import LlamaKgAdapterForCausalLM
from model.mistral_v3 import MistralKgAdapterForCausalLM

DATA_TASK = {"tuqa_mc1": "mc", "tuqa_mc2": "mc2", "obqa": "mc", "csqa": "mc", "medqa": "mc", "cwq": "qa", "wqsp": "qa", "graphextqa": "qa"}


def build_kg_adapter_init_model(args, MODEL_CLASS, online_load=False, structure=None):
    print("loading kg-adapter initial model....")
    nodes_emb = torch.load(args.node_emb_path) if args.node_emb_path is not None else None
    if isinstance(nodes_emb, dict):
        nodes_emb = nodes_emb['nodes_emb']
    if 'llama' in args.pretrained_path.lower():
        model = MODEL_CLASS(config=args.model_config)  # .type(torch.float16)
        base_model_state = LlamaForCausalLM.from_pretrained(
            args.pretrained_path, low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16).state_dict()
    elif 'mistral' in args.pretrained_path.lower() or 'zephyr' in args.pretrained_path.lower():
        model = MODEL_CLASS(config=args.model_config)  # .type(torch.float16)
        base_model_state = MistralForCausalLM.from_pretrained(
            args.pretrained_path, low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16).state_dict()
    else:
        assert "only support llama or mistral model"
        model = None
        base_model_state = None

    # freeze & copy parameters
    print("initializing and freezing weights....")

    for name, param in model.named_parameters():
        if 'kg_adapter' not in name or "embed_nodes" in name:
            if name in base_model_state.keys():
                param.data.copy_(base_model_state[name].cpu())
                param.requires_grad = False
            elif "embed_nodes" in name and nodes_emb is not None:
                param.data.copy_(nodes_emb.cpu())
                param.requires_grad = False
            else:
                print("unexpect not init weight :", name)

        elif "rand_init" in args.ablation_exp_set:
            continue
        else:
            # Structural Weight Initialization
            # reference to LST: "Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning"
            map_name = name.replace("kg_adapter_", "").replace("node_layernorm", "input_layernorm").replace("t2n_",
                                                                                                            "").replace(
                "n2t_", "").replace("node_", "").replace(
                "cross", "self").replace(
                "sg", "input").replace(
                "text", "input").replace("ffn_layernorm", "post_attention_layernorm").replace("ffn", 'mlp')
            if map_name in base_model_state.keys():
                # weight magnitude as importance score of each row: "Pruning filters for efficient convnets"
                tmp = base_model_state[map_name].cpu()
                if len(param.size()) == 1:
                    select_row_ids = tmp.topk(param.size(0))[1].sort()[0]
                    tmp = tmp.index_select(0, select_row_ids)
                else:
                    select_row_ids = tmp.norm(p=1, dim=1).topk(param.size(0))[1].sort()[0]
                    tmp = tmp.index_select(0, select_row_ids)
                    select_col_ids = tmp.norm(p=1, dim=0).topk(param.size(1))[1].sort()[0]
                    tmp = tmp.index_select(1, select_col_ids)
                param.data.copy_(tmp)
            else:
                print("not init weight :", name, map_name)

        # process inf and nan value for bf16/pf16
        # clamp_value = torch.where(
        #     torch.isinf(param.data).any(),
        #     torch.finfo(torch.float16).max - 1000,
        #     torch.finfo(torch.float16).max, )
        # torch.clamp_(param.data, min=-clamp_value, max=clamp_value)
        # param.data = torch.where(torch.isnan(param.data), torch.zeros_like(param.data), param.data)

    del base_model_state
    del nodes_emb

    if not online_load:
        all_p_num = sum([param.nelement() for param in model.parameters()])
        model.half()
        if structure is None:
            path = f"{args.kg_adapter_model_path}_base_model_{args.pretrained_path.split('/')[-1]}_p_num_{all_p_num}"
            model.save_pretrained(path, max_shard_size="1GB")
            print(
                f"saving model to {args.kg_adapter_model_path}_base_model_{args.pretrained_path.split('/')[-1]}_p_num_{all_p_num}")
        else:
            if "rand_init" in args.ablation_exp_set:
                structure = structure + "_rand_init"
            path = f"{args.kg_adapter_model_path}_base_model_{args.pretrained_path.split('/')[-1]}_p_num_{all_p_num}_s_{structure}"
            model.save_pretrained(path, max_shard_size="1GB")
            print(
                f"{args.kg_adapter_model_path}_base_model_{args.pretrained_path.split('/')[-1]}_p_num_{all_p_num}_s_{structure}")

        return path


class KgAdapterModule(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.f_log = open(check_filename_available(self.args.out_dir + self.args.exp_name + "/log.txt"), 'w')
        # self.fabric = self.init_fabric()
        # self.model, self.tokenizer = self.load_model()
        if 'llama' in args.pretrained_path.lower():
            self.MODEL_CLASS = LlamaKgAdapterForCausalLM
        elif 'mistral' in args.pretrained_path.lower() or 'zephyr' in args.pretrained_path.lower():
            self.MODEL_CLASS = MistralKgAdapterForCausalLM
        # if args.dev:
        #     from model.mistral_v2 import MistralKgAdapterForCausalLM_Dev
        #     self.MODEL_CLASS = MistralKgAdapterForCausalLM_Dev
        # if args.dev2 and ('mistral' in args.pretrained_path.lower() or 'zephyr' in args.pretrained_path.lower()):
        #     from model.mistral_v3 import MistralKgAdapterForCausalLM
        #     self.MODEL_CLASS = MistralKgAdapterForCausalLM_Dev

        if self.args.peft_type.lower() == "kg-adapter":
            self.model, self.tokenizer = self.load_kg_adapter_model()
        elif "peft" in self.args.peft_type.lower():
            self.model, self.tokenizer = self.load_peft_model()
        else:   # peft_type == "base"
            self.model, self.tokenizer = self.load_hf_model()

        self.df = pd.DataFrame()

        csv_test_data_path = args.test_data_path

        def load_test_data(name):
            tmp = pd.read_csv(csv_test_data_path, index_col=0)
            tmp = tmp[tmp['typ'] == name].reset_index(drop=True)
            self.df = pd.concat([self.df, tmp]).reset_index(drop=True)

        for data_name in DATA_TASK:
            if data_name in args.test_set:
                load_test_data(data_name)

        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.test_step_outputs = []

        self.output_sg_state = True if "output_sg" in args.test_set else False
        if self.args.peft_type.lower() == "kg-adapter":
            self.save_hyperparameters(self.args)

    def init_fabric(self):
        args = self.args
        fabric = L.Fabric(
            accelerator=args.accelerator,
            strategy=fbs.DeepSpeedStrategy(config=args.ds_config) if len(args.devices) > 1 else "auto",
            precision=args.precision,
            devices=args.devices,
        )
        fabric.launch()
        return fabric

    # def load_model(self):
    #     args = self.args
    #     fabric = L.Fabric(
    #         accelerator=args.accelerator,
    #         strategy=fbs.DeepSpeedStrategy(config=args.ds_config) if len(args.devices) > 1 else "auto",
    #         precision=args.precision,
    #         devices=args.devices,
    #     )
    #     fabric.launch()
    #     print("Loading llama....")
    #     config = LLaMAConfig(block_size=args.max_seq_length)
    #     if not os.path.isfile(args.pretrained_path):
    #         raise FileNotFoundError(
    #             f"Can't find the pretrained weights at {args.pretrained_path}."
    #             " Please follow the instructions in the README to download them."
    #         )
    #     checkpoint = torch.load(args.pretrained_path)
    #
    #     with fabric.init_module():
    #         model = LLaMA(config)
    #         # strict=False because missing keys due to adapter weights not containted in state dict
    #         model.load_state_dict(checkpoint, strict=False)
    #
    #     del fabric
    #     mark_only_adapter_as_trainable(model)
    #
    #     num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    #     print(f"Number of trainable parameters: {num_params}")
    #
    #     tokenizer = Tokenizer(self.args.tokenizer_path)
    #
    #     return model, tokenizer

    def load_hf_model(self):
        args = self.args
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_path,
                                                     low_cpu_mem_usage=True,
                                                     torch_dtype=torch.bfloat16,
                                                     config=args.model_config
                                                     )
        self.args.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        return model, tokenizer

    def load_peft_model(self):
        args = self.args
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_path,
                                                     low_cpu_mem_usage=True,
                                                     torch_dtype=torch.bfloat16,
                                                     config=args.model_config
                                                     )

        if "lora" in self.args.peft_type.lower():
            r = 64
            if "64" in args.peft_type.lower():
                r = 64
            elif "32" in args.peft_type.lower():
                r = 32
            a = r * 4
            peft_config = LoraConfig(
                r=r,
                target_modules=["q_proj", "v_proj"],
                lora_alpha=a,
            )
        # peft_config = get_peft_config(args)
        model = get_peft_model(model, peft_config)
        args.model_config = model.config
        self.args.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        model.print_trainable_parameters()

        return model, tokenizer

    def load_kg_adapter_model(self):
        args = self.args
        if args.node_emb_path is not None:
            nodes_emb = torch.load(args.node_emb_path)
            if isinstance(nodes_emb, dict):
                nodes_emb = nodes_emb['nodes_emb']
            self.args.model_config.node_num = nodes_emb.size(0)
            self.args.model_config.kg_adapter_node_emb_size = nodes_emb.size(-1)
            print("kg nodes num: ", nodes_emb.size(0))
            del nodes_emb
        else:
            print("not use pretrained kg embedding")
            assert not args.model_config.use_node_emb
        if args.num_relations == 1:
            print("not use edge type")

        with init_empty_weights():
            model = self.MODEL_CLASS(config=self.args.model_config)

        param_names = [k for k in model.state_dict().keys()]
        structure = hashlib.md5(str(param_names).encode('utf-8')).hexdigest()
        if "rand_init" in args.ablation_exp_set:
            structure = structure + "_rand_init"
        all_p_num = sum([param.nelement() for param in model.parameters()])
        args.model_all_p_num = all_p_num
        if args.debug:
            model = self.MODEL_CLASS(config=args.model_config).to(torch.bfloat16)
        else:
            if (not os.path.exists(
                    f"{args.kg_adapter_model_path}_base_model_{args.pretrained_path.split('/')[-1]}_p_num_{all_p_num}")
                and not os.path.exists(
                        f"{args.kg_adapter_model_path}_base_model_{args.pretrained_path.split('/')[-1]}_p_num_{all_p_num}_s_{structure}")) \
                    or args.kg_adapter_online_load:
                print("not use preprocessed kg-adapter model, initializing now ...")
                model_path = build_kg_adapter_init_model(self.args, self.MODEL_CLASS,
                                                         online_load=args.kg_adapter_online_load,
                                                         structure=structure)
                model = self.MODEL_CLASS.from_pretrained(
                    model_path,
                    config=self.args.model_config,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                )

            else:
                if os.path.exists(
                        f"{args.kg_adapter_model_path}_base_model_{args.pretrained_path.split('/')[-1]}_p_num_{all_p_num}_s_{structure}"):
                    model_path = f"{args.kg_adapter_model_path}_base_model_{args.pretrained_path.split('/')[-1]}_p_num_{all_p_num}_s_{structure}"
                elif os.path.exists(
                        f"{args.kg_adapter_model_path}_base_model_{args.pretrained_path.split('/')[-1]}_p_num_{all_p_num}"):
                    model_path = f"{args.kg_adapter_model_path}_base_model_{args.pretrained_path.split('/')[-1]}_p_num_{all_p_num}"
                else:
                    model_path = None
                    assert "not find available model path"
                print("using preprocessed model from :", model_path)
                model = self.MODEL_CLASS.from_pretrained(
                    model_path,
                    config=self.args.model_config,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                )

                # loaded_param_names = [k for k in model.state_dict().keys()]
                # if param_names != loaded_param_names:
                #     assert "loaded params not match!"

        # freezing weights
        print("freezing weights....")
        for name, param in model.named_parameters():
            if 'kg_adapter' not in name and "embed_edges" not in name:
                param.requires_grad = False
            if 'lora' in name:
                param.requires_grad = True
            if "init_kg_emb" in args.exp_set and 'embed_nodes' in name:  # not use pretrained kg emb
                from torch.nn.init import kaiming_normal_
                param.data = kaiming_normal_(param)
            # if "train_head" in args.ablation_exp_set and "lm_head" in name:
            #     param.requires_grad = True

        # load and config tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = 'left'

        self.args.pad_id = tokenizer.pad_token_id
        return model, tokenizer

    def configure_optimizers(self):
        args = self.args
        if "deepspeed" in args.strategy and "offload" in args.strategy:
            optimizer = DeepSpeedCPUAdam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                          weight_decay=args.weight_decay)
        self.args.one_epoch_update_steps = int(
            len(self.trainer.datamodule.train_set) // args.micro_batch_size // args.gradient_accumulation_iters)
        self.args.total_update_steps = int(args.one_epoch_update_steps * args.max_epochs)
        self.args.warmup_steps = int(args.warm_up_epoch * self.args.one_epoch_update_steps)
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.args.total_update_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    # def loss_fn(self, logits, targets):
    #     # shift the targets such that output n predicts token n+1
    #     logits = logits[..., :-1, :].contiguous()
    #     targets = targets[..., 1:].contiguous()
    #     loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
    #     return loss

    def forward(self, x: torch.Tensor, y=None, mask=None, sg=None):
        if "kg-adapter" in self.args.peft_type:
            return self.model(input_ids=x, labels=y, attention_mask=mask, sg=sg)
        elif "peft" in self.args.peft_type:
            return self.model(input_ids=x, labels=y, attention_mask=mask)
        else:
            return self.model(x)

    def training_step(self, batch, batch_idx: int):
        if "kg-adapter" in self.args.peft_type:
            idx, x, y, mask, prompt_len, x_no_res, x_no_res_mask, sg = batch
        else:
            idx, x, y, mask, prompt_len, x_no_res, x_no_res_mask = batch
            sg = None
        if "kg-adapter" in self.args.peft_type:
            logits = self(x, y=y, mask=mask, sg=sg)
            loss = logits['loss']
        elif "peft" in self.args.peft_type:
            logits = self(x, y=y, mask=mask)
            loss = logits['loss']
        else:
            logits = self(x)
            loss = self.loss_fn(logits, y)

        self.train_step_outputs.append(loss.item())
        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        train_loss = outputs['loss']
        self.log('train_loss', train_loss, prog_bar=True, logger=True)

        # gpu_cache = torch.cuda.memory_reserved()
        # if gpu_cache / 1e9 + 2 > 32: #or batch_idx % 100 == 0:
        #     torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx: int):
        res = {"idx": torch.zeros(1), "val_loss": 0.0, "generate_text": ""}
        if "kg-adapter" in self.args.peft_type:
            idx, x, y, mask, prompt_len, x_no_res, x_no_res_mask, sg = batch
        else:
            idx, x, y, mask, prompt_len, x_no_res, x_no_res_mask = batch
            sg = None
        if idx[-1].item() not in self.df.index:
            self.validation_step_outputs.append(res)
            return
        if isinstance(self.model, PeftModelForCausalLM) or "kg-adapter" in self.args.peft_type:
            with torch.no_grad():
                output = self.model.generate(input_ids=x_no_res, attention_mask=x_no_res_mask, sg=sg,
                                             max_new_tokens=100, pad_token_id=self.tokenizer.pad_token_id,
                                             output_attentions=self.output_sg_state, return_dict_in_generate=True)
                logits = self(x, y=y, mask=mask, sg=sg)

            generate_ids = output[0]
            if self.output_sg_state:
                sg_states = output[1]
                res['sg_states'] = sg_states
            generate_text = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False)

            loss = logits['loss']
            res['idx'], res['val_loss'], res['generate_text'] = idx, loss, generate_text
        elif "base" in self.args.peft_type or "peft" in self.args.peft_type:
            generate_ids = self.model.generate(input_ids=x_no_res, attention_mask=x_no_res_mask,
                                               max_new_tokens=100, pad_token_id=self.tokenizer.pad_token_id)
            generate_text = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False)
            res['idx'], res['val_loss'], res['generate_text'] = idx, 0.0, generate_text
        else:
            input = x.squeeze()[:prompt_len]
            generate_ids = self.generate(input, 200, temperature=1, top_k=None, eos_id=self.tokenizer.eos_id)
            generate_text = self.tokenizer.decode(generate_ids)
            self.model.reset_cache()
            logits = self(x)
            loss = self.loss_fn(logits, y)
            res['idx'], res['val_loss'], res['generate_text'] = idx, loss, generate_text

        self.validation_step_outputs.append(res)
        # return {"idx": idx, "val_loss": loss, "val_em": val_em}

    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        ...
        # gpu_cache = torch.cuda.memory_reserved()
        # if gpu_cache / 1e9 + 2 > 32:    # or batch_idx % 100 == 0:
        #     torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        print('validation_epoch_end')
        all_val_out = self.all_gather(self.validation_step_outputs)
        all_train_out = self.all_gather(self.train_step_outputs)
        val_loss = torch.mean(torch.tensor([torch.mean(x['val_loss']) for x in all_val_out])).item()
        train_loss = np.mean([x.item() for x in all_train_out])
        val_em = 0.0
        # val_em = torch.sum(torch.tensor([torch.sum(x['val_em']) for x in all_val_out])).item()
        for batch_out in all_val_out:
            ids = batch_out['idx'].tolist()
            gen_texts = batch_out['generate_text']
            for i, text in zip(ids, gen_texts):
                if "### Assistant:" in text:
                    output = text.split("### Assistant:")[1].strip()
                elif "### Response:" in text:
                    output = text.split("### Response:")[1].strip()
                elif "[/INST]" in text:
                    output = text.split("[/INST]")[1].strip()
                elif "<|assistant|>" in text:
                    output = text.split("<|assistant|>")[1].strip()
                elif "#Your Judgement#:" in text:
                    output = text.split("#Your Judgement#:")[-1].strip()
                elif '\nA:' in text:
                    output = text.split("\nA:")[-1].split("\n")[0].strip()
                else:
                    output = text.split('\n')[-1]

                labels = eval(self.df.iloc[i]['label'])
                if isinstance(labels, tuple):
                    if isinstance(labels[-1], list):
                        labels = set(labels[-1])
                    else:
                        labels = {labels[-1]}
                else:
                    labels = set(eval(self.df.iloc[i]['label']))

                task_typ = DATA_TASK[self.df.iloc[i]['typ']]

                if task_typ == "mc2":  # for multiple choice
                    options = eval(self.df.iloc[i]['choices'])
                    select_option = get_choice_option(output, options)
                    correct = len(labels & select_option)
                elif task_typ == "mc":  # for single choice
                    options = eval(self.df.iloc[i]['choices'])
                    select_option = get_choice_option(output, options)
                    correct = int(labels == select_option)
                elif task_typ == "qa":
                    from eval.utils import cal_kgqa_metrics
                    labels = eval(self.df.iloc[i]['label'])
                    f1, h1, em = cal_kgqa_metrics(output, labels)
                    correct = h1
                    select_option = (f1, h1, em)  # use this column to save all metrics
                elif task_typ == "tf":  # for halu ture or false
                    correct, select_option = get_true_or_false_option(output, labels)
                else:
                    correct = 0
                    select_option = "None"
                    assert "not available test task type"

                val_em += correct
                self.df.loc[i, 'output'] = output
                self.df.loc[i, 'raw_output'] = text
                self.df.loc[i, 'choice'] = str(select_option)
                self.df.loc[i, 'correct'] = correct

        save_file_name = self.args.out_dir + self.args.exp_name + "/results/" + "test_result_ep" + str(
            self.trainer.current_epoch) + "_rank_" + str(self.global_rank) + ".csv"

        if self.trainer.state.stage[:] != "sanity_check":
            self.df.to_csv(save_file_name)

            if self.output_sg_state:
                tmp = []
                for batch_out in all_val_out:
                    ids = batch_out['idx'].tolist()
                    tmp.append([ids, batch_out['sg_states']])
                torch.save(tmp, save_file_name.replace(".csv", ".bin"))

        self.log('avg_val_loss', val_loss, logger=True, sync_dist=True)
        self.log('val_em', val_em, logger=True, sync_dist=True)

        # calculate generation result scores
        def cal_mc_data_score(name, eval_dict):
            if self.args.eval_data_version is not None:
                tmp_dev = self.df[(self.df['typ'] == name) & (self.df['split'] == 'dev')]
                if name == "csqa":
                    tmp_test = self.df[(self.df['typ'] == name) & (self.df['split'] == 'ih_test')]
                else:
                    tmp_test = self.df[(self.df['typ'] == name) & (self.df['split'] == 'test')]
                eval_dict[f'{name}_dev_acc'] = sum(tmp_dev['correct']) / len(tmp_dev)
                eval_dict[f'{name}_test_acc'] = sum(tmp_test['correct']) / len(tmp_test)
            else:
                tmp = self.df[self.df['typ'] == name]
                eval_dict[f'{name}_acc'] = sum(tmp['correct']) / len(tmp)

        def cal_kgqa_data_score(name, eval_dict):
            tmp = self.df[self.df['typ'] == name]
            eval_dict[f"{name}_acc"] = 0
            eval_dict[f"{name}_F1"] = 0
            eval_dict[f"{name}_Hits@1"] = 0
            cnt = 0
            for i in range(len(tmp)):
                if len(eval(tmp.iloc[i]['label'])) == 0 or str(tmp.iloc[i]['choice']) == "nan":
                    continue
                f1, h1, em = eval(tmp.iloc[i]['choice'])
                eval_dict[f"{name}_acc"] += em
                eval_dict[f"{name}_F1"] += f1
                eval_dict[f"{name}_Hits@1"] += h1
                cnt += 1
            for key in eval_dict:
                if name in key:
                    eval_dict[key] = eval_dict[key] / cnt if cnt else 0

        cal_scores_map = {"tuqa_mc1": cal_mc_data_score, "tuqa_mc2": cal_mc_data_score, "halu": cal_mc_data_score,
                          "obqa": cal_mc_data_score,
                          "csqa": cal_mc_data_score, "medqa": cal_mc_data_score, "wqsp": cal_kgqa_data_score,
                          "cwq": cal_kgqa_data_score, "graphextqa": cal_kgqa_data_score}

        if self.trainer.is_global_zero:
            eval_dict = {}
            for data_name, cal_score in cal_scores_map.items():
                if data_name in self.args.test_set:
                    cal_score(data_name, eval_dict)

            print("using lm-evaluation-harness test...")
            from myeval import llm_eval

            # calculate harness-llm-eval result scores
            llm_eval_res = llm_eval(self.model, self.args, tokenizer=self.tokenizer,
                                    epoch=str(self.trainer.current_epoch))

            if llm_eval_res is not None:
                avg_acc = []
                for k in llm_eval_res['results']:
                    for metric in llm_eval_res['results'][k]:
                        value = llm_eval_res['results'][k][metric]
                        if metric in ['acc', 'mc1', 'mc2']:
                            avg_acc.append(value)
                avg_acc = np.mean(avg_acc)
            else:
                avg_acc = 0.0

            self.log('val_avg_acc', avg_acc, logger=True)
            result = {"avg_train_loss": train_loss, "avg_val_loss": val_loss, "avg_val_acc": avg_acc, "val_em": val_em}
            result.update(eval_dict)
            print(str(result))

            if self.f_log is not None:
                self.f_log.write("----valid at epoch " + str(self.trainer.current_epoch) + " at global rank " + str(
                    self.global_rank) + ": ")
                self.f_log.write(str(result))
                self.f_log.write('\n')
                self.f_log.write(str(llm_eval_res))
                self.f_log.write('\n')
                self.f_log.write('\n')
                self.f_log.flush()

            if self.trainer.state.stage[:] != "sanity_check":
                try:
                    df1 = pd.read_csv(self.args.out_dir + self.args.exp_name + "/results/" + "test_result_ep" + str(
                        self.trainer.current_epoch) + "_rank_0.csv", index_col=0)
                    df2 = pd.read_csv(self.args.out_dir + self.args.exp_name + "/results/" + "test_result_ep" + str(
                        self.trainer.current_epoch) + "_rank_1.csv", index_col=0)

                    df = pd.concat([df1[df1.index % 2 == 0], df2[df2.index % 2 == 1]]).sort_index()
                    df.to_csv(self.args.out_dir + self.args.exp_name + "/results/" + "test_result_ep" + str(
                        self.trainer.current_epoch) + ".csv")
                except:
                    print("fail to build concat df file")

        self.validation_step_outputs.clear()  # free memory
        self.train_step_outputs.clear()
        self.trainer.strategy.barrier()

    def test_step(self, batch, batch_idx):
        input_ids, input_mask, input_text_len = batch
        generate_ids = self.model.generate(input_ids=input_ids, attention_mask=input_mask, max_new_tokens=200)
        generate_text = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)
        output = [text[len(input_text_len):].strip() for text in generate_text]
        self.test_step_outputs.append({"output": output})

    def on_test_epoch_end(self):
        df = pd.read_csv(self.args.test_data_path, index_col=0)

        test_em = 0.0
        idx = 0

        for outputs in self.test_step_outputs:
            for output in outputs['output']:
                label = set(df.iloc[idx]['true_label'])
                options = [x.split(") ")[-1] for x in df.iloc[idx]['prompt'].split("\n")[1:]]
                select_option = get_choice_option(output, options)  # set
                test_em += int(label == select_option)
                df.loc[idx, 'output'] = output
                df.loc[idx, 'choice'] = str(select_option)
                df.loc[idx, 'correct'] = int(label == select_option)
                idx += 1
        # val_em = torch.tensor(val_em, dtype=torch.float32)
        self.log('test_em', test_em, sync_dist=True, logger=True)

        result = {"test_em": test_em}
        print(str(result))

        if self.f_log is not None:
            self.f_log.write("----test at epoch " + str(self.trainer.current_epoch) + ": ")
            self.f_log.write(str(result))
            self.f_log.write('\n')
            self.f_log.flush()

        if self.trainer.state.stage[:] != "sanity_check":
            df.to_csv(self.args.out_dir + self.args.exp_name + "/results/" + "test_result_ep" + str(
                self.trainer.current_epoch) + ".csv")

        self.test_step_outputs.clear()  # free memory

    def on_save_checkpoint(self, checkpoint):
        if "kg-adapter" in self.args.peft_type:
            return
        if isinstance(self.model, PeftModelForCausalLM):
            # checkpoint['state_dict'] = get_peft_model_state_dict(self.model)
            self.model.save_pretrained(
                save_directory=self.args.save_path + "/peft_ckp_ep" + str(self.trainer.current_epoch))
        # else:
        #     checkpoint['state_dict'] = adapter_state_from_state_dict(checkpoint['state_dict'])

        print("only save adapter parameter")
        return

    # def save(self):
    #     file_path = Path(self.args.save_path)
    #     if isinstance(fabric.strategy, DeepSpeedStrategy):
    #         from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    #
    #         tmp_path = file_path.with_suffix(".tmp")
    #         fabric.save(tmp_path, {"model": model})
    #         fabric.barrier()
    #         if fabric.global_rank == 0:
    #             # Create a consolidated checkpoint with the same name next to the deepspeed checkpoint
    #             # and only keep the adapter weights
    #             state_dict = get_fp32_state_dict_from_zero_checkpoint(tmp_path)
    #             state_dict = adapter_state_from_state_dict(state_dict)
    #             torch.save(state_dict, file_path)
    #             shutil.rmtree(tmp_path)
    #     else:
    #         state_dict = adapter_state_from_state_dict(model.state_dict())
    #         if fabric.global_rank == 0:
    #             torch.save(state_dict, file_path)
    #         fabric.barrier()
    #
    # def generate(
    #         self,
    #         idx: torch.Tensor,
    #         max_new_tokens: int,
    #         *,
    #         max_seq_length: Optional[int] = None,
    #         temperature: float = 1.0,
    #         top_k: Optional[int] = None,
    #         eos_id: Optional[int] = None,
    # ) -> torch.Tensor:
    #     """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    #
    #     The implementation of this function is modified from A. Karpathy's nanoGPT.
    #
    #     Args:
    #         model: The model to use.
    #         idx: Tensor of shape (T) with indices of the prompt sequence.
    #         max_new_tokens: The number of new tokens to generate.
    #         max_seq_length: The maximum sequence length allowed.
    #         temperature: Scales the predicted logits by 1 / temperature
    #         top_k: If specified, only sample among the tokens with the k highest probabilities
    #         eos_id: If specified, stop generating any more token once the <eos> token is triggered
    #     """
    #     # create an empty tensor of the expected final shape and fill in the current tokens
    #     model = self.model
    #     T = idx.size(0)
    #     T_new = T + max_new_tokens
    #     if max_seq_length is None:
    #         max_seq_length = min(T_new, model.config.block_size)
    #
    #     max_new_tokens = max_seq_length - T
    #     T_new = T + max_new_tokens
    #
    #     device, dtype = idx.device, idx.dtype
    #     # create an empty tensor of the expected final shape and fill in the current tokens
    #     empty = torch.empty(T_new, dtype=dtype, device=device)
    #     empty[:T] = idx
    #     idx = empty
    #     input_pos = torch.arange(0, T, device=device)
    #
    #     # generate max_new_tokens tokens
    #     for _ in range(max_new_tokens):
    #         x = idx.index_select(0, input_pos).view(1, -1)
    #
    #         # forward
    #         logits = model(x, max_seq_length, input_pos)
    #         logits = logits[0, -1] / temperature
    #
    #         # optionally crop the logits to only the top k options
    #         if top_k is not None:
    #             v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    #             logits = torch.where(logits < v[[-1]], -float("Inf"), logits)
    #
    #         probs = torch.nn.functional.softmax(logits, dim=-1)
    #         idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)
    #
    #         # advance
    #         input_pos = input_pos[-1:] + 1
    #
    #         # concatenate the new generation
    #         idx = idx.index_copy(0, input_pos, idx_next)
    #
    #         # if <eos> token is triggered, return the output (stop generation)
    #         if idx_next == eos_id:
    #             return idx[:input_pos]  # include the EOS token
    #
    #     return idx

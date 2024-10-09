import torch
import os
import shutil
import traceback
from argparse import ArgumentParser
import lightning as L
from lightning import Trainer
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from transformers import AutoConfig
from mydata import MyDataModule
from mymodel import KgAdapterModule
from utils import save_args, load_peft_weights, convert_deepspeed_checkpoint_to_peft, get_edge2id, eval_llm


def convert_deepspeed_ckpts(args, model):
    if "deepspeed" in args.strategy:
        ckpt_path = args.save_path
        path_lst = os.listdir(ckpt_path)
        for file_name in path_lst:
            if '.ckpt' in file_name:
                print("now processing ckpt :", file_name)
                try:
                    file_path = convert_deepspeed_checkpoint_to_peft(ckpt_path, file_name, model)
                    if os.path.isfile(ckpt_path + '/' + file_name):
                        os.remove(ckpt_path + '/' + file_name)
                    else:
                        shutil.rmtree(ckpt_path + '/' + file_name)
                    if "peft" in args.peft_type:
                        move_to_path = ckpt_path + '/peft_ckp_ep' + file_name.split("epoch=")[1][0] + '/'
                        shutil.copy(file_path, move_to_path + "adapter_model.bin")
                except:
                    print("fail to convert checkpoint, maybe not have enough memery and will try again in next epoch")


def load_callbacks(args):
    callbacks = []
    callbacks.append(ModelCheckpoint(
        dirpath=args.save_path,
        save_weights_only=True,
        save_last=False,
        verbose=True,
        monitor=args.monitor,
        mode='max',
        save_top_k=args.save_top_k,   # 2
        # every_n_epochs=1
    ))

    callbacks.append(EarlyStopping(
        monitor=args.monitor,
        mode='max',
        min_delta=0.00,
        patience=args.patience,
        verbose=False
    ))

    callbacks.append(LearningRateMonitor(
        logging_interval='step'))

    return callbacks


def main(args):
    torch.set_float32_matmul_precision("high")
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    L.seed_everything(42, workers=True)
    if args.debug:
        args.num_workers = 0
        args.devices = eval(args.devices)
        print("running in debug mode.....")
    # elif args.eval:
    #     args.devices = eval(args.devices)
    #     print("running in eval mode.....")
    else:
        # args.num_workers = 0
        # args.devices = eval(args.devices)
        args.devices = [x for x in range(len(eval(args.devices)))]

    os.makedirs(args.out_dir + args.exp_name, exist_ok=True)
    os.makedirs(args.out_dir + args.exp_name + "/results", exist_ok=True)
    args.save_path = args.save_path + args.exp_name

    # MPS backend currently does not support all operations used in this example.
    # If you want to use MPS, set accelerator='auto' and also set PYTORCH_ENABLE_MPS_FALLBACK=1
    if args.accelerator is None:
        args.accelerator = "cpu" if torch.backends.mps.is_available() else "auto"

    args.batch_size_per_device = args.batch_size // len(args.devices)
    args.gradient_accumulation_iters = args.batch_size_per_device // args.micro_batch_size

    logger = TensorBoardLogger(save_dir=args.out_dir + args.exp_name, name="tb_logs")
    # set deepspeed config
    if "deepspeed" in args.strategy and len(args.devices) > 1:
        ds_config = {
            "stage": 2,
            "offload_optimizer": False,
            "offload_parameters": False,
        }
        if "3" in args.strategy:
            ds_config["stage"] = 3
        if "offload" in args.strategy:
            ds_config["offload_optimizer"] = True
            ds_config["offload_parameters"] = True
        args.ds_config = ds_config
        strategy = DeepSpeedStrategy(stage=ds_config['stage'],
                                     offload_optimizer=ds_config['offload_optimizer'],
                                     offload_parameters=ds_config['offload_parameters'])
    #
    # elif "deepspeed" in args.strategy and len(args.devices) == 1:
    #     print("deepspeed strategy must run with more than one gpu, change the strategy to auto")
    #     strategy = 'auto'
    else:
        strategy = 'auto'

    callbacks = load_callbacks(args)
    trainer = Trainer(
        fast_dev_run=2 if args.debug else False,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=strategy,
        precision=args.precision,
        max_epochs=args.num_epochs,
        log_every_n_steps=50,
        num_sanity_val_steps=2,
        accumulate_grad_batches=args.gradient_accumulation_iters,
        gradient_clip_val=1,
        logger=logger,
        callbacks=callbacks,
        # deterministic=True,
        # detect_anomaly = False,
    )

    # set kg-adapter config
    args.max_node_num_per_batch = 2500
    if args.peft_type == "kg-adapter":
        # set kg-adapter hyperparameter
        init_config = AutoConfig.from_pretrained(args.pretrained_path)

        if args.debug:
            init_config.num_hidden_layers = 5
        init_config.kg_adapter_enc_range = [0, 0] if args.debug else eval(args.kg_adapter_enc_range)  # [2, 16],
        init_config.kg_adapter_dec_range = [0, 5] if args.debug else eval(args.kg_adapter_dec_range)  # [16, 32],
        init_config.kg_adapter_node_emb_size = args.kg_adapter_node_emb_size  # 100
        init_config.kg_adapter_hidden_size = args.kd_adapter_hidden_size  # 64
        # node_num=65714,  # CSQA+OBQA+TruthfulQA nodes
        init_config.kg_adapter_intermediate_size = args.kd_adapter_hidden_size * 4
        init_config.kg_adapter_info_merge = args.kg_adapter_info_merge  # choose from [gate, linear, sum]
        init_config.share_ca = False if "no_share_ca" in args.exp_set else True
        init_config.dynamic_prune = True if "dynamic_prune" in args.exp_set else False
        init_config.align_mask = True if "align_mask" in args.exp_set else False
        init_config.use_gnn = False if "no_gnn" in args.ablation_exp_set else True
        init_config.enc_interact_with_LLM = True if "no_dec" in args.exp_set else False
        init_config.use_node_emb = False if "no_node_emb" in args.exp_set else True
        init_config.use_edge_emb = True if "use_edge_emb" in args.exp_set else False
        init_config.mix_emb = True if "mix_emb" in args.exp_set else False
        init_config.use_trips = True if ("use_trips" in args.exp_set or "use_cat_trips" in args.exp_set) else False
        init_config.use_SRGAT = True if "use_SRGAT" in args.exp_set else False
        init_config.enc_sa = True if "enc_sa" in args.exp_set else False
        init_config.num_relations = args.num_relations  # if 'mixdata' in args.train_data_version else 62      #11 for merged_rel 62 for not merged_rel
        init_config.output_sg = True if "output_sg" in args.test_set else False

        init_config.keep_ratio = args.keep_ratio

        init_config.exp_set = args.exp_set
        del init_config.torch_dtype     # has bug with lightning.logger -> save_hyperparameters

        # experimental features config
        init_config.dev = True if args.dev else False
        init_config.fuse_rate = args.fuse_rate
        init_config.scaling_rate = args.scaling_rate
        init_config.add_lora = True if "add_lora" in args.ablation_exp_set else False
        init_config.train_lm_head = True if "train_lm_head" in args.ablation_exp_set else False
        init_config.use_prefix = True if "use_prefix" in args.ablation_exp_set else False
        init_config.use_kg_encoder = True if "use_kg_encoder" in args.ablation_exp_set else False

        init_config.no_res = True if "no_res" in args.ablation_exp_set else False
        init_config.linear_scale = True if "linear_scale" in args.ablation_exp_set else False
        init_config.linear_emb = True if "linear_emb" in args.ablation_exp_set else False
        init_config.info_merge_pos = args.info_merge_pos

        args.model_config = init_config
    else:
        args.model_config = AutoConfig.from_pretrained(args.pretrained_path)
        if args.debug:
            args.model_config.num_hidden_layers = 1

    # Loading Model
    model = KgAdapterModule(args)

    # Loading Data
    data_module = MyDataModule(args, tokenizer=model.tokenizer)

    if args.eval:
        if args.peft_type == "kg-adapter" and args.ckpt_path is not None:
            print("loading check point form:", args.ckpt_path)
            ckpt_state_dict = torch.load(args.ckpt_path)
            model.model.load_state_dict(ckpt_state_dict, strict=False)
        trainer.validate(model, data_module)
    else:
        trainer.fit(model, data_module)
    # convert deepspeed checkpoints to PEFT checkpoints that only keep trainable weights
    if args.save_top_k > 0:
        convert_deepspeed_ckpts(args, model)

    # from myeval import llm_eval
    # llm_eval(model, args)
    # TODO: test with different prompt?
    # path = "/raid_sdb/home/tsy/KGLLM/peft_llama-adapter_lr9e-3_wu2_DS2_pad-left/peft_ckp_ep0"
    # test_model = AutoPeftModelForCausalLM.from_pretrained(path) #callbacks[0].best_model_path)
    # TODO: test with lm-eval and build our own task class
    # best_ckpt_path = getattr(trainer.checkpoint_callback, "best_model_path", None)
    # print(best_ckpt_path)
    # eval_llm(model, args, "/raid_sdb/home/tsy/KGLLM/kg-adapter_lr1e-4_wu1_DS2/peft_ckpt_epoch=2-step=687.bin")
    # trainer.test(model, dataloaders=data_module, ckpt_path="best")


if __name__ == "__main__":
    parser = ArgumentParser()
    torch.set_float32_matmul_precision("high")

    # Basic Setting
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--dev2', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--exp_name', default='TEST', type=str)
    parser.add_argument('--pretrained_path', default='LLMs/zephyr-alpha', type=str)
    parser.add_argument('--kg_adapter_model_path', default='models/kg-adapter-llama', type=str)
    parser.add_argument('--kg_adapter_online_load', action='store_true')
    parser.add_argument('--save_path', default='ckpt/', type=str)
    parser.add_argument('--ckpt_path', default=None, type=str)
    parser.add_argument('--out_dir', default='outputs/', type=str)
    parser.add_argument('--peft_type', default='kg-adapter', type=str)

    # Data Setting
    parser.add_argument('--data_path', default='data', type=str)
    parser.add_argument('--test_data_path', default='data/all_data_test.csv', type=str)
    parser.add_argument('--train_data_version', default=None, type=str)
    parser.add_argument('--eval_data_version', default=None, type=str)
    parser.add_argument('--test_data_version', default=None, type=str)
    parser.add_argument('--test_set', default='', type=str)     # tuqa_mc1+tuqa_mc2+halueval
    parser.add_argument('--node_emb_path', default=None, type=str)

    # Kg-adapter Hyperparameters
    parser.add_argument('--exp_set', default='', type=str)  # loss_only_on_ans, no_kg, init_kg_emb, no_share_ca
    parser.add_argument('--num_relations', default=38, type=int)    # 11: cskg  772: wqsp 801: cwq
    parser.add_argument('--keep_ratio', default=1.0, type=float)
    parser.add_argument('--fuse_rate', default=1.0, type=float)       # control the rate of text rep fuse to kg rep
    parser.add_argument('--scaling_rate', default=1.0, type=float)    # same as the alpha / r in lora
    parser.add_argument('--kg_adapter_info_merge', default='gate', type=str)
    parser.add_argument('--kd_adapter_hidden_size', default=64, type=int)
    parser.add_argument('--kg_adapter_node_emb_size', default=100, type=int)
    parser.add_argument('--kg_adapter_enc_range', default='[2, 16]', type=str)
    parser.add_argument('--kg_adapter_dec_range', default='[16, 32]', type=str)

    # Ablation Studies Setting
    parser.add_argument('--ablation_exp_set', default='', type=str)
    parser.add_argument('--info_merge_pos', default='before', type=str) # before: merge_info -> SA; mid: SA->merge_info->FFN; after: FNN-> merge_info -> ...
        # no_kg_train, no_kg_test

    # Training Setting
    parser.add_argument('--strategy', default='auto', type=str)
    parser.add_argument('--accelerator', default='auto', type=str)
    parser.add_argument('--monitor', default='val_avg_acc', type=str)
    parser.add_argument('--save_top_k', default=0, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--precision', default='bf16-mixed', type=str)
    parser.add_argument('--devices', default='[3]')
    parser.add_argument('--num_workers', default=4, type=int)

    # Hyperparameters
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--warm_up_epoch', default=1, type=float)
    parser.add_argument('--micro_batch_size', default=2, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--max_epochs', default=10, type=int)  # don't change, it will affect the change of LR, when using lr scheduler
    parser.add_argument('--weight_decay', default=0.02, type=float)
    parser.add_argument('--max_seq_length', default=1024, type=int)

    args = parser.parse_args()
    try:
        main(args)
        # save return state for auto_run.py, if not use then don't care
        global_log = open("auto_run_log.txt", 'a+')
        global_log.write(str(args.exp_name))
        global_log.write('\n')
        global_log.flush()
    except:
        global_log = open("auto_error_log.txt", 'a+')
        global_log.write(str(args.exp_name))
        global_log.write('\n')
        global_log.flush()
        exstr = traceback.format_exc()
        print(exstr)

    global_log.close()

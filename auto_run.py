import os
import random
import sys
import time
import datetime
import threading
import torch

def gpu_info(gpu_index):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('\n')[gpu_index].split('|')
    power = int(gpu_status[1].split()[-3][:-1])
    used_memory = int(gpu_status[2].split('/')[0].strip()[:-3])
    all_memory = int(gpu_status[2].split('/')[-1].strip()[:-3])
    free_memory = all_memory - used_memory
    return power, free_memory


def waiting_gpu(interval=1, need_memory=28000):
    # need_memory_lst = [28000]  # [12000, 20000]
    need_memory_lst = [need_memory]
    num_lst = [1]
    gid = [x for x in range(torch.cuda.device_count())]
    while True:
        for need_memory, num in zip(need_memory_lst, num_lst):
            candid_gid_lst = []
            for gpu_id in gid:
                gpu_power, free_gpu_memory = gpu_info(gpu_id)
                if free_gpu_memory >= need_memory and check_used_gpu_num():
                    if gpu_id not in candid_gid_lst:
                        candid_gid_lst.append((free_gpu_memory, gpu_id))

                if free_gpu_memory < need_memory and gpu_id in candid_gid_lst:
                    candid_gid_lst.remove((free_gpu_memory, gpu_id))

                gpu = 'gpu id:%d' % gpu_id
                gpu_power_str = 'gpu power:%d W |' % gpu_power
                gpu_memory_str = 'free memory:%d MiB |' % free_gpu_memory
                gpu_select_rule = 'memory=%d ; num=%d |' % (need_memory, num)
                sys.stdout.write(
                    '\r' + gpu + ' ' + gpu_memory_str + ' ' + gpu_power_str + gpu_select_rule + " candid_gid_lst:" + str(
                        candid_gid_lst) + " | " +
                    "waiting cmd ids:" + str(set(range(len(cmds))) - set(running_id) - set(finished_id)) + " | "
                                                                                                           "running cmd ids:" + str(
                        running_id)
                )
                sys.stdout.flush()

            if len(candid_gid_lst) >= num:
                if len(running_id) == 0:
                    candid_gid_lst.sort(reverse=True)
                else:
                    candid_gid_lst.sort()
                candid_gid_lst = [x[1] for x in candid_gid_lst[:num]]
                candid_gid_lst.sort()
                return candid_gid_lst

            time.sleep(interval)


def check_used_gpu_num():
    global running_id
    now_running_num = len(running_id)

    if (datetime.datetime.now().hour >= 23 or datetime.datetime.now().hour <= 8) and now_running_num <= 3:
        return True
    if now_running_num <= 2:
        return True
    else:
        return False


def ExecCmd(idx, cmd, gpu_ids):
    global running_id
    global finished_id
    global cmd_screen
    try:
        print("-------------------------------")
        cnt = random.randint(0, 1000)
        print(f"start running: {cmd})")
        print(f"time: {datetime.datetime.now()} ||| screen_id: {cmd_screen[idx]}_{cnt}")
        os.system(f"screen -dmS {cmd_screen[idx]}_{cnt}")
        os.system(f'screen -x -S {cmd_screen[idx]}_{cnt} -p 0 -X stuff "conda activate LLM\n" ')
        if "devices" not in cmd:
            cmd = f"CUDA_VISIBLE_DEVICES={str(gpu_ids)[1:-1].replace(' ', '')} " + cmd
            cmd += " --devices '%s'" % str(gpu_ids)

        # split to avoid "too long remote command" error
        cmd1 = " ".join(cmd.split(" ")[:len(cmd.split(" ")) // 2]) + " "
        cmd2 = " ".join(cmd.split(" ")[len(cmd.split(" ")) // 2:])
        assert cmd1 + cmd2 == cmd

        os.system(f'screen -x -S {cmd_screen[idx]}_{cnt} -p 0 -X stuff "{cmd1}"')
        os.system(f'screen -x -S {cmd_screen[idx]}_{cnt} -p 0 -X stuff "{cmd2}\n"')
        running_id.append(idx)

        finished = False
        error = False
        now_task_name = eval(cmd.split("--exp_name ")[1].split(" --")[0])

        time.sleep(60 * 8)

        while (not finished) and (not error):
            with open("auto_run_log.txt", 'r') as f_log:
                finished_task = f_log.read()
                if now_task_name in finished_task:
                    finished = True

            with open("auto_error_log.txt", 'r+') as e_log:
                error_task = e_log.read()
                if now_task_name in error_task:
                    error = True
                    content = error_task.replace(now_task_name + "\n", "").replace(now_task_name, "")
                    with open("auto_error_log.txt", 'w') as tmp:
                        tmp.write(content)

            time.sleep(60 * 10)

        if finished:
            print("cmd %s is finished at %s" % (cmd, datetime.datetime.now()), "\n")
            running_id.remove(idx)
            finished_id.append(idx)
        elif error:
            running_id.remove(idx)
    except:
        print('fail at:', idx)
        running_id.remove(idx)


global running_id
global finished_id
global cmd_screen
global cmds

if __name__ == '__main__':
    running_id = []
    finished_id = []
    cmds = [

        # train
        (
             "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
             "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
             "--pretrained_path 'LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
             "--node_emb_path 'data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
             "--test_data_path 'data/all_test_3_v2.csv' "
             "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
             "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
             "--kg_adapter_node_emb_size 1024 --num_relations 38 "
             "--dev2 --save_top_k 3 "
        ),

        # load checkpoint and test
        (
            "python mymain.py --exp_name 'Our-bset-cast-study' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' --eval "
            "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
            "--pretrained_path 'LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
            "--node_emb_path 'data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
            "--ckpt_path 'ckpt/kg-adapterV4_lr5e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+V4_r1/peft_ckpt_epoch=3-step=312.bin' "
            "--test_data_path 'data/all_test_3_v2.csv' "
            "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens+output_sg' --monitor 'val_em' "
            "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
            "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        ),

        # ---------------Running commands for other experimental configurations for reference-----------------------#
        
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-5_wu1_llama2_mixdata_v5+use_edge_emb+mix_emb' --peft_type 'kg-adapter' --lr 5e-5 --warm_up_epoch 1 --strategy 'deepspeed' "
        #     "--train_data_version 'mixdata_v5' --test_data_version 'mixdata_v5' --eval_data_version 'v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_all_data_(71625,100)_nodes_emb.pt' "
        #     "--test_set 'use_predata+tuqa_mc1+halu' --monitor 'val_mc2' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-5_wu1_llama2_mixdata_v5+use_edge_emb+mix_emb' --peft_type 'kg-adapter' --lr 1e-5 --warm_up_epoch 1 --strategy 'deepspeed' "
        #     "--train_data_version 'mixdata_v5' --test_data_version 'mixdata_v5' --eval_data_version 'v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_all_data_(71625,100)_nodes_emb.pt' "
        #     "--test_set 'use_predata+tuqa_mc1+halu' --monitor 'val_mc2' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu1_llama2_mixdata_v7+use_edge_emb+mix_emb' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 1 --strategy 'deepspeed' "
        #     "--train_data_version 'mixdata_v7' --test_data_version 'mixdata_v7' --eval_data_version 'v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_all_data_(71625,100)_nodes_emb.pt' "
        #     "--test_set 'use_predata+tuqa_mc1+halu' --monitor 'val_mc2' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-5_wu1_llama2_mixdata_v7+use_edge_emb+mix_emb' --peft_type 'kg-adapter' --lr 5e-5 --warm_up_epoch 1 --strategy 'deepspeed' "
        #     "--train_data_version 'mixdata_v7' --test_data_version 'mixdata_v7' --eval_data_version 'v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_all_data_(71625,100)_nodes_emb.pt' "
        #     "--test_set 'use_predata+tuqa_mc1+halu' --monitor 'val_mc2' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-5_wu1_llama2_mixdata_v7+use_edge_emb+mix_emb' --peft_type 'kg-adapter' --lr 1e-5 --warm_up_epoch 1 --strategy 'deepspeed' "
        #     "--train_data_version 'mixdata_v7' --test_data_version 'mixdata_v7' --eval_data_version 'v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_all_data_(71625,100)_nodes_emb.pt' "
        #     "--test_set 'use_predata+tuqa_mc1+halu' --monitor 'val_mc2' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr7e-6_wu1_llama2_opca_mixdata_v6-1+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' --peft_type 'kg-adapter' --lr 7e-6 --warm_up_epoch 1 --strategy 'deepspeed' "
        #     "--train_data_version 'mixdata_v6-1_llama' --test_data_version 'mixdata_v6-1_llama' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama2-7b-openorca-mc-v1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_all_data_(91122,100)_nodes_emb.pt' "
        #     "--test_set 'tuqa_mc1+halu' --monitor 'val_mc2' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr2e-5_wu1_llama2_opca_mixdata_v6-1+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT_nt_sp' --peft_type 'kg-adapter' --lr 2e-5 --warm_up_epoch 1 --strategy 'deepspeed' "
        #     "--train_data_version 'mixdata_v6-1_llama' --test_data_version 'mixdata_v6-1_llama' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama2-7b-openorca-mc-v1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_all_data_(91122,100)_nodes_emb.pt' "
        #     "--test_set 'tuqa_mc1+halu+add_special_tokens' --monitor 'val_mc2' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr2e-5_wu1_llama2_opca_mixdata_v6-1+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT_nt_nosp' --peft_type 'kg-adapter' --lr 2e-5 --warm_up_epoch 1 --strategy 'deepspeed' "
        #     "--train_data_version 'mixdata_v6-1_llama' --test_data_version 'mixdata_v6-1_llama' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama2-7b-openorca-mc-v1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_all_data_(91122,100)_nodes_emb.pt' "
        #     "--test_set 'tuqa_mc1+halu' --monitor 'val_mc2' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-5_wu1_mistral-inst_mixdata_v9-1+use_edge_emb+mix_emb' --peft_type 'kg-adapter' --lr 1e-5 --warm_up_epoch 1 --strategy 'deepspeed' "
        #     "--train_data_version 'mixdata_v9-1_mistral' --test_data_version 'mixdata_v9-1_mistral' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-instruct' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_all_data_(91122,100)_nodes_emb.pt' "
        #     "--test_set 'tuqa_mc1+halu' --monitor 'val_mc2' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-6_wu0.1_mistral-base_mixdata_v9-1+use_edge_emb+mix_emb_nt' --peft_type 'kg-adapter' --lr 5e-6 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'mixdata_v9-1_mistral' --test_data_version 'mixdata_v9-1_mistral' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_all_data_(91122,100)_nodes_emb.pt' "
        #     "--test_set 'tuqa_mc1+tuqa_kg' --monitor 'val_avg_acc' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-6_wu0.1_mistral-base_mixdata_v9-1+use_edge_emb+mix_emb_nt_not_replace' --peft_type 'kg-adapter' --lr 5e-6 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'mixdata_v9-1_mistral' --test_data_version 'mixdata_v9-1_mistral' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_all_data_(91122,100)_nodes_emb.pt' "
        #     "--test_set 'tuqa_mc1+tuqa_kg+not_replace' --monitor 'val_avg_acc' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-7_wu0.1_mistral-base_mixdata_v9-1+use_edge_emb+mix_emb_nt' --peft_type 'kg-adapter' --lr 5e-7 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'mixdata_v9-1_mistral' --test_data_version 'mixdata_v9-1_mistral' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_all_data_(91122,100)_nodes_emb.pt' "
        #     "--test_set 'tuqa_mc1+tuqa_kg' --monitor 'val_avg_acc' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-7_wu0.1_mistral-base_mixdata_v9-1+use_edge_emb+mix_emb_nt_not_replace' --peft_type 'kg-adapter' --lr 5e-7 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'mixdata_v9-1_mistral' --test_data_version 'mixdata_v9-1_mistral' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_all_data_(91122,100)_nodes_emb.pt' "
        #     "--test_set 'tuqa_mc1+tuqa_kg+not_replace' --monitor 'val_avg_acc' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-5_wu0.1_mistral-base_mixdata_v9-1+use_edge_emb+mix_emb_nt_not_replace' --peft_type 'kg-adapter' --lr 5e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'mixdata_v9-1_mistral' --test_data_version 'mixdata_v9-1_mistral' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_all_data_(91122,100)_nodes_emb.pt' "
        #     "--test_set 'tuqa_mc1+tuqa_kg+not_replace' --monitor 'val_avg_acc' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_mistral-base_mixdata_v9-1+use_edge_emb+mix_emb_nt_not_replace' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'mixdata_v9-1_mistral' --test_data_version 'mixdata_v9-1_mistral' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_all_data_(91122,100)_nodes_emb.pt' "
        #     "--test_set 'tuqa_mc1+tuqa_kg+not_replace' --monitor 'val_avg_acc' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-5_wu0.1_mistral-base_mixdata_v9-1+use_edge_emb+mix_emb_nt_not_replace' --peft_type 'kg-adapter' --lr 1e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'mixdata_v9-1_mistral' --test_data_version 'mixdata_v9-1_mistral' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_all_data_(91122,100)_nodes_emb.pt' "
        #     "--test_set 'tuqa_mc1+tuqa_kg+not_replace' --monitor 'val_avg_acc' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-6_wu0.1_mistral-base_mixdata_v9-1+use_edge_emb+mix_emb_nt_not_replace' --peft_type 'kg-adapter' --lr 1e-6 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'mixdata_v9-1_mistral' --test_data_version 'mixdata_v9-1_mistral' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_all_data_(91122,100)_nodes_emb.pt' "
        #     "--test_set 'tuqa_mc1+tuqa_kg+not_replace' --monitor 'val_avg_acc' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-3_wu0.1_mistral-base_mixdata_v9-1+use_edge_emb+mix_emb_nt_not_replace' --peft_type 'kg-adapter' --lr 5e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'mixdata_v9-1_mistral' --test_data_version 'mixdata_v9-1_mistral' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_all_data_(91122,100)_nodes_emb.pt' "
        #     "--test_set 'tuqa_mc1+tuqa_kg+not_replace' --monitor 'val_avg_acc' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-7_wu0.1_mistral-base_mixdata_v9-1+use_edge_emb+mix_emb_nt_not_replace_re' --peft_type 'kg-adapter' --lr 5e-7 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'mixdata_v9-1_mistral' --test_data_version 'mixdata_v9-1_mistral' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_all_data_(91122,100)_nodes_emb.pt' "
        #     "--test_set 'tuqa_mc1+tuqa_kg+not_replace' --monitor 'val_avg_acc' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-3_wu0.1_zephyr-alpha_obqa_v1+use_edge_emb+mix_emb_nt_not_replace' --peft_type 'kg-adapter' --lr 5e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+not_replace' --monitor 'val_avg_acc' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-3_wu0.1_zephyr-alpha_obqa_v1+use_edge_emb+mix_emb_nt_replace' --peft_type 'kg-adapter' --lr 5e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst' --monitor 'val_avg_acc' "
        # ),
        #
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-2_wu0.1_zephyr-alpha_obqa_v1+use_edge_emb+mix_emb_nt_not_replace' --peft_type 'kg-adapter' --lr 5e-2 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+not_replace' --monitor 'val_avg_acc' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-2_wu0.1_zephyr-alpha_obqa_v1+use_edge_emb+mix_emb_nt_replace' --peft_type 'kg-adapter' --lr 5e-2 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst' --monitor 'val_avg_acc' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_zephyr-alpha_obqa_v1+use_edge_emb+mix_emb_nt_not_replace' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+not_replace' --monitor 'val_avg_acc' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_zephyr-alpha_obqa_v1+use_edge_emb+mix_emb_nt_replace_re' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst' --monitor 'val_avg_acc' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-5_wu0.1_zephyr-alpha_obqa_v1+use_edge_emb+mix_emb_nt' --peft_type 'kg-adapter' --lr 1e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst' --monitor 'val_avg_acc' "
        # ),
        #
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-4_wu0.1_zephyr-alpha_obqa_v1+use_edge_emb+mix_emb_nt' --peft_type 'kg-adapter' --lr 1e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst' --monitor 'val_avg_acc' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-5_wu0.1_zephyr-alpha_obqa_v1+use_edge_emb+mix_emb_nt' --peft_type 'kg-adapter' --lr 1e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst' --monitor 'val_avg_acc' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_zephyr-alpha_obqa_v1+use_edge_emb+mix_emb_nt_sp' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_avg_acc' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr7e-4_wu0.1_mistral-base_obqa_v1+use_edge_emb+mix_emb' --peft_type 'kg-adapter' --lr 7e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_mistral_v1' --test_data_version 'obqa_mistral_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg' --monitor 'val_avg_acc' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr7e-4_wu0.1_mistral-inst_obqa_v1+use_edge_emb+mix_emb' --peft_type 'kg-adapter' --lr 7e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_mistral_v1' --test_data_version 'obqa_mistral_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-instruct' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg' --monitor 'val_avg_acc' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr7e-4_wu0.1_llama-chat_obqa_v1+use_edge_emb+mix_emb' --peft_type 'kg-adapter' --lr 7e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2_v1' --test_data_version 'obqa_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b-chat' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst' --monitor 'val_avg_acc' "
        # ),
        #
        #
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr7e-4_wu0.1_mistral-base_csqa_v1+use_edge_emb+mix_emb' --peft_type 'kg-adapter' --lr 7e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_mistral_v1' --test_data_version 'csqa_mistral_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'csqa_kg' --monitor 'val_avg_acc' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr7e-4_wu0.1_mistral-inst_csqa_v1+use_edge_emb+mix_emb' --peft_type 'kg-adapter' --lr 7e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_mistral_v1' --test_data_version 'csqa_mistral_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-instruct' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'csqa_kg' --monitor 'val_avg_acc' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr7e-4_wu0.1_llama-chat_csqa_v1+use_edge_emb+mix_emb' --peft_type 'kg-adapter' --lr 7e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_llama2_v1' --test_data_version 'csqa_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b-chat' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'csqa_kg+no_user_inst+task_system_inst' --monitor 'val_avg_acc' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_zephyr-alpha_wqsp_v1+no_edge' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v1' --test_data_version 'wqsp_zephyr_v1' --eval_data_version '0' --num_relations 1 "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst' --monitor 'val_em' "
        # ),

        ################################################ TEST ################################################
        # (
        #     "python mymain.py --eval --exp_name 'mistral-base_cwq_test' --peft_type 'base' --strategy 'deepspeed' "
        #     "--train_data_version 'cwq_mistral_v1' --test_data_version 'cwq_mistral_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+no_node_emb+use_cat_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'cwq' "
        # ),
        # (
        #     "python mymain.py --eval --exp_name 'mistral-inst_cwq_test' --peft_type 'base' --strategy 'deepspeed' "
        #     "--train_data_version 'cwq_mistral_v1' --test_data_version 'cwq_mistral_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-instruct' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+no_node_emb+use_cat_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'cwq' "
        # ),
        # (
        #     "python mymain.py --eval --exp_name 'llama2-chat_cwq_test' --peft_type 'base' --strategy 'deepspeed' "
        #     "--train_data_version 'cwq_llama2_v1' --test_data_version 'cwq_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b-chat' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+no_node_emb+use_cat_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'cwq+no_user_inst+task_system_inst' "
        # ),
        # (
        #     "python mymain.py --eval --exp_name 'llama2-base_cwq_test' --peft_type 'base' --strategy 'deepspeed' "
        #     "--train_data_version 'cwq_llama2_v1' --test_data_version 'cwq_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+no_node_emb+use_cat_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'cwq+no_user_inst+task_system_inst' "
        # ),

        ################################################ /TEST/ ################################################

        ################################################ Ablation Studies ################################################

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+ab_linear_emb' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2  --ablation_exp_set 'linear_emb' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+ab_linear_emb+no_kg' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2  --ablation_exp_set 'linear_emb+no_kg' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr6e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+ab_pos_mid' --peft_type 'kg-adapter' --lr 6e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2  --info_merge_pos 'mid' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr6e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+ab_pos_after' --peft_type 'kg-adapter' --lr 6e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2  --info_merge_pos 'after' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+ab_d_16' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 --kd_adapter_hidden_size 16 "
        #     "--dev2  "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr6e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+ab_d_32' --peft_type 'kg-adapter' --lr 6e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 --kd_adapter_hidden_size 32 "
        #     "--dev2  "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr1e-3_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+ab_d_128' --peft_type 'kg-adapter' --lr 1e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 --kd_adapter_hidden_size 128 "
        #     "--dev2  "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr6e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+ab_no_res' --peft_type 'kg-adapter' --lr 6e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2  --ablation_exp_set 'no_res' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr6e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+ab_linear_scale' --peft_type 'kg-adapter' --lr 6e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2  --ablation_exp_set 'linear_scale' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+ab_no_mix_emb' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 "
        # ),

        # -----------------------------------------llama-3b-------------------------#
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama-3b_obqa_v2+SRGAT_[dec]_26+V4_r1' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama-3b_v2' --test_data_version 'obqa_llama-3b_v2' --eval_data_version 'obqa_llama-3b_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/open_llama_3b_v2' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,26] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 --save_top_k 3 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama-3b_csqa_v2+SRGAT_[dec]_26+V4_r1' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_llama-3b_v2' --test_data_version 'csqa_llama-3b_v2' --eval_data_version 'csqa_llama-3b_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/open_llama_3b_v2' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,26] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 --save_top_k 3 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama-3b_obqa_v2+SRGAT_[dec]_26+ab_no_kg' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama-3b_v2' --test_data_version 'obqa_llama-3b_v2' --eval_data_version 'obqa_llama-3b_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/open_llama_3b_v2' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,26] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2  --ablation_exp_set 'no_kg' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama-3b_obqa_v2+SRGAT_[dec]_26+ab_no_gnn' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama-3b_v2' --test_data_version 'obqa_llama-3b_v2' --eval_data_version 'obqa_llama-3b_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/open_llama_3b_v2' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,26] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2  --ablation_exp_set 'no_gnn' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama-3b_obqa_v2+SRGAT_[dec]_26+ab_no_trip' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama-3b_v2' --test_data_version 'obqa_llama-3b_v2' --eval_data_version 'obqa_llama-3b_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/open_llama_3b_v2' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,26] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama-3b_obqa_v2+SRGAT_[dec]_26+ab_no_mix_emb' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama-3b_v2' --test_data_version 'obqa_llama-3b_v2' --eval_data_version 'obqa_llama-3b_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/open_llama_3b_v2' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,26] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 "
        # ),
        # -----------------------------------------llama-3b-------------------------#

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_mistral-7b-base_obqa_v2+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_mistral_v2' --test_data_version 'obqa_mistral_v2' --eval_data_version 'obqa_mistral_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b-chat' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 --save_top_k 3 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama2-7b-chat_obqa_v2+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2_v2' --test_data_version 'obqa_llama2_v2' --eval_data_version 'obqa_llama2_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b-chat' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 --save_top_k 3 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama2-7b-chat_csqa_v2+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_llama2_v2' --test_data_version 'csqa_llama2_v2' --eval_data_version 'csqa_llama2_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b-chat' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 --save_top_k 3 "
        # ),

        # ----------------------------------- TEST ------------------------------
        # (
        #     "python mymain.py --exp_name 'Our-bset-cast-study' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' --eval "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--ckpt_path '/raid_sdb/home/tsy/KGLLM_ckpt/kg-adapterV4_lr5e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+V4_r1/peft_ckpt_epoch=3-step=312.bin' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens+output_sg' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        # ),

        # (
        #     "python mymain.py --exp_name 'mistral-7b-base_obqa_v2' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' --eval --peft_type 'base' "
        #     "--train_data_version 'obqa_mistral_v2' --test_data_version 'obqa_mistral_v2' --eval_data_version 'obqa_mistral_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        # ),
        # (
        #     "python mymain.py --exp_name 'llama-13b-base_obqa_v2' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' --eval --peft_type 'base' "
        #     "--train_data_version 'obqa_llama2-13b_v2' --test_data_version 'obqa_llama2-13b_v2' --eval_data_version 'obqa_llama2-13b_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/Llama-2-13b-hf' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        # ),
        # (
        #     "python mymain.py --exp_name 'mistral-7b-inst_obqa_v2' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' --eval --peft_type 'base' "
        #     "--train_data_version 'obqa_mistral_v2' --test_data_version 'obqa_mistral_v2' --eval_data_version 'obqa_mistral_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-instruct' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        # ),
        # (
        #     "python mymain.py --exp_name 'llama-3b-base_obqa_v2' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' --eval --peft_type 'base' "
        #     "--train_data_version 'obqa_llama-3b_v2' --test_data_version 'obqa_llama-3b_v2' --eval_data_version 'obqa_llama-3b_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/open_llama_3b_v2' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        # ),


        # (
        #     "python mymain.py --exp_name 'zephyr-7b_obqa_v2' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' --eval --peft_type 'base' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        # ),
        # (
        #     "python mymain.py --exp_name 'zephyr-7b_csqa_v2' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' --eval --peft_type 'base' "
        #     "--train_data_version 'csqa_zephyr_v2' --test_data_version 'csqa_zephyr_v2' --eval_data_version 'csqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        # ),
        # (
        #     "python mymain.py --exp_name 'zephyr-7b_wqsp_v1' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' --eval --peft_type 'base' "
        #     "--train_data_version 'wqsp_zephyr_v1' --test_data_version 'wqsp_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        # ),
        # (
        #     "python mymain.py --exp_name 'zephyr-7b_cwq_v1' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' --eval --peft_type 'base' "
        #     "--train_data_version 'cwq_zephyr_v1' --test_data_version 'cwq_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'cwq_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        # ),


        # (
        #     "python mymain.py --exp_name 'llama2-7b-base_obqa_v2' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' --eval --peft_type 'base' "
        #     "--train_data_version 'obqa_llama2_v2' --test_data_version 'obqa_llama2_v2' --eval_data_version 'obqa_llama2_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        # ),
        # (
        #     "python mymain.py --exp_name 'llama2-7b-base_csqa_v2' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' --eval --peft_type 'base' "
        #     "--train_data_version 'csqa_llama2_v2' --test_data_version 'csqa_llama2_v2' --eval_data_version 'csqa_llama2_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        # ),
        # (
        #     "python mymain.py --exp_name 'llama2-7b-base_wqsp_v2' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' --eval --peft_type 'base' "
        #     "--train_data_version 'wqsp_llama2_v1' --test_data_version 'wqsp_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        # ),
        # (
        #     "python mymain.py --exp_name 'llama2-7b-base_cwq_v2' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' --eval --peft_type 'base' "
        #     "--train_data_version 'cwq_llama2_v1' --test_data_version 'cwq_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'cwq_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        # ),

        # (
        #     "python mymain.py --exp_name 'llama2-7b-chat_obqa_v2' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' --eval --peft_type 'base' "
        #     "--train_data_version 'obqa_llama2_v2' --test_data_version 'obqa_llama2_v2' --eval_data_version 'obqa_llama2_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b-chat' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        # ),
        # (
        #     "python mymain.py --exp_name 'llama2-7b-chat_csqa_v2' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' --eval --peft_type 'base' "
        #     "--train_data_version 'csqa_llama2_v2' --test_data_version 'csqa_llama2_v2' --eval_data_version 'csqa_llama2_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b-chat' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        # ),
        # (
        #     "python mymain.py --exp_name 'llama2-7b-chat_wqsp_v1' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' --eval --peft_type 'base' "
        #     "--train_data_version 'wqsp_llama2_v1' --test_data_version 'wqsp_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b-chat' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        # ),
        # (
        #     "python mymain.py --exp_name 'llama2-7b-chat_cwq_v1' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' --eval --peft_type 'base' "
        #     "--train_data_version 'cwq_llama2_v1' --test_data_version 'cwq_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b-chat' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'cwq_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        # ),


        # -----------------------------------------llama2-7b-------------------------#
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama2-7b_obqa_v2+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2_v2' --test_data_version 'obqa_llama2_v2' --eval_data_version 'obqa_llama2_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 --save_top_k 3 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama2-7b_csqa_v2+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_llama2_v2' --test_data_version 'csqa_llama2_v2' --eval_data_version 'csqa_llama2_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 --save_top_k 3 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama2-7b_obqa_v2+SRGAT_[dec]_32+ab_no_kg' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2_v2' --test_data_version 'obqa_llama2_v2' --eval_data_version 'obqa_llama2_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2  --ablation_exp_set 'no_kg' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama2-7b_obqa_v2+SRGAT_[dec]_32+ab_no_gnn' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2_v2' --test_data_version 'obqa_llama2_v2' --eval_data_version 'obqa_llama2_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2  --ablation_exp_set 'no_gnn' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama2-7b_obqa_v2+SRGAT_[dec]_32+ab_no_trip' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2_v2' --test_data_version 'obqa_llama2_v2' --eval_data_version 'obqa_llama2_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama2-7b_obqa_v2+SRGAT_[dec]_32+ab_no_mix_emb' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2_v2' --test_data_version 'obqa_llama2_v2' --eval_data_version 'obqa_llama2_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 "
        # ),
        # -----------------------------------------llama2-7b-------------------------#

        # ---------------------------------OBQA---------------------------------------#
        # (
        #     "python mymain.py --exp_name 'lora_64_lr1e-5_wu0.1_llama2_obqa_v2' --peft_type 'kg-adapter' --lr 1e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2_v2' --test_data_version 'obqa_llama2_v2' --eval_data_version 'obqa_llama2_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        #     "--peft_type 'peft_lora' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr6e-4_wu0.1_llama2-7b_obqa_v2+SRGAT_[dec]_32_r2' --peft_type 'kg-adapter' --lr 6e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2_v2' --test_data_version 'obqa_llama2_v2' --eval_data_version 'obqa_llama2_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama-3b_obqa_v2+SRGAT_[dec]_26_r2' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama-3b_v2' --test_data_version 'obqa_llama-3b_v2' --eval_data_version 'obqa_llama-3b_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/open_llama_3b_v2' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,26] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama2-13b_obqa_v2+SRGAT_[dec]_40_r2' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2-13b_v2' --test_data_version 'obqa_llama2-13b_v2' --eval_data_version 'obqa_llama2-13b_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/Llama-2-13b-hf' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,40] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr1e-3_wu0.1_llama2-7b-chat_obqa_v2+SRGAT_[dec]_32_r2' --peft_type 'kg-adapter' --lr 1e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2_v2' --test_data_version 'obqa_llama2_v2' --eval_data_version 'obqa_llama2_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b-chat' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_mistral-base-7b_obqa_v2+SRGAT_[dec]_32_r2' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_mistral_v2' --test_data_version 'obqa_mistral_v2' --eval_data_version 'obqa_mistral_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_mistral-inst-7b_obqa_v2+SRGAT_[dec]_32_r2' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_mistral_v2' --test_data_version 'obqa_mistral_v2' --eval_data_version 'obqa_mistral_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-instruct' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 "
        # ),
        # ---------------------------------CSQA---------------------------------------#
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama2-base_v2+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_llama2_v2' --test_data_version 'csqa_llama2_v2' --eval_data_version 'csqa_llama2_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--save_top_k 3 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_csqa_v2+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_zephyr_v2' --test_data_version 'csqa_zephyr_v2' --eval_data_version 'csqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--save_top_k 3 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr2e-4_wu0.1_zephyr_csqa_v2+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 2e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_zephyr_v2' --test_data_version 'csqa_zephyr_v2' --eval_data_version 'csqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--save_top_k 3 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr1e-4_wu0.1_zephyr_csqa_v2+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 1e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_zephyr_v2' --test_data_version 'csqa_zephyr_v2' --eval_data_version 'csqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--save_top_k 3 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr9e-5_wu0.1_zephyr_csqa_v2+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 9e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_zephyr_v2' --test_data_version 'csqa_zephyr_v2' --eval_data_version 'csqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--save_top_k 3 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr8e-5_wu0.1_zephyr_csqa_v2+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 8e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_zephyr_v2' --test_data_version 'csqa_zephyr_v2' --eval_data_version 'csqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--save_top_k 3 "
        # ),

        # ----------------------------MedQA-------------------------------------------#

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr1e-3_wu0.1_zephyr_medqa_v2+SRGAT_[dec]_32' --peft_type 'kg-adapter' --lr 1e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'medqa_zephyr_v2' --test_data_version 'medqa_zephyr_v2' --eval_data_version 'medqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/MedQA_(3448,768)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'medqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 768 --num_relations 38 --max_seq_length 512 "
        # ),
        # (
        #     "python mymain.py --exp_name 'lora_64_lr1e-5_wu0.1_zephyr_medqa_max512' --peft_type 'kg-adapter' --lr 1e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'medqa_zephyr_v2' --test_data_version 'medqa_zephyr_v2' --eval_data_version 'medqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'medqa+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        #     "--peft_type 'peft_lora' --max_seq_length 512 "
        # ),
        # (
        #     "python mymain.py --exp_name 'lora_64_lr5e-5_wu0.1_zephyr_medqa_max512' --peft_type 'kg-adapter' --lr 5e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'medqa_zephyr_v2' --test_data_version 'medqa_zephyr_v2' --eval_data_version 'medqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'medqa+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        #     "--peft_type 'peft_lora' --max_seq_length 512 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_medqa_v2+SRGAT_[dec]_32_max512' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'medqa_zephyr_v2' --test_data_version 'medqa_zephyr_v2' --eval_data_version 'medqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/MedQA_(3448,768)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'medqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 768 --num_relations 38 --max_seq_length 512 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-5_wu0.1_zephyr_medqa_v2+SRGAT_[dec]_32_max512' --peft_type 'kg-adapter' --lr 5e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'medqa_zephyr_v2' --test_data_version 'medqa_zephyr_v2' --eval_data_version 'medqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/MedQA_(3448,768)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'medqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 768 --num_relations 38 --max_seq_length 512 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-5_wu0.1_zephyr_medqa_v2+SRGAT_[dec]_32_max512+no_kg' --peft_type 'kg-adapter' --lr 5e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'medqa_zephyr_v2' --test_data_version 'medqa_zephyr_v2' --eval_data_version 'medqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/MedQA_(3448,768)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'medqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 768 --num_relations 38 --max_seq_length 512 "
        #     "--ablation_exp_set 'no_kg' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-5_wu0.1_zephyr_medqa_v2+SRGAT_[dec]_32_max512' --peft_type 'kg-adapter' --lr 5e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'medqa_zephyr_v2' --test_data_version 'medqa_zephyr_v2' --eval_data_version 'medqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/MedQA_(3448,768)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'medqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 768 --num_relations 38 --max_seq_length 512 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr1e-4_wu0.1_zephyr_medqa_v2+SRGAT_[dec]_32' --peft_type 'kg-adapter' --lr 1e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'medqa_zephyr_v2' --test_data_version 'medqa_zephyr_v2' --eval_data_version 'medqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/MedQA_(3448,768)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'medqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 768 --num_relations 38 --max_seq_length 512 "
        # ),
        # -------------------------/MedQA/--------------------------------------------#


        # -----------------------------------------WQSP--------------------------------------#
            # ---------------------------- zephyr ----------------------#

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr1e-3_wu0.1_zephyr_wqsp_v3+SRGAT_[dec]_max512_32' --peft_type 'kg-adapter' --lr 1e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v3' --test_data_version 'wqsp_zephyr_v3' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/WQSP+CWQ_(66791,50)_FreeBase_TransE_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/wqsp+cwq_test.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 50 --num_relations 1070 "
        #     "--max_seq_length 512 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr1e-4_wu0.1_zephyr_wqsp_v3+SRGAT_[dec]_max512_32' --peft_type 'kg-adapter' --lr 1e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v3' --test_data_version 'wqsp_zephyr_v3' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/WQSP+CWQ_(66791,50)_FreeBase_TransE_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/wqsp+cwq_test.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 50 --num_relations 1070 "
        #     "--max_seq_length 512 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr7e-4_wu0.1_zephyr_wqsp_v3+SRGAT_[dec]_max1024_32' --peft_type 'kg-adapter' --lr 7e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v3' --test_data_version 'wqsp_zephyr_v3' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/WQSP+CWQ_(66791,50)_FreeBase_TransE_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/wqsp+cwq_test.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 50 --num_relations 1070 "
        #     "--max_seq_length 1024 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr3e-4_wu0.1_zephyr_wqsp_v3+SRGAT_[dec]_max1024_32' --peft_type 'kg-adapter' --lr 3e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v3' --test_data_version 'wqsp_zephyr_v3' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/WQSP+CWQ_(66791,50)_FreeBase_TransE_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/wqsp+cwq_test.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 50 --num_relations 1070 "
        #     "--max_seq_length 1024 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr2e-4_wu0.1_zephyr_wqsp_v3+SRGAT_[dec]_max1024_32+no_kg' --peft_type 'kg-adapter' --lr 2e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v3' --test_data_version 'wqsp_zephyr_v3' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/WQSP+CWQ_(66791,50)_FreeBase_TransE_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/wqsp+cwq_test.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 50 --num_relations 1070 "
        #     "--max_seq_length 1024 --ablation_exp_set 'no_kg' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr2e-4_wu0.1_zephyr_wqsp_v3+SRGAT_[dec]_max1024_32+no_KGE' --peft_type 'kg-adapter' --lr 2e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v3' --test_data_version 'wqsp_zephyr_v3' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/wqsp+cwq_test.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 --num_relations 1070 "
        #     "--max_seq_length 1024 "
        # ),

        (
            "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_wqsp_v3+SRGAT_[dec]_max1024_32+ep20' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
            "--train_data_version 'wqsp_zephyr_v3' --test_data_version 'wqsp_zephyr_v3' --eval_data_version '0' "
            "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
            "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/WQSP+CWQ_(66791,50)_FreeBase_TransE_emb.pt' "
            "--test_data_path '/raid_sdb/home/tsy/KG_data/wqsp+cwq_test.csv' "
            "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
            "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
            "--kg_adapter_node_emb_size 50 --num_relations 1070 "
            "--max_seq_length 1024 --max_epochs 20 --num_epochs 20 --patience 8 "
        ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr8e-5_wu0.1_zephyr_wqsp_v3+SRGAT_[dec]_max1024_32' --peft_type 'kg-adapter' --lr 8e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v3' --test_data_version 'wqsp_zephyr_v3' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/WQSP+CWQ_(66791,50)_FreeBase_TransE_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/wqsp+cwq_test.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 50 --num_relations 1070 "
        #     "--max_seq_length 1024 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr4e-4_wu0.1_zephyr_wqsp_v3+SRGAT_[dec]_max1024_32' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v3' --test_data_version 'wqsp_zephyr_v3' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/WQSP+CWQ_(66791,50)_FreeBase_TransE_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/wqsp+cwq_test.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 50 --num_relations 1070 "
        #     "--max_seq_length 1024 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr6e-4_wu0.1_zephyr_wqsp_v3+SRGAT_[dec]_max1024_32' --peft_type 'kg-adapter' --lr 6e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v3' --test_data_version 'wqsp_zephyr_v3' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/WQSP+CWQ_(66791,50)_FreeBase_TransE_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/wqsp+cwq_test.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 50 --num_relations 1070 "
        #     "--max_seq_length 1024 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_wqsp_v3+SRGAT_[dec]_32_max512+no_kg' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v3' --test_data_version 'wqsp_zephyr_v3' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/WQSP+CWQ_(66791,50)_FreeBase_TransE_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/wqsp+cwq_test.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 50 --num_relations 1070 "
        #     "--max_seq_length 512  --ablation_exp_set 'no_kg' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_wqsp_v2+SRGAT_[dec]_32+V4' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v2' --test_data_version 'wqsp_zephyr_v2' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/WQSP+CWQ_(155696,50)_FreeBase_TransE_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--num_relations 805 "
        #     "--max_seq_length 512 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_wqsp_v1+SRGAT_[dec]_32+V4_r2' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v1' --test_data_version 'wqsp_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--num_relations 805 "
        #     "--save_top_k 3 --max_seq_length 512 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_wqsp_v1+SRGAT_[dec]_32+V4_r2_no_edge_emb' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v1' --test_data_version 'wqsp_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+mix_emb+use_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--num_relations 805 "
        #     "--save_top_k 3 --max_seq_length 512 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr4e-4_wu0.1_zephyr_wqsp_v1+SRGAT_[dec]_32+V4_r3' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v1' --test_data_version 'wqsp_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--num_relations 805 "
        #     "--save_top_k 3 --max_seq_length 1024 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr2e-3_wu0.1_llama2-7b_wqsp_v1+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 2e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_llama2_v1' --test_data_version 'wqsp_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--num_relations 805 "
        #     "--save_top_k 3 --max_seq_length 1024 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr2e-4_wu0.1_zephyr_wqsp_v1+SRGAT_[dec]_32+V4_r2' --peft_type 'kg-adapter' --lr 2e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v1' --test_data_version 'wqsp_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--num_relations 805 "
        #     "--save_top_k 0 --max_seq_length 1024 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr1e-4_wu0.1_zephyr_wqsp_v1+SRGAT_[dec]_32+V4_r2' --peft_type 'kg-adapter' --lr 1e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v1' --test_data_version 'wqsp_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--num_relations 805 "
        #     "--save_top_k 0 --max_seq_length 1024 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr1e-3_wu0.1_llama2-7b_wqsp_v1+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 1e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_llama2_v1' --test_data_version 'wqsp_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--num_relations 805 "
        #     "--save_top_k 3 --max_seq_length 1024 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr6e-4_wu0.1_zephyr_wqsp_v1+SRGAT_[dec]_32+V4_r2' --peft_type 'kg-adapter' --lr 6e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v1' --test_data_version 'wqsp_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--num_relations 805 "
        #     "--save_top_k 3 --max_seq_length 1024 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr8e-4_wu0.1_zephyr_wqsp_v1+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 8e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v1' --test_data_version 'wqsp_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--num_relations 805 "
        #     "--save_top_k 3 --max_seq_length 1024 "
        # ),
            # ------------------- llama-7b ----------------------------------#
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr1e-3_wu0.1_llama2-7b_wqsp_v1+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 1e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_llama2_v1' --test_data_version 'wqsp_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--num_relations 805 "
        #     "--save_top_k 3 --max_seq_length 1024 "
        # ),

        # -----------------------------------------GraphextQA--------------------------------------#
        (
            "python mymain.py --exp_name 'kg-adapterV4_lr5e-5_wu0.1_zephyr_graphextqa+SRGAT_[dec]_32_max800_ep20' --peft_type 'kg-adapter' --lr 5e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
            "--train_data_version 'graphextqa_zephyr_v2' --test_data_version 'graphextqa_zephyr_v2' --eval_data_version '0' "
            "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
            "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/graphextqa_(41224,512)_Wikidata5m_SimplE_emb.pt' "
            "--test_data_path '/raid_sdb/home/tsy/KG_data/graphextqa_test.csv' "
            "--test_set 'graphextqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
            "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
            "--kg_adapter_node_emb_size 512 --num_relations 495 --max_seq_length 800 --max_epochs 20 --num_epochs 20 "
        ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_graphextqa+SRGAT_[dec]_32+no_kg' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'graphextqa_zephyr_v2' --test_data_version 'graphextqa_zephyr_v2' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/graphextqa_(41224,512)_Wikidata5m_SimplE_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/graphextqa_test.csv' "
        #     "--test_set 'graphextqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 512 --num_relations 495 --max_seq_length 512 "
        #     "--ablation_exp_set 'no_kg' "
        # ),
        # (
        #     "python mymain.py --exp_name 'lora_64_lr5e-5_wu0.1_zephyr_graphextqa_max512' --peft_type 'kg-adapter' --lr 5e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'graphextqa_zephyr_v2' --test_data_version 'graphextqa_zephyr_v2' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/graphextqa_test.csv' "
        #     "--test_set 'graphextqa+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        #     "--peft_type 'peft_lora' --max_seq_length 512 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr1e-4_wu0.1_zephyr_graphextqa+SRGAT_[dec]_32_max512' --peft_type 'kg-adapter' --lr 1e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'graphextqa_zephyr_v2' --test_data_version 'graphextqa_zephyr_v2' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/graphextqa_(41224,512)_Wikidata5m_SimplE_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/graphextqa_test.csv' "
        #     "--test_set 'graphextqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 512 --num_relations 495 --max_seq_length 512 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr1e-4_wu0.1_zephyr_graphextqa+SRGAT_[dec]_32_max512' --peft_type 'kg-adapter' --lr 1e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'graphextqa_zephyr_v2' --test_data_version 'graphextqa_zephyr_v2' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/graphextqa_(41224,512)_Wikidata5m_SimplE_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/graphextqa_test.csv' "
        #     "--test_set 'graphextqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 512 --num_relations 495 --max_seq_length 512 "
        # ),
        # -----------------------------------------/GraphextQA/--------------------------------------#

        # --------------------------------------------CWQ-----------------------------#
            # ---------------------------- zephyr ----------------------#

        (
            "python mymain.py --exp_name 'kg-adapterV4_lr7e-4_wu0.1_zephyr_cwq_v3+SRGAT_[dec]_32_max1024' --peft_type 'kg-adapter' --lr 7e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
            "--train_data_version 'cwq_zephyr_v3' --test_data_version 'cwq_zephyr_v3' --eval_data_version '0' "
            "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
            "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/WQSP+CWQ_(66791,50)_FreeBase_TransE_emb.pt' "
            "--test_data_path '/raid_sdb/home/tsy/KG_data/wqsp+cwq_test.csv' "
            "--test_set 'cwq+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
            "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
            "--kg_adapter_node_emb_size 50 --num_relations 1070 "
            "--max_seq_length 1024 "
        ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_cwq_v3+SRGAT_[dec]_32_max512+no_kg' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'cwq_zephyr_v3' --test_data_version 'cwq_zephyr_v3' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/WQSP+CWQ_(66791,50)_FreeBase_TransE_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/wqsp+cwq_test.csv' "
        #     "--test_set 'cwq+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 50 --num_relations 1070 "
        #     "--max_seq_length 1024 --ablation_exp_set 'no_kg' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr7e-5_wu0.1_zephyr_cwq_v1+SRGAT_[dec]_32+V4_no_edge_emb' --peft_type 'kg-adapter' --lr 5e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'cwq_zephyr_v1' --test_data_version 'cwq_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+mix_emb+use_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'cwq+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--num_relations 805 "
        #     "--save_top_k 0 --max_seq_length 2048 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr1e-3_wu0.1_zephyr_cwq_v1+SRGAT_[dec]_32+V4_r2' --peft_type 'kg-adapter' --lr 1e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'cwq_zephyr_v1' --test_data_version 'cwq_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'cwq+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--num_relations 805 "
        #     "--save_top_k 0 --max_seq_length 2048 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr7e-4_wu0.1_zephyr_cwq_v1+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 7e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'cwq_zephyr_v1' --test_data_version 'cwq_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'cwq+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--num_relations 805 "
        #     "--save_top_k 0 --max_seq_length 2048 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr1e-3_wu0.1_zephyr_cwq_v1+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 1e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'cwq_zephyr_v1' --test_data_version 'cwq_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'cwq+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--num_relations 805 "
        #     "--save_top_k 3 --max_seq_length 2048 "
        # ),
            #------------------- llama-7b ----------------------------------#
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr1e-4_wu0.1_llama2-7b_cwq_v1+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 1e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'cwq_llama2_v1' --test_data_version 'cwq_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'cwq+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--num_relations 805 "
        #     "--save_top_k 3 --max_seq_length 2048 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-5_wu0.1_llama2-7b_cwq_v1+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 5e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'cwq_llama2_v1' --test_data_version 'cwq_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'cwq+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--num_relations 805 "
        #     "--save_top_k 3 --max_seq_length 2048 "
        # ),


        # -----------------------------------------llama2-13b-------------------------#
        # (
        #     "python mymain.py --exp_name 'llama-13b-base_obqa_v2' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' --eval --peft_type 'base' "
        #     "--train_data_version 'obqa_llama2-13b_v2' --test_data_version 'obqa_llama2-13b_v2' --eval_data_version 'obqa_llama2-13b_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/Llama-2-13b-hf' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        # ),
        # (
        #     "python mymain.py --exp_name 'llama-13b-chat_obqa_v2' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' --eval --peft_type 'base' "
        #     "--train_data_version 'obqa_llama2-13b_v2' --test_data_version 'obqa_llama2-13b_v2' --eval_data_version 'obqa_llama2-13b_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/Llama-2-13b-chat-hf' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        # ),
        #
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama2-13b_obqa_v2+SRGAT_[dec]_40+V4_r1' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2-13b_v2' --test_data_version 'obqa_llama2-13b_v2' --eval_data_version 'obqa_llama2-13b_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/Llama-2-13b-hf' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,40] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 --save_top_k 3 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama2-13b_obqa_v2+SRGAT_[dec]_40+ab_no_kg' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2-13b_v2' --test_data_version 'obqa_llama2-13b_v2' --eval_data_version 'obqa_llama2-13b_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/Llama-2-13b-hf' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,40] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2  --ablation_exp_set 'no_kg' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama2-13b_obqa_v2+SRGAT_[dec]_40+ab_no_gnn' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2-13b_v2' --test_data_version 'obqa_llama2-13b_v2' --eval_data_version 'obqa_llama2-13b_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/Llama-2-13b-hf' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,40] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2  --ablation_exp_set 'no_gnn' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama2-13b_obqa_v2+SRGAT_[dec]_40+ab_no_trip' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2-13b_v2' --test_data_version 'obqa_llama2-13b_v2' --eval_data_version 'obqa_llama2-13b_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/Llama-2-13b-hf' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,40] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_llama2-13b_obqa_v2+SRGAT_[dec]_40+ab_no_mix_emb' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2-13b_v2' --test_data_version 'obqa_llama2-13b_v2' --eval_data_version 'obqa_llama2-13b_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/Llama-2-13b-hf' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,40] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 "
        # ),
        # -----------------------------------------llama2-13b-------------------------#

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+ab_no_kg' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2  --ablation_exp_set 'no_kg' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+ab_no_gnn' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2  --ablation_exp_set 'no_gnn' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+ab_no_trip' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 "
        # ),
        #
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+ab_no_mix_emb' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 "
        # ),
        #
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr6e-4_wu0.1_zephyr_obqa_v2+SRGAT+ab_[dec]_24' --peft_type 'kg-adapter' --lr 6e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,24] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr6e-4_wu0.1_zephyr_obqa_v2+SRGAT+ab_[dec]_16' --peft_type 'kg-adapter' --lr 6e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,16] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr6e-4_wu0.1_zephyr_obqa_v2+SRGAT+ab_[dec]_8' --peft_type 'kg-adapter' --lr 6e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,8] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr6e-4_wu0.1_zephyr_obqa_v2+SRGAT+ab_[dec]_16_back' --peft_type 'kg-adapter' --lr 6e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [16,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_csqa_v1+SRGAT+ab_[enc,dec]+no_kg_train&test' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_zephyr_v1' --test_data_version 'csqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'csqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--ablation_exp_set 'no_kg' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_llama2_obqa_v1+SRGAT+ab_[enc,dec]+no_kg_train&test' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2_v1' --test_data_version 'obqa_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--ablation_exp_set 'no_kg' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_llama2_obqa_v1+SRGAT+ab_[enc,dec]_32' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2_v1' --test_data_version 'obqa_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_llama2_csqa_v1+SRGAT+ab_[enc,dec]+no_kg_train&test' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_llama2_v1' --test_data_version 'csqa_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'csqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--ablation_exp_set 'no_kg' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_llama2_csqa_v1+SRGAT+ab_[enc,dec]_32' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_llama2_v1' --test_data_version 'csqa_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'csqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+ab_[enc,dec]+no_kg_train&test' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--ablation_exp_set 'no_kg' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+ab_[enc,dec]+no_kg_test' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),
        #
        #
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+ab_[enc,dec]+mix_emb' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+no_node_emb+mix_emb+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+ab_[enc,dec]+mix_emb+trip_rep' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+no_node_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+ab_[enc,dec]+node_emb' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+ab_[enc,dec]+node_emb+edge_emb' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+ab_[enc,dec]+node_emb+edge_emb+mix_emb' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),
        #
        #
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+ab_[enc,dec]+no_enc' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+ab_[enc,dec]+no_gnn' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--ablation_exp_set 'no_gnn' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+ab_[enc,dec]+d_32' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --kd_adapter_hidden_size 32 --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+ab_[enc,dec]+d_100' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --kd_adapter_hidden_size 100 --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+ab_[enc,dec]+d_128' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --kd_adapter_hidden_size 128 --keep_ratio 1.0 "
        # ),
        #
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+ab_[enc,dec]+hop1' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_hop1' --test_data_version 'obqa_zephyr_hop1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+ab_[enc,dec]+hop3' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_hop3' --test_data_version 'obqa_zephyr_hop3' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'lora_64_lr1e-4_wu0.1_llama2_csqa_v1' --peft_type 'kg-adapter' --lr 1e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_llama2_v1' --test_data_version 'csqa_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'csqa_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        #     "--peft_type 'peft_lora' "
        # ),
        # (
        #     "python mymain.py --exp_name 'lora_64_lr1e-4_wu0.1_llama2_obqa_v1' --peft_type 'kg-adapter' --lr 1e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2_v1' --test_data_version 'obqa_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        #     "--peft_type 'peft_lora' "
        # ),

        # (
        #     "python mymain.py --exp_name 'lora_64_lr5e-5_wu0.1_zephyr-alpha_obqa_v1' --peft_type 'kg-adapter' --lr 5e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        #     "--peft_type 'peft_lora' "
        # ),
        #
        # (
        #     "python mymain.py --exp_name 'lora_64_lr5e-5_wu0.1_zephyr-alpha_csqa_v1' --peft_type 'kg-adapter' --lr 5e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_zephyr_v1' --test_data_version 'csqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'csqa_kg+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        #     "--peft_type 'peft_lora' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr3e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+ab_[enc,dec]+d_128' --peft_type 'kg-adapter' --lr 3e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --kd_adapter_hidden_size 128 --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+ab_[enc,dec]+d_128' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --kd_adapter_hidden_size 128 --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr6e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+ab_[enc,dec]+d_128' --peft_type 'kg-adapter' --lr 6e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --kd_adapter_hidden_size 128 --keep_ratio 1.0 "
        # ),

        ################################################ /Ablation Studies/ ################################################

        ################################################# OBQA ###############################################

        # (
        #     "python mymain.py --exp_name 'kg-adapterV3_lr5e-3_wu0.1_zephyr_obqa_v2+SRGAT_[enc,dec]_32' --peft_type 'kg-adapter' --lr 5e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2'  "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV3_lr7e-3_wu0.1_zephyr_obqa_v2+SRGAT_[enc,dec]_32' --peft_type 'kg-adapter' --lr 7e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2'  "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr6e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+V4+no_kg_2node' --peft_type 'kg-adapter' --lr 6e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 --ablation_exp_set no_kg "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr1e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 1e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 --save_top_k 3 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr2e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 2e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 --save_top_k 3 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr3e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 3e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 --save_top_k 3 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr4e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 --save_top_k 3 "
        # ),


        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr6e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 6e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 --save_top_k 3 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr7e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 7e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 --save_top_k 3 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr8e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+V4_r1' --peft_type 'kg-adapter' --lr 8e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 --save_top_k 3 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+V4+train_lm' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 --ablation_exp_set train_lm_head "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV3_lr7e-3_wu0.1_zephyr_obqa_v2+SRGAT_[enc,dec]_32+prefix' --peft_type 'kg-adapter' --lr 7e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev --ablation_exp_set use_prefix "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV3_lr6e-3_wu0.1_zephyr_obqa_v2+SRGAT_[enc,dec]_32+prefix+no_kg_2node' --peft_type 'kg-adapter' --lr 6e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev --ablation_exp_set 'use_prefix+no_kg' "
        # ),
        #
        # (
        #     "python mymain.py --exp_name 'kg-adapterV3_lr8e-3_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+prefix' --peft_type 'kg-adapter' --lr 8e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev --ablation_exp_set use_prefix "
        # ),
        #         (
        #     "python mymain.py --exp_name 'kg-adapterV3_lr9e-3_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+prefix' --peft_type 'kg-adapter' --lr 9e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev --ablation_exp_set use_prefix "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV3_lr7e-3_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+prefix+no_kg_2node' --peft_type 'kg-adapter' --lr 7e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev --ablation_exp_set 'use_prefix+no_kg' "
        # ),
        #
        #
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr6e-4_wu0.1_zephyr_obqa_v2+SRGAT_[enc,dec]_32+no_kg_2node' --peft_type 'kg-adapter' --lr 6e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2'  "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--ablation_exp_set no_kg "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr_obqa_v2+SRGAT_[enc,dec]_32' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2'  "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr7e-4_wu0.1_zephyr_obqa_v2+SRGAT_[enc,dec]_32' --peft_type 'kg-adapter' --lr 7e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2'  "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_zephyr_obqa_v2+SRGAT_[enc,dec]_32+no_kg' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2'  "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--ablation_exp_set no_kg "
        # ),

        # (
        #     "python mymain.py --exp_name 'lora_64_lr1e-5_wu0.1_zephyr-alpha_obqa_v2' --peft_type 'kg-adapter' --lr 1e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v2' --test_data_version 'obqa_zephyr_v2' --eval_data_version 'obqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        #     "--peft_type 'peft_lora' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_mistral-inst_obqa_v2+SRGAT+[enc,dec]_32' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_mistral_v2' --test_data_version 'obqa_mistral_v2' --eval_data_version 'obqa_mistral_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-instruct' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_mistral-inst_obqa_v2+SRGAT+[enc,dec]_32+no_kg_1node' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_mistral_v2' --test_data_version 'obqa_mistral_v2' --eval_data_version 'obqa_mistral_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-instruct' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--ablation_exp_set 'no_kg' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_llama-chat_obqa_v2+SRGAT+[enc,dec]_32' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2_v2' --test_data_version 'obqa_llama2_v2' --eval_data_version 'obqa_llama2_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b-chat' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_llama-chat_obqa_v2+SRGAT+[enc,dec]_32+no_kg_1node' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_llama2_v2' --test_data_version 'obqa_llama2_v2' --eval_data_version 'obqa_llama2_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b-chat' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--ablation_exp_set 'no_kg' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_mistral-base_obqa_v2+SRGAT+[enc,dec]_32' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_mistral_v2' --test_data_version 'obqa_mistral_v2' --eval_data_version 'obqa_mistral_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_mistral_obqa_v2+SRGAT+[enc,dec]_32+no_kg_1node' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_mistral_v2' --test_data_version 'obqa_mistral_v2' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--ablation_exp_set 'no_kg' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter-V2_lr5e-3_wu0.1_zephyr-alpha_obqa_v1+SRGAT+[enc,dec]_32_fr=0' --peft_type 'kg-adapter' --lr 5e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] "
        #     "--dev --ablation_exp_set use_prefix "
        #     "--keep_ratio 1.0 --scaling_rate 1.0 --fuse_rate 0.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter-V2_lr7e-3_wu0.1_zephyr-alpha_obqa_v1+SRGAT+[enc,dec]_32_fr=1' --peft_type 'kg-adapter' --lr 7e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] "
        #     "--dev --ablation_exp_set use_prefix "
        #     "--keep_ratio 1.0 --scaling_rate 1.0 --fuse_rate 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter-V2_lr5e-3_wu0.1_zephyr-alpha_obqa_v1+SRGAT+[dec]_32_fr=0' --peft_type 'kg-adapter' --lr 5e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] "
        #     "--dev --ablation_exp_set use_prefix "
        #     "--keep_ratio 1.0 --scaling_rate 1.0 --fuse_rate 0.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter-V2_lr7e-3_wu0.1_zephyr-alpha_obqa_v1+SRGAT+[dec]_32_fr=1' --peft_type 'kg-adapter' --lr 7e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] "
        #     "--dev --ablation_exp_set use_prefix "
        #     "--keep_ratio 1.0 --scaling_rate 1.0 --fuse_rate 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-6_wu0.1_zephyr-alpha_obqa_v1+SRGAT+[enc,dec]_32+lora-32' --peft_type 'kg-adapter' --lr 1e-6 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--ablation_exp_set 'add_lora' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter-V2_lr7e-6_wu0.1_zephyr-alpha_obqa_v1+SRGAT+[dec]_32_fr=0+lora-32' --peft_type 'kg-adapter' --lr 7e-6 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] "
        #     "--dev --ablation_exp_set use_prefix "
        #     "--keep_ratio 1.0 --scaling_rate 1.0 --fuse_rate 0.0 "
        #     "--ablation_exp_set 'add_lora' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter-V2_lr1e-5_wu0.1_zephyr-alpha_obqa_v1+SRGAT+[dec]_32_fr=1+lora-32' --peft_type 'kg-adapter' --lr 1e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] "
        #     "--dev --ablation_exp_set use_prefix "
        #     "--keep_ratio 1.0 --scaling_rate 1.0 --fuse_rate 1.0 "
        #     "--ablation_exp_set 'add_lora' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+[dec]_32' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+[dec]_30' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [2,32] --keep_ratio 1.0 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+[enc,dec]_32+fuse_rate=0.5' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--fuse_rate 0.5 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+[enc,dec]_32+rand_init' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--ablation_exp_set 'rand_init' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-3_wu0.1_zephyr-alpha_obqa_v1+SRGAT+[enc,dec]_32+rand_init' --peft_type 'kg-adapter' --lr 1e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--ablation_exp_set 'rand_init' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+[enc,dec]_32+rand_init' --peft_type 'kg-adapter' --lr 1e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--ablation_exp_set 'rand_init' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+[enc,dec]_30' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [2,32,2] --kg_adapter_dec_range [3,32,2] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+[enc,dec]_32' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr6e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+[enc,dec]_32' --peft_type 'kg-adapter' --lr 6e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+[enc,dec]_30' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [2,32,2] --kg_adapter_dec_range [3,32,2] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr6e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+[enc,dec]_30' --peft_type 'kg-adapter' --lr 6e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [2,32,2] --kg_adapter_dec_range [3,32,2] --keep_ratio 1.0 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_obqa_v1+SRGAT+[enc,dec]_32+kr=0.99' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'obqa_zephyr_v1' --test_data_version 'obqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'obqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 0.99 "
        # ),
        ######################################## / OBQA / #######################################################

        ######################################## CSQA ###########################################################
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_llama-chat_obqa_v1+SRGAT+ab_[enc,dec]_32' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_llama2_v1' --test_data_version 'csqa_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b-chat' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'csqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV3_lr9e-3_wu0.1_zephyr_csqa_v2+SRGAT_[enc,dec]_32+prefix' --peft_type 'kg-adapter' --lr 9e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_zephyr_v2' --test_data_version 'csqa_zephyr_v2' --eval_data_version 'csqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev --ablation_exp_set use_prefix "
        # ),
        #
        # (
        #     "python mymain.py --exp_name 'kg-adapterV3_lr1e-3_wu0.1_zephyr_csqa_v2+SRGAT_[dec]_32+prefix' --peft_type 'kg-adapter' --lr 1e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_zephyr_v2' --test_data_version 'csqa_zephyr_v2' --eval_data_version 'csqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev --ablation_exp_set use_prefix "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_csqa_v2+SRGAT_[dec]_32+V4+no_kg_2node' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_zephyr_v2' --test_data_version 'csqa_zephyr_v2' --eval_data_version 'csqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 --ablation_exp_set no_kg"
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapterV4_lr5e-4_wu0.1_zephyr_csqa_v2+SRGAT_[dec]_32+V4+train_lm' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_zephyr_v2' --test_data_version 'csqa_zephyr_v2' --eval_data_version 'csqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [0,32] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--dev2 --ablation_exp_set train_lm_head "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr6e-4_wu0.1_zephyr_csqa_v2+SRGAT_[enc,dec]_32' --peft_type 'kg-adapter' --lr 6e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_zephyr_v2' --test_data_version 'csqa_zephyr_v2' --eval_data_version 'csqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-4_wu0.1_zephyr_csqa_v2+SRGAT_[enc,dec]_32' --peft_type 'kg-adapter' --lr 1e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_zephyr_v2' --test_data_version 'csqa_zephyr_v2' --eval_data_version 'csqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-3_wu0.1_zephyr_csqa_v2+SRGAT_[enc,dec]_32' --peft_type 'kg-adapter' --lr 1e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_zephyr_v2' --test_data_version 'csqa_zephyr_v2' --eval_data_version 'csqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_zephyr_csqa_v2+SRGAT_[enc,dec]_32+no_kg' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_zephyr_v2' --test_data_version 'csqa_zephyr_v2' --eval_data_version 'csqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38 "
        #     "--ablation_exp_set no_kg "
        # ),
        # (
        #     "python mymain.py --exp_name 'lora_64_lr5e-4_wu0.1_zephyr-alpha_csqa_v2' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_zephyr_v2' --test_data_version 'csqa_zephyr_v2' --eval_data_version 'csqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        #     "--peft_type 'peft_lora' "
        # ),
        # (
        #     "python mymain.py --exp_name 'lora_64_lr5e-5_wu0.1_zephyr-alpha_csqa_v2' --peft_type 'kg-adapter' --lr 5e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_zephyr_v2' --test_data_version 'csqa_zephyr_v2' --eval_data_version 'csqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        #     "--peft_type 'peft_lora' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-4_wu0.1_zephyr_csqa_v2+SRGAT_[enc,dec]_32' --peft_type 'kg-adapter' --lr 1e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_zephyr_v2' --test_data_version 'csqa_zephyr_v2' --eval_data_version 'csqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/obqa+csqa_v2_(34908,1024)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'csqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 1024 --num_relations 38"
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_mistral_csqa_v1+SRGAT+ab_[enc,dec]_32' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_mistral_v1' --test_data_version 'csqa_mistral_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'csqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_mistral_csqa_v1+SRGAT+ab_[enc,dec]+no_kg' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_mistral_v1' --test_data_version 'csqa_mistral_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'csqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--ablation_exp_set 'no_kg' "
        # ),
        #
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-4_wu0.1_mistral_csqa_v1+SRGAT+ab_[enc,dec]_32' --peft_type 'kg-adapter' --lr 1e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_mistral_v1' --test_data_version 'csqa_mistral_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'csqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-3_wu0.1_mistral_csqa_v1+SRGAT+ab_[enc,dec]_32' --peft_type 'kg-adapter' --lr 1e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_mistral_v1' --test_data_version 'csqa_mistral_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'csqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_csqa_v1+SRGAT+[enc,dec]_32' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'csqa_zephyr_v1' --test_data_version 'csqa_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'csqa_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),

        ##################################### /CSQA/ ##########################################################

        ################################################ MedQA ################################################
        # (
        #     "python mymain.py --exp_name 'kg-adapterV3_lr1e-3_wu0.1_zephyr_medqa_v2+SRGAT_[enc,dec]_32' --peft_type 'kg-adapter' --lr 1e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'medqa_zephyr_v2' --test_data_version 'medqa_zephyr_v2' --eval_data_version 'medqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/MedQA_(3448,768)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'medqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 768 --num_relations 38 "
        #     "--dev "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV3_lr5e-4_wu0.1_zephyr_medqa_v2+SRGAT_[enc,dec]_32' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'medqa_zephyr_v2' --test_data_version 'medqa_zephyr_v2' --eval_data_version 'medqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/MedQA_(3448,768)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'medqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 768 --num_relations 38 "
        #     "--dev "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV3_lr1e-4_wu0.1_zephyr_medqa_v2+SRGAT_[enc,dec]_32' --peft_type 'kg-adapter' --lr 1e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'medqa_zephyr_v2' --test_data_version 'medqa_zephyr_v2' --eval_data_version 'medqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/MedQA_(3448,768)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'medqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 768 --num_relations 38 "
        #     "--dev "
        # ),
        #
        # (
        #     "python mymain.py --exp_name 'kg-adapterV3_lr1e-3_wu0.1_zephyr_medqa_v2+SRGAT_[enc,dec]_32+prefix' --peft_type 'kg-adapter' --lr 1e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'medqa_zephyr_v2' --test_data_version 'medqa_zephyr_v2' --eval_data_version 'medqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/MedQA_(3448,768)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'medqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 768 --num_relations 38 "
        #     "--dev --ablation_exp_set use_prefix "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV3_lr5e-4_wu0.1_zephyr_medqa_v2+SRGAT_[enc,dec]_32+prefix' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'medqa_zephyr_v2' --test_data_version 'medqa_zephyr_v2' --eval_data_version 'medqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/MedQA_(3448,768)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'medqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 768 --num_relations 38 "
        #     "--dev --ablation_exp_set use_prefix "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV3_lr1e-4_wu0.1_zephyr_medqa_v2+SRGAT_[enc,dec]_32+prefix' --peft_type 'kg-adapter' --lr 1e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'medqa_zephyr_v2' --test_data_version 'medqa_zephyr_v2' --eval_data_version 'medqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/MedQA_(3448,768)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'medqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 768 --num_relations 38 "
        #     "--dev --ablation_exp_set use_prefix "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapterV3_lr5e-5_wu0.1_zephyr_medqa_v2+SRGAT_[enc,dec]_32+prefix' --peft_type 'kg-adapter' --lr 5e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'medqa_zephyr_v2' --test_data_version 'medqa_zephyr_v2' --eval_data_version 'medqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/MedQA_(3448,768)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'medqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 768 --num_relations 38 "
        #     "--dev --ablation_exp_set use_prefix "
        # ),
        #
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_zephyr_medqa_v2+SRGAT_[enc,dec]_32' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'medqa_zephyr_v2' --test_data_version 'medqa_zephyr_v2' --eval_data_version 'medqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/MedQA_(3448,768)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'medqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 768 --num_relations 38"
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-4_wu0.1_zephyr_medqa_v2+SRGAT_[enc,dec]_32' --peft_type 'kg-adapter' --lr 1e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'medqa_zephyr_v2' --test_data_version 'medqa_zephyr_v2' --eval_data_version 'medqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/MedQA_(3448,768)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'medqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 768 --num_relations 38"
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-5_wu0.1_zephyr_medqa_v2+SRGAT_[enc,dec]_32' --peft_type 'kg-adapter' --lr 5e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'medqa_zephyr_v2' --test_data_version 'medqa_zephyr_v2' --eval_data_version 'medqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/MedQA_(3448,768)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'medqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 768 --num_relations 38"
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-3_wu0.1_zephyr_medqa_v2+SRGAT_[enc,dec]_32' --peft_type 'kg-adapter' --lr 1e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'medqa_zephyr_v2' --test_data_version 'medqa_zephyr_v2' --eval_data_version 'medqa_zephyr_v2' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/MedQA_(3448,768)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/all_test_3_v2.csv' "
        #     "--test_set 'medqa+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--kg_adapter_node_emb_size 768 --num_relations 38"
        # ),
        ################################################ /MedQA/ ################################################

        ################################################ WQSP ################################################
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr2e-4_wu0.1_zephyr-alpha_wqsp_v1+SRGAT+[enc,dec]_32' --peft_type 'kg-adapter' --lr 2e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v1' --test_data_version 'wqsp_zephyr_v1' --eval_data_version '0' --num_relations 805 "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),
        #
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr2e-4_wu0.1_zephyr-alpha_wqsp_v1+SRGAT+[enc,dec]_32' --peft_type 'kg-adapter' --lr 2e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v1' --test_data_version 'wqsp_zephyr_v1' --eval_data_version '0' --num_relations 805 "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_mistral_wqsp_v1+SRGAT+[enc,dec]_32' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_mistral_v1' --test_data_version 'wqsp_mistral_v1' --eval_data_version '0' --num_relations 805 "
        #     "--pretrained_path '/raid_sdb/LLMs/Mistral-7B-v0.1' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr2e-4_wu0.1_zephyr-alpha_wqsp_v1+SRGAT+[dec]_30' --peft_type 'kg-adapter' --lr 2e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v1' --test_data_version 'wqsp_zephyr_v1' --eval_data_version '0' --num_relations 805 "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [2,32] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr7e-4_wu0.1_zephyr-alpha_wqsp_v1+SRGAT+[dec]_30' --peft_type 'kg-adapter' --lr 7e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v1' --test_data_version 'wqsp_zephyr_v1' --eval_data_version '0' --num_relations 805 "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [2,32] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-3_wu0.1_zephyr-alpha_wqsp_v1+SRGAT+[dec]_30' --peft_type 'kg-adapter' --lr 1e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v1' --test_data_version 'wqsp_zephyr_v1' --eval_data_version '0' --num_relations 805 "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [2,32] --keep_ratio 1.0 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-3_wu0.1_zephyr-alpha_wqsp_v1+SRGAT+[enc,dec]_32+no_kg' --peft_type 'kg-adapter' --lr 1e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v1' --test_data_version 'wqsp_zephyr_v1' --eval_data_version '0' --num_relations 805 "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--ablation_exp_set 'no_kg' "
        # ),
        # (
        #     "python mymain.py --exp_name 'lora_64_lr5e-5_wu0.1_zephyr-alpha_wqsp_v1' --peft_type 'kg-adapter' --lr 5e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v1' --test_data_version 'wqsp_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        #     "--peft_type 'peft_lora' "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr2e-4_wu0.1_llama2-base_wqsp_v1+SRGAT+[dec]_30' --peft_type 'kg-adapter' --lr 2e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_llama2_v1' --test_data_version 'wqsp_llama2_v1' --eval_data_version '0' --num_relations 805 "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,0] --kg_adapter_dec_range [2,32] --keep_ratio 1.0 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-4_wu0.1_llama2-base_wqsp_v1+SRGAT+[enc,dec]_32' --peft_type 'kg-adapter' --lr 1e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_llama2_v1' --test_data_version 'wqsp_llama2_v1' --eval_data_version '0' --num_relations 805 "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-3_wu0.1_llama2-base_wqsp_v1+SRGAT+[enc,dec]_32+no_kg' --peft_type 'kg-adapter' --lr 1e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_llama2_v1' --test_data_version 'wqsp_llama2_v1' --eval_data_version '0' --num_relations 805 "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        #     "--ablation_exp_set 'no_kg' "
        # ),
        # (
        #     "python mymain.py --exp_name 'lora_64_lr5e-5_wu0.1_llama2_wqsp_v1' --peft_type 'kg-adapter' --lr 5e-5 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_llama2_v1' --test_data_version 'wqsp_llama2_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/llama-2-7b' --exp_set 'loss_only_on_ans' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens+no_kg' --monitor 'val_em' "
        #     "--peft_type 'peft_lora' "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr2e-3_wu0.1_zephyr-alpha_wqsp_v1+SRGAT+[enc,dec]_32' --peft_type 'kg-adapter' --lr 2e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v1' --test_data_version 'wqsp_zephyr_v1' --eval_data_version '0' --num_relations 805 "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-3_wu0.1_zephyr-alpha_wqsp_v1+SRGAT+[enc,dec]_32' --peft_type 'kg-adapter' --lr 5e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'wqsp_zephyr_v1' --test_data_version 'wqsp_zephyr_v1' --eval_data_version '0' --num_relations 805 "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'wqsp+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),

        #################################### /WQSP/ ##############################################################

        #################################### CWQ ###############################################################
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr1e-3_wu0.1_zephyr-alpha_cwq_v1+SRGAT+[enc,dec]_32' --peft_type 'kg-adapter' --lr 1e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'cwq_zephyr_v1' --test_data_version 'cwq_zephyr_v1' --eval_data_version '0' --num_relations 805 "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'cwq+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr2e-3_wu0.1_zephyr-alpha_cwq_v1+SRGAT+[enc,dec]_32' --peft_type 'kg-adapter' --lr 2e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'cwq_zephyr_v1' --test_data_version 'cwq_zephyr_v1' --eval_data_version '0' --num_relations 805 "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'cwq+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-3_wu0.1_zephyr-alpha_cwq_v1+SRGAT+[enc,dec]_32' --peft_type 'kg-adapter' --lr 5e-3 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'cwq_zephyr_v1' --test_data_version 'cwq_zephyr_v1' --eval_data_version '0' --num_relations 805 "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT+no_node_emb' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'cwq+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr4e-4_wu0.1_zephyr-alpha_cwq_v1+SRGAT+[enc,dec]_32' --peft_type 'kg-adapter' --lr 4e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'cwq_zephyr_v1' --test_data_version 'cwq_zephyr_v1' --eval_data_version '0' --num_relations 805 "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT+no_node_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'cwq+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),
        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr7e-4_wu0.1_zephyr-alpha_cwq_v1+SRGAT+[enc,dec]_32' --peft_type 'kg-adapter' --lr 7e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'cwq_zephyr_v1' --test_data_version 'cwq_zephyr_v1' --eval_data_version '0' --num_relations 805 "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+use_cat_trips+use_SRGAT+no_node_emb' "
        #     "--node_emb_path '/raid_sdb/home/tsy/KG_data/KG_emb/CSKG_TransE_4+6_data_(81590,100)_nodes_emb.pt' "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'cwq+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,32,2] --kg_adapter_dec_range [1,32,2] --keep_ratio 1.0 "
        # ),

        # (
        #     "python mymain.py --exp_name 'kg-adapter_lr5e-4_wu0.1_zephyr-alpha_cwq_v1+SRGAT+ec2/30' --peft_type 'kg-adapter' --lr 5e-4 --warm_up_epoch 0.1 --strategy 'deepspeed' "
        #     "--train_data_version 'cwq_zephyr_v1' --test_data_version 'cwq_zephyr_v1' --eval_data_version '0' "
        #     "--pretrained_path '/raid_sdb/LLMs/zephyr-alpha' --exp_set 'loss_only_on_ans+no_share_ca+use_edge_emb+mix_emb+no_node_emb+use_cat_trips+use_SRGAT' --num_relations 805 "
        #     "--test_data_path '/raid_sdb/home/tsy/KG_data/csv_test_data/all_test_6.csv' "
        #     "--test_set 'cwq_kg+no_user_inst+task_system_inst+add_special_tokens' --monitor 'val_em' "
        #     "--kg_adapter_enc_range [0,2] --kg_adapter_dec_range [2,32] --keep_ratio 1.0 "
        # ),

        ################################################ /CWQ/ ################################################

    ]

    GPU_USAGE_FOR_DATASET = {"13b": 40000, "--eval ": 30000, "3b": 20000, "csqa": 24000, "obqa": 24000,
                             "max_seq_length 512": 30000, "max_seq_length 1024": 30000,
                             'wqsp': 35000, 'cwq': 25000}
    need_gpu_memory = 30000
    threads = []
    cmd_screen = [f"exp_{x}" for x in range(len(cmds))]
    print("begin run all cmds %s" % datetime.datetime.now())
    while (len(finished_id) != len(cmds)):
        for idx, cmd in enumerate(cmds):
            if idx in finished_id or idx in running_id:
                continue
            print("waiting gpus for cmd: ", idx, "||| ", cmd)
            print("now running cmd idx: ", running_id, "|||", "now waiting cmd num",
                  len(cmds) - len(finished_id) - len(running_id))
            for k, v in GPU_USAGE_FOR_DATASET.items():
                if k in cmd:
                    need_gpu_memory = v
                    break
            gpu_ids = waiting_gpu(need_memory=need_gpu_memory)
            th = threading.Thread(target=ExecCmd, args=(idx, cmd, gpu_ids,))
            th.start()
            threads.append(th)
            time.sleep(60 * 5)

    # waiting all threads over
    for th in threads:
        th.join()

    print("all cmds over %s" % datetime.datetime.now())

import os
import sys
import time
import datetime
import threading


def gpu_info(gpu_index):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('\n')[gpu_index].split('|')
    power = int(gpu_status[1].split()[-3][:-1])
    memory = int(gpu_status[2].split('/')[0].strip()[:-3])
    return power, memory


def waiting_gpu(interval=2):
    need_memory_lst = [15000]
    num_lst = [1]
    gid = [0, 1, 2, 3]
    while True:
        for memory, num in zip(need_memory_lst, num_lst):
            candid_gid_lst = []
            for gpu_id in gid:
                gpu_power, gpu_memory = gpu_info(gpu_id)

                if gpu_memory <= memory and check_used_gpu_num():
                    if gpu_id not in candid_gid_lst:
                        candid_gid_lst.append((gpu_memory, gpu_id))

                if gpu_memory > memory and gpu_id in candid_gid_lst:
                    candid_gid_lst.remove((gpu_memory, gpu_id))

                gpu = 'gpu id:%d' % gpu_id
                gpu_power_str = 'gpu power:%d W |' % gpu_power
                gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
                gpu_select_rule = 'memory=%d ; num=%d |' % (memory, num)
                sys.stdout.write('\r' + gpu + ' ' + gpu_memory_str + ' ' + gpu_power_str + gpu_select_rule + " candid_gid_lst:" + str(candid_gid_lst))
                sys.stdout.flush()

            if len(candid_gid_lst) >= num:
                candid_gid_lst.sort()
                candid_gid_lst = [x[1] for x in candid_gid_lst[:num]]
                return candid_gid_lst

            time.sleep(interval)


def check_used_gpu_num():
    global running_id
    now_running_num = len(running_id)
    if datetime.datetime.now().hour >= 23 or datetime.datetime.now().hour <= 8:
        return True
    elif now_running_num < 1:
        return True
    else:
        return False


def ExecCmd(idx, cmd, gpu_ids):
    global running_id
    global finished_id
    try:
        print("cmd %s start %s" % (cmd, datetime.datetime.now()), "\n")
        running_id.append(idx)
        if "devices" not in cmd:
            cmd += " --devices '%s'" % str(gpu_ids)
        print(cmd)
        res = os.system(cmd)
        if res == 0:
            print("cmd return code:",  res)
            print("cmd %s over %s" % (cmd, datetime.datetime.now()), "\n")
            running_id.remove(idx)
            finished_id.append(idx)
        else:
            running_id.remove(idx)
    except:
        print('fail at:', idx)
        running_id.remove(idx)


global running_id
global finished_id

if __name__ == '__main__':
    running_id = []
    finished_id = []
    #     cmd = "CUDA_VISIBLE_DEVICES=%d .........." % gpu_id
    cmds = [
        "python generate.py --batch_size 1 --exp_name 'peft_llama-adapter_lr9e-3_wu2_DS2_pad-left' --ckpt_ep 0 --suffix 'Your Answer:' ",
        "python generate.py --batch_size 1 --exp_name 'peft_llama-adapter_lr9e-3_wu2_DS2_pad_train_right_gen_left' --ckpt_ep 1 --suffix 'Your Answer:' ",

        "python generate.py --batch_size 1 --exp_name 'peft_llama-adapter_lr9e-3_wu2_DS2_pad_train_right_gen_left' --ckpt_ep 1 --suffix 'Answer:' ",
    ]

    threads = []

    print("begin run all cmds %s" % datetime.datetime.now())
    while(len(finished_id)!=len(cmds)):
        for idx, cmd in enumerate(cmds):
            if idx in finished_id or idx in running_id:
                continue
            print("waiting gpus for cmd: ", idx, "||| ", cmd)
            print("now running program idx: ", running_id)
            gpu_ids = waiting_gpu()
            print("gpu_ids:", str(gpu_ids))
            th = threading.Thread(target=ExecCmd, args=(idx, cmd, gpu_ids,))
            th.start()
            threads.append(th)
            time.sleep(180)

    # waiting all threads over
    for th in threads:
        th.join()

    print("all cmds over %s" % datetime.datetime.now())

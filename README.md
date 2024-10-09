# KG-Adapter
Code for the paper "KG-Adapter: Enabling Knowledge Graph Integration in Large Language Models through Parameter-Efficient Fine-Tuning"
Accepted by "ACL 2024 Findings"
![Model_v2](https://github.com/Ogmx/KG-Adapter/assets/37243586/daf63dc3-5c7c-431d-9187-e71892cbd325)

# Updare V1:
* add code and data for OBQA dataset
* Note: The current version is the original unorganized code, there are some redundant information, it may be difficult to run directly, please refer to the code mainly.
  
---

# How to use:
* Install all required libraries
* Download the data and ckpt files and place them in the root directory: [google drive]([https://www.baidu.com](https://drive.google.com/drive/folders/15MNxrVev-2YXd6BYv_ngpe-729gq5wmX?usp=drive_link))
* python auto_run.py

# File Structure
```
│  auto_error_log.txt
│  auto_run.py
│  auto_run_log.txt
│  eval_old.py
│  mydata.py
│  myeval.py
│  mymain.py
│  mymodel.py
│  utils.py
│  __init__.py
│  
├─ckpt: put checkpoints here
│  └─kg-adapterV4_lr5e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+V4_r1
│          peft_ckpt_epoch=3-step=312.bin
│          
├─data: put all data and KG embedding here
│  │  all_test_3_v2.csv
│  │  dev_obqa_zephyr_v2.pt
│  │  test_obqa_zephyr_v2.pt
│  │  train_obqa_zephyr_v2.pt
│  │  
│  └─KG_emb
│          obqa+csqa_v2_(34908,1024)_nodes_emb.pt
│          
├─eval
│  └─.......
│          
├─LLMs: put LLMs here
│  └─zephyr-alpha
│
│
├─model: KG-Adapter model for different base LLMs
│      llama_v3.py
│      mistral_v3.py
│      
├─outputs
│  └─kg-adapterV4_lr5e-4_wu0.1_zephyr_obqa_v2+SRGAT_[dec]_32+V4_r1
```


        

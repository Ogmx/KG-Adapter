# KG-Adapter
Code for the paper "KG-Adapter: Enabling Knowledge Graph Integration in Large Language Models through Parameter-Efficient Fine-Tuning"

Accepted by "ACL 2024 Findings"
![Model_v2](https://github.com/Ogmx/KG-Adapter/assets/37243586/daf63dc3-5c7c-431d-9187-e71892cbd325)

# Update V1:
* add code and data for OBQA dataset
* **Note: The current version is the original unorganized code, there are some redundant information, it may be difficult to run directly, please refer to the code mainly.**
  
---

# How to use:
* Install all required libraries
* Download the data and ckpt files and place them in the root directory: [google drive](https://drive.google.com/drive/folders/15MNxrVev-2YXd6BYv_ngpe-729gq5wmX?usp=drive_link)
* python auto_run.py  (it will automatically create a screen and run the command)
# File Structure
```
│  auto_error_log.txt
│  auto_run.py
│  auto_run_log.txt
│  eval_old.py
│  mydata.py:  data module of PyTorch Lightning 
│  mymain.py
│  mymodel.py: model module of PyTorch Lightning
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

# Cite
```
@inproceedings{tian-etal-2024-kg,
    title = "{KG}-Adapter: Enabling Knowledge Graph Integration in Large Language Models through Parameter-Efficient Fine-Tuning",
    author = "Tian, Shiyu  and
      Luo, Yangyang  and
      Xu, Tianze  and
      Yuan, Caixia  and
      Jiang, Huixing  and
      Wei, Chen  and
      Wang, Xiaojie",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.229",
    doi = "10.18653/v1/2024.findings-acl.229",
    pages = "3813--3828",
    abstract = "Although large language models (LLMs) show remarkable capabilities and generalizability across various tasks, they are criticized for lack of expertise. One promising solution is to combine knowledge graphs (KGs) with LLMs, and recent studies focus on integrating KGs into LLMs through prompt-based methods. However, these approaches fail to use the structural information of the KGs, suffer from the problem of knowledge conflict, and over-reliance on super LLMs. To address these challenges, we propose KG-Adapter, a parameter-level KG integration method based on parameter-efficient fine-tuning (PEFT). Specifically, we introduce a novel adapter structure designed for decoder-only LLMs, which can encode KGs from both node-centered and relation-centered perspectives, and then perform joint reasoning with LLMs to generate responses end-to-end. Experiments with diverse models on four datasets for two different tasks all demonstrate significant improvements. With only 28M parameters trained, we make the 7B-parameter LLM outperform the previous full-parameter fine-tuned state-of-the-art method and comparable to the prompt-based ChatGPT methods.",
}
```

import torch
import os
import numpy as np
import pickle as pkl
import torch.utils.data as data
import pandas as pd
import lightning as L
from torch.utils.data import DataLoader
from torch_geometric.utils import unbatch_edge_index, to_dense_batch, subgraph, erdos_renyi_graph


# from torch_geometric.loader import DataLoader


class MyDataModule(L.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

    def prepare_data(self):
        return

    def setup(self, stage):
        if self.args.peft_type == "kg-adapter":
            self.train_set = KgAdapterDataset(self.args, "train", tokenizer=self.tokenizer)
            self.val_set = KgAdapterDataset(self.args, "test", tokenizer=self.tokenizer)
            self.test_set = KgAdapterDataset(self.args, "test", tokenizer=self.tokenizer)
        else:
            self.train_set = KgAdapterDataset(self.args, "train", tokenizer=self.tokenizer)
            self.val_set = KgAdapterDataset(self.args, "test", tokenizer=self.tokenizer)
            self.test_set = KgAdapterDataset(self.args, "test", tokenizer=self.tokenizer)
            # self.train_set = SFTDataset(self.args.data_dir, file_name="train.pt")
            # self.val_set = SFTDataset(self.args.data_dir, file_name="val.pt")
            # self.test_set = SFTDataset(self.args.data_dir, file_name="test.csv", tokenizer=self.tokenizer)

    def train_dataloader(self):
        # !! Note: use pad_right when training and use pad_left when generating
        # more detail can be found at https://github.com/huggingface/transformers/issues/3021
        if self.args.peft_type == "kg-adapter":
            train_loader = DataLoader(
                self.train_set, batch_size=self.args.micro_batch_size, shuffle=True, num_workers=self.args.num_workers,
                collate_fn=lambda x: kg_adapter_right_pad_collate_fn(x, self.args),
            )
        else:
            train_loader = DataLoader(
                self.train_set, batch_size=self.args.micro_batch_size, shuffle=True, num_workers=self.args.num_workers,
                collate_fn=right_pad_collate_fn,
            )
        return train_loader

    def val_dataloader(self):
        # !! Note: use pad_right when training and use pad_left when generating
        if self.args.peft_type == "kg-adapter":
            val_loader = DataLoader(
                self.val_set, batch_size=self.args.micro_batch_size * 8, shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=lambda x: kg_adapter_left_pad_collate_fn(x, self.args),
            )
        else:
            val_loader = DataLoader(
                self.val_set, batch_size=self.args.micro_batch_size * 8, shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=left_pad_collate_fn,
            )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_set, batch_size=self.args.micro_batch_size, shuffle=False, num_workers=self.args.num_workers,
            persistent_workers=True,
        )
        return test_loader


class SFTDataset(data.Dataset):
    def __init__(self, path, file_name, tokenizer=None, add_text=None, debug=False):
        if '.csv' in file_name:
            self.data = pd.read_csv(path + '/' + file_name, index_col=0)
            self.data_type = "csv"
            self.add_text = add_text
            self.tokenizer = tokenizer
        else:
            self.data = torch.load(os.path.join(path, file_name))
            self.data_type = "pt"
        if debug:
            self.data = self.data[:16]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data_type == "pt":
            idx = torch.tensor(idx).type(torch.int64)
            input_ids = self.data[idx]["input_ids"].type(torch.int64)
            labels = self.data[idx]["labels"].type(torch.int64)
            prompt_len = torch.tensor(len(self.data[idx]["input_ids_no_response"])).type(torch.int64)
            return idx, input_ids, labels, prompt_len

        elif self.data_type == "csv":  # used in test stage
            input_text = self.data.iloc[idx]['prompt']
            if self.add_text:
                prefix = self.add_text[0] + '\n' if self.add_text[0] != '' else ""
                suffix = '\n' + self.add_text[1] if self.add_text[1] != '' else ""
                input_text = prefix + input_text + suffix
            input_text_len = torch.tensor(len(input_text))
            tokenizer_output = self.tokenizer(input_text, padding='max_length', max_length=2048, return_tensors='pt')
            input_ids = tokenizer_output['input_ids'].squeeze()
            input_mask = tokenizer_output['attention_mask'].squeeze()

            return input_ids, input_mask, input_text_len

        else:
            assert "unavailable data type"


class KgAdapterDataset(data.Dataset):
    def __init__(self, args, stage, tokenizer=None):
        # kg_emb = args.kg_emb
        self.args = args
        self.exp_set = args.exp_set
        self.max_seq_length = args.max_seq_length
        self.tokenizer = tokenizer
        if stage == "train" and os.path.exists(f"{args.data_path}/{stage}_{args.train_data_version}.pt"):
            print(f"loading {stage} data.....")
            self.data = torch.load(
                f"{args.data_path}/{stage}_{args.train_data_version}.pt")
            self.data_type = 'pt'
        elif stage == "test" and os.path.exists(f"{args.data_path}/{stage}_{args.test_data_version}.pt"):
            print(f"loading {stage} data.....")
            self.data = torch.load(
                f"{args.data_path}/{stage}_{args.test_data_version}.pt")

            if args.eval_data_version is not None and os.path.exists(
                    f"{args.data_path}/dev_{args.eval_data_version}.pt"):
                print("loading dev data ....")
                self.data_dev = torch.load(
                    f"{args.data_path}/dev_{args.eval_data_version}.pt")

                for x in self.data:
                    x['idx'] += len(self.data_dev)

                self.data = np.concatenate((self.data_dev, self.data))

            self.data_type = 'pt'
        else:
            assert "unavailable data"
        # TODO: dynamic loading data
        # else:
        #     assert "unavailable data"
        #     if os.path.exists(f"{text_path}/{args.data_name}_{stage}.pt"):
        #         self.text_data = torch.load(f"{text_path}/{args.data_name}_{stage}.pt")
        #         self.data_type = "pt"
        #     elif os.path.exists(f"{text_path}/{args.data_name}_{stage}.csv"):
        #         self.text_data = pd.read_csv(f"{text_path}/{args.data_name}_{stage}.csv", index_col=0)
        #         self.data_type = "csv"
        #     else:
        #         assert "not find data"
        #
        #     if 'OBQA' in args.data_name and 'CSQA' in args.data_name:
        #         tmp1 = torch.load(f"{kg_path}/OBQA_{stage}_{kg_emb}_pyg.pt")
        #         tmp2 = torch.load(f"{kg_path}/CSQA_{stage}_{kg_emb}_pyg.pt")
        #         self.kg_data = tmp1[:-1] + tmp2[:-1] + tmp1[-1].append(tmp2[-1])
        #     elif os.path.exists(f"{kg_path}/{args.data_name}_{stage}_{kg_emb}_pyg.pt"):
        #         self.kg_data = torch.load(f"{kg_path}/{args.data_name}_{stage}_{kg_emb}_pyg.pt")
        #     else:
        #         assert "not find kg data"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data_type == "pt":
            idx = torch.tensor(idx).type(torch.int64)
            input_ids = self.data[idx]["input_ids"].type(torch.int64)
            if "loss_only_on_ans" in self.exp_set:
                labels = self.data[idx]["labels"].type(torch.int64)
            else:
                labels = input_ids.clone()
            if len(input_ids) > self.max_seq_length:
                input_ids = input_ids[-self.max_seq_length:]
                labels = labels[-self.max_seq_length:]

            prompt_len = torch.tensor(len(self.data[idx]["input_ids_no_response"])).type(torch.int64)
            if self.args.peft_type != "kg-adapter":
                sg = None
            else:
                sg = self.data[idx]['sg']

            return idx, input_ids, labels, prompt_len, sg

        # elif self.data_type == "csv":  # used in test stage
        #     input_text = self.data.iloc[idx]['prompt']
        #     if self.add_text:
        #         prefix = self.add_text[0] + '\n' if self.add_text[0] != '' else ""
        #         suffix = '\n' + self.add_text[1] if self.add_text[1] != '' else ""
        #         input_text = prefix + input_text + suffix
        #     input_text_len = torch.tensor(len(input_text))
        #     tokenizer_output = self.tokenizer(input_text, padding='max_length', max_length=2048, return_tensors='pt')
        #     input_ids = tokenizer_output['input_ids'].squeeze()
        #     input_mask = tokenizer_output['attention_mask'].squeeze()

            return input_ids, input_mask, input_text_len

        else:
            assert "unavailable data type"


def right_pad_collate_fn(data):
    idx = [data[i][0].type(torch.int64) for i in range(len(data))]
    input_ids = [data[i][1].type(torch.int64) for i in range(len(data))]
    labels = [data[i][2].type(torch.int64) for i in range(len(data))]
    prompt_len = [data[i][3].type(torch.int64) for i in range(len(data))]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    def build_mask_right(x):
        mask = [1] * len(x) + [0] * (max_len - len(x))
        return torch.tensor(mask)

    x_no_res = torch.stack([pad_right(input_ids[i][:prompt_len[i]], pad_id=0) for i in range(len(input_ids))])
    x_no_res_mask = torch.stack([build_mask_right(input_ids[i][:prompt_len[i]]) for i in range(len(input_ids))])

    mask = torch.stack([build_mask_right(x) for x in input_ids])
    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-100) for x in labels])

    prompt_len = torch.stack(prompt_len)
    idx = torch.stack(idx)

    return idx, x, y, mask, prompt_len, x_no_res, x_no_res_mask


def left_pad_collate_fn(data):
    idx = [data[i][0].type(torch.int64) for i in range(len(data))]
    input_ids = [data[i][1].type(torch.int64) for i in range(len(data))]
    labels = [data[i][2].type(torch.int64) for i in range(len(data))]
    prompt_len = [data[i][3].type(torch.int64) for i in range(len(data))]

    max_len = max(len(s) for s in input_ids)

    def pad_left(x, pad_id):
        # pad left based on the longest sequence
        n = max_len - len(x)
        return torch.cat((torch.full((n,), pad_id, dtype=x.dtype), x))

    def build_mask_left(x):
        mask = [0] * (max_len - len(x)) + [1] * len(x)
        return torch.tensor(mask)

    x_no_res = torch.stack([pad_left(input_ids[i][:prompt_len[i]], pad_id=0) for i in range(len(input_ids))])
    x_no_res_mask = torch.stack([build_mask_left(input_ids[i][:prompt_len[i]]) for i in range(len(input_ids))])

    mask = torch.stack([build_mask_left(x) for x in input_ids])
    x = torch.stack([pad_left(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_left(x, pad_id=-100) for x in labels])

    prompt_len = torch.stack(prompt_len)
    idx = torch.stack(idx)

    return idx, x, y, mask, prompt_len, x_no_res, x_no_res_mask


def build_full_pad_graph(num_nodes=2, edge_prob=1):
    from torch_geometric.data import Data
    rand_sg = Data(x=torch.zeros(num_nodes, dtype=torch.long),
                   edge_index=erdos_renyi_graph(num_nodes=num_nodes, edge_prob=edge_prob, directed=True))
    rand_sg.edge_type = torch.zeros(rand_sg.edge_index.size(1), dtype=torch.long)
    rand_sg.node_type = torch.zeros(num_nodes, dtype=torch.long)
    rand_sg.nid2swid = [[0] for x in range(num_nodes)]
    rand_sg.eid2swid = [[0] for x in range(rand_sg.edge_index.size(1))]

    return rand_sg


def kg_adapter_right_pad_collate_fn(data, args):
    from torch_geometric.loader import DataLoader
    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    def build_mask_right(x):
        mask = [1] * len(x) + [0] * (max_len - len(x))
        return torch.tensor(mask)

    idx = [data[i][0].type(torch.int64) for i in range(len(data))]
    input_ids = [data[i][1].type(torch.int64) for i in range(len(data))]
    labels = [data[i][2].type(torch.int64) for i in range(len(data))]
    prompt_len = [data[i][3].type(torch.int64) for i in range(len(data))]
    sg_lst = []
    n2w_lst = []
    for i in range(len(data)):
        sg_data = data[i][4].clone() if data[i][4] is not None else None
        if sg_data is None:
            sg_data = build_full_pad_graph()

        if 'n2w' in sg_data.keys:
            n2w_lst.append(sg_data.n2w)
            del sg_data.n2w
        else:
            n2w_lst.append([])
        if 'trips' in sg_data.keys:
            del sg_data.trips
        if len(sg_data.x) <= 1 or len(sg_data.edge_type) <= 1:
            sg_data = build_full_pad_graph()
        if "no_kg" in args.ablation_exp_set:
            sg_data = build_full_pad_graph()
        sg_lst.append(sg_data)

    # cut to max nodes num to limit the max GPU memery usage
    max_node_num = args.max_node_num_per_batch
    for sg in sg_lst:
        keep_edge_idx = []
        edges = sg.edge_index.T
        if len(sg.x) > max_node_num:
            for i in range(edges.size(0)):
                edge = edges[i]
                if edge[0] < max_node_num and edge[1] < max_node_num:
                    keep_edge_idx.append(i)
            sg.edge_index = sg.edge_index[:, torch.tensor(keep_edge_idx)]
            sg.edge_type = sg.edge_type[torch.tensor(keep_edge_idx)]
            sg.x = sg.x.view(-1)[:max_node_num].long()
            assert sg.validate()
        sg.num_nodes = sg.x.size(0)
        sg.num_edges = sg.edge_index.size(1)
        if args.num_relations == 1:
            sg.edge_type = torch.zeros(sg.edge_type.shape, dtype=sg.edge_type.dtype)

    loader = DataLoader(sg_lst, batch_size=len(sg_lst))
    sg = next(iter(loader))

    bsz = len(sg.ptr) - 1

    sg.node_ids, sg.node_mask = to_dense_batch(sg.x, sg.batch)
    sg.max_node_num = max(sg.node_mask.sum(-1))
    sg.prune_mask = torch.ones(sg.num_nodes)

    # process text data
    max_len = max(len(s.view(-1)) for s in input_ids)

    if "align_mask" in args.exp_set:
        for bs in range(bsz):
            tmp = torch.cat([n2w_lst[bs], torch.zeros(max_len - n2w_lst[bs].size(0), n2w_lst[bs].size(1))])
            tmp = torch.cat([tmp, torch.zeros(tmp.size(0), sg.max_node_num - tmp.size(1))], dim=1)
            n2w_lst[bs] = tmp
        sg.align_mask = torch.stack(n2w_lst)

    if "mix_emb" in args.exp_set and 'nid2swid' in sg.keys:
        nid2swid = []
        max_swid = max([len(x) for xs in sg.nid2swid for x in xs])
        for bs in range(bsz):
            tmp = []
            for swid in sg.nid2swid[bs]:
                tmp.append(swid + (max_swid - len(swid)) * [args.pad_id])
            tmp += (sg.max_node_num - len(tmp)) * [[args.pad_id] * max_swid]
            nid2swid.append(torch.tensor(tmp, dtype=torch.int64))
        sg.nid2swid = torch.stack(nid2swid)

    if "mix_emb" in args.exp_set and 'eid2swid' in sg.keys:
        eid2swid = []
        max_swid = max([len(x) for xs in sg.eid2swid for x in xs])
        for bs in range(bsz):
            for swid in sg.eid2swid[bs]:
                eid2swid.append(swid + (max_swid - len(swid)) * [args.pad_id])
        tmp = torch.tensor(eid2swid, dtype=torch.int64)
        sg.eid2swid = tmp

    if "use_edge_emb" not in args.exp_set:
        sg.edge_type = torch.zeros_like(sg.edge_type)

    if "use_cat_trips" in args.exp_set:
        cnt = 0
        cul_edge_num = [0]
        for x in sg.num_edges:
            cnt += x.item()
            cul_edge_num.append(cnt)

        trip_rep = []
        trip_num = []
        for bs in range(bsz):
            node = sg.x[sg.ptr[bs]: sg.ptr[bs + 1]]
            # src = sg.edge_type.unique()
            # tgt = sg.edge_type[cul_edge_num[bs]: cul_edge_num[bs+1]].unique()
            # edge = torch.searchsorted(src, tgt)
            edge = sg.edge_type[cul_edge_num[bs]: cul_edge_num[bs + 1]]

            trip_rep.append(torch.cat([node, edge]).tolist())
            trip_num.append([len(node), len(edge)])

        max_trip_num = max([len(x) for x in trip_rep])
        trip_mask = torch.zeros(bsz, max_trip_num)
        node_mask = torch.zeros(bsz, max_trip_num)
        edge_mask = torch.zeros(bsz, max_trip_num)
        for bs in range(bsz):
            trip_mask[bs, :len(trip_rep[bs])] = 1
            node_mask[bs, :trip_num[bs][0]] = 1
            edge_mask[bs, trip_num[bs][0]: trip_num[bs][0] + trip_num[bs][1]] = 1
            trip_rep[bs] = trip_rep[bs] + [0] * (max_trip_num - len(trip_rep[bs]))

        trip_rep = torch.tensor(trip_rep)
        trip_mask = trip_mask.bool()
        node_mask = node_mask.bool()
        edge_mask = edge_mask.bool()

        sg.trips = {"trip_ids": trip_rep, "trip_num": trip_num, "trip_mask": trip_mask, "node_mask": node_mask,
                    "edge_mask": edge_mask}

    x_no_res = torch.stack([pad_right(input_ids[i][:prompt_len[i]], pad_id=args.pad_id) for i in range(len(input_ids))])
    x_no_res_mask = torch.stack([build_mask_right(input_ids[i][:prompt_len[i]]) for i in range(len(input_ids))])

    mask = torch.stack([build_mask_right(x) for x in input_ids])
    x = torch.stack([pad_right(x, pad_id=args.pad_id) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-100) for x in labels])

    prompt_len = torch.stack(prompt_len)
    idx = torch.stack(idx)

    return idx, x, y, mask, prompt_len, x_no_res, x_no_res_mask, sg


def kg_adapter_left_pad_collate_fn(data, args):
    from torch_geometric.loader import DataLoader
    def pad_left(x, pad_id):
        # pad left based on the longest sequence
        n = max_len - len(x)
        return torch.cat((torch.full((n,), pad_id, dtype=x.dtype), x))

    def build_mask_left(x):
        mask = [0] * (max_len - len(x)) + [1] * len(x)
        return torch.tensor(mask)

    idx = [data[i][0].type(torch.int64) for i in range(len(data))]
    input_ids = [data[i][1].type(torch.int64) for i in range(len(data))]
    labels = [data[i][2].type(torch.int64) for i in range(len(data))]
    prompt_len = [data[i][3].type(torch.int64) for i in range(len(data))]
    sg_lst = []
    n2w_lst = []
    for i in range(len(data)):
        sg_data = data[i][4].clone()
        if 'n2w' in sg_data.keys:
            n2w_lst.append(sg_data.n2w)
            del sg_data.n2w
        else:
            n2w_lst.append([])
        if 'trips' in sg_data.keys:
            del sg_data.trips
        if len(sg_data.x) <= 1:
            sg_data = build_full_pad_graph()
        if "no_kg" in args.ablation_exp_set:
            sg_data = build_full_pad_graph()
        sg_lst.append(sg_data)

    # cut to max nodes num to limit the max GPU memery usage
    max_node_num = args.max_node_num_per_batch
    for sg in sg_lst:
        keep_edge_idx = []
        edges = sg.edge_index.T
        if len(sg.x) > max_node_num:
            for i in range(edges.size(0)):
                edge = edges[i]
                if edge[0] < max_node_num and edge[1] < max_node_num:
                    keep_edge_idx.append(i)
            sg.edge_index = sg.edge_index[:, torch.tensor(keep_edge_idx)]
            sg.edge_type = sg.edge_type[torch.tensor(keep_edge_idx)]
            sg.x = sg.x.view(-1)[:max_node_num].long()
            assert sg.validate()
        sg.num_nodes = sg.x.size(0)
        sg.num_edges = sg.edge_index.size(1)
        if args.num_relations == 1:
            sg.edge_type = torch.zeros(sg.edge_type.shape, dtype=sg.edge_type.dtype)

    loader = DataLoader(sg_lst, batch_size=len(sg_lst))
    sg = next(iter(loader))
    bsz = len(sg.ptr) - 1
    # node_ids = []
    # node_mask = []
    # bsz = len(sg.ptr) - 1
    # max_len = max([sg.ptr[i] - sg.ptr[i - 1] for i in range(len(sg.ptr))][1:]).item()
    # for bs in range(bsz):
    #     batch = sg.x[sg.batch == bs].view(-1)
    #     node_ids.append(pad_left(batch, pad_id=0))
    #     node_mask.append(build_mask_left(batch))
    #
    # sg.node_ids = torch.stack(node_ids).type(torch.int64)
    # sg.node_mask = torch.stack(node_mask)
    # sg.max_node_num = max_len
    sg.node_ids, sg.node_mask = to_dense_batch(sg.x, sg.batch)
    sg.max_node_num = max(sg.node_mask.sum(-1))
    sg.prune_mask = torch.ones(sg.num_nodes)

    # process text data
    max_len = max(len(s.view(-1)) for s in input_ids)

    if "align_mask" in args.exp_set:
        for bs in range(bsz):
            tmp = torch.cat([torch.zeros(max_len - n2w_lst[bs].size(0), n2w_lst[bs].size(1)), n2w_lst[bs]])
            tmp = torch.cat([torch.zeros(tmp.size(0), sg.max_node_num - tmp.size(1)), tmp], dim=1)
            n2w_lst[bs] = tmp
        sg.align_mask = torch.stack(n2w_lst)

    if "mix_emb" in args.exp_set and 'nid2swid' in sg.keys:
        nid2swid = []
        max_swid = max([len(x) for xs in sg.nid2swid for x in xs])
        for bs in range(bsz):
            tmp = []
            for swid in sg.nid2swid[bs]:
                tmp.append((max_swid - len(swid)) * [args.pad_id] + swid)
            tmp = (sg.max_node_num - len(tmp)) * [[args.pad_id] * max_swid] + tmp
            nid2swid.append(torch.tensor(tmp, dtype=torch.int64))
        sg.nid2swid = torch.stack(nid2swid)

    if "mix_emb" in args.exp_set and 'eid2swid' in sg.keys:
        eid2swid = []
        max_swid = max([len(x) for xs in sg.eid2swid for x in xs])
        for bs in range(bsz):
            for swid in sg.eid2swid[bs]:
                eid2swid.append((max_swid - len(swid)) * [args.pad_id] + swid)
        tmp = torch.tensor(eid2swid, dtype=torch.int64)
        sg.eid2swid = tmp

    if "use_edge_emb" not in args.exp_set:
        sg.edge_type = torch.zeros_like(sg.edge_type)

    if "use_cat_trips" in args.exp_set:
        cnt = 0
        cul_edge_num = [0]
        for x in sg.num_edges:
            cnt += x.item()
            cul_edge_num.append(cnt)

        trip_rep = []
        trip_num = []
        for bs in range(bsz):
            node = sg.x[sg.ptr[bs]: sg.ptr[bs + 1]]
            # src = sg.edge_type.unique()
            # tgt = sg.edge_type[cul_edge_num[bs]: cul_edge_num[bs+1]].unique()
            # edge = torch.searchsorted(src, tgt)
            edge = sg.edge_type[cul_edge_num[bs]: cul_edge_num[bs + 1]]

            trip_rep.append(torch.cat([node, edge]).tolist())
            trip_num.append([len(node), len(edge)])

        max_trip_num = max([len(x) for x in trip_rep])
        trip_mask = torch.zeros(bsz, max_trip_num)
        node_mask = torch.zeros(bsz, max_trip_num)
        edge_mask = torch.zeros(bsz, max_trip_num)
        for bs in range(bsz):
            trip_mask[bs, :len(trip_rep[bs])] = 1
            node_mask[bs, :trip_num[bs][0]] = 1
            edge_mask[bs, trip_num[bs][0]: trip_num[bs][0] + trip_num[bs][1]] = 1
            trip_rep[bs] = trip_rep[bs] + [0] * (max_trip_num - len(trip_rep[bs]))

        trip_rep = torch.tensor(trip_rep)
        trip_mask = trip_mask.bool()
        node_mask = node_mask.bool()
        edge_mask = edge_mask.bool()

        sg.trips = {"trip_ids": trip_rep, "trip_num": trip_num, "trip_mask": trip_mask, "node_mask": node_mask,
                    "edge_mask": edge_mask}

    x_no_res = torch.stack([pad_left(input_ids[i][:prompt_len[i]], pad_id=args.pad_id) for i in range(len(input_ids))])
    x_no_res_mask = torch.stack([build_mask_left(input_ids[i][:prompt_len[i]]) for i in range(len(input_ids))])

    mask = torch.stack([build_mask_left(x) for x in input_ids])
    x = torch.stack([pad_left(x, pad_id=args.pad_id) for x in input_ids])
    y = torch.stack([pad_left(x, pad_id=-100) for x in labels])

    prompt_len = torch.stack(prompt_len)
    idx = torch.stack(idx)

    return idx, x, y, mask, prompt_len, x_no_res, x_no_res_mask, sg

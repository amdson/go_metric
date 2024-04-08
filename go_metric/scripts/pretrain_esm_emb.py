import numpy as np
import json, torch
from torch.utils.data import DataLoader
from go_metric.data_utils import *
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from Bio import AlignIO
import os, re

device = torch.device('cuda:1')

import pickle
with open('/home/andrew/go_metric/checkpoints/esm_emb_hparams.pkl', 'rb') as f:
    hparams = pickle.load(f)

train_path = "/home/andrew/go_metric/data/go_bench"
train_dataset = BertSeqDataset.from_pickle(f"{train_path}/train.pkl")
val_dataset = BertSeqDataset.from_pickle(f"{train_path}/val.pkl")
test_dataset = BertSeqDataset.from_pickle(f"{train_path}/test.pkl")

tokenizer = AutoTokenizer.from_pretrained(hparams.model_name)
collate_seqs = get_custom_seq_collator(tokenizer, max_length=hparams.max_length, add_special_tokens=True)
val_dataloader_params = {"shuffle": False, "batch_size": 24, "collate_fn":collate_seqs}

train_loader = DataLoader(train_dataset, **val_dataloader_params, num_workers=6)
val_loader = DataLoader(val_dataset, **val_dataloader_params)
test_loader = DataLoader(test_dataset, **val_dataloader_params)

hparams.num_classes = train_dataset[0]['labels'].shape[0]
model = AutoModel.from_pretrained(hparams.model_name)
model.to(device)
model.eval()

def embed_dataset(dataloader, model):
    prot_id = []
    embedding_l = []
    for batch in dataloader:
        prot_id.extend(batch['prot_id'])
        tokenized_sequences = batch["seq"].to(device)
        attention_mask = batch["mask"].to(device)
        seq_lens = attention_mask.sum(dim=1).cpu().numpy()-2
        with torch.no_grad():
            embeddings = model(
                input_ids=tokenized_sequences, attention_mask=attention_mask
            )
            embeddings = embeddings[0].cpu().numpy()
            for seq_num, seq_len in enumerate(seq_lens):
                # slice off first and last positions (special tokens)
                embedding = embeddings[seq_num][1 : seq_len + 1].mean(axis=0)
                embedding_l.append(embedding)
        if(len(embedding_l) % 10 == 0):
            print(len(embedding_l)/(len(dataloader)*24))
    res_dict = {"prot_id": prot_id, "embeddings": np.vstack(embedding_l)}
    return res_dict

train_dict = embed_dataset(train_loader, model)
with open("emb/pretrained_esm_train.pkl", "wb") as f:
    pickle.dump(train_dict, f)
val_dict = embed_dataset(val_loader, model)
with open("emb/pretrained_esm_val.pkl", "wb") as f:
    pickle.dump(val_dict, f)
test_dict = embed_dataset(test_loader, model)
with open("emb/pretrained_esm_test.pkl", "wb") as f:
    pickle.dump(test_dict, f)
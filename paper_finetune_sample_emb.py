import json, torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import go_bench
from go_bench.load_tools import load_GO_tsv_file, load_protein_sequences, convert_to_sparse_matrix
from go_metric.data_utils import *
from go_metric.models.bottleneck_dpg_conv import DPGModule
from scipy.sparse import csr_matrix, csc_matrix, dok_matrix, vstack, hstack
from sklearn.metrics import precision_recall_fscore_support

from go_metric.data_utils import *
from go_metric.models.bottleneck_dpg_conv import DPGModule
train_path = "/home/andrew/go_metric/data/go_bench"

train_dataset = BertSeqDataset.from_pickle(f"{train_path}/train.pkl")
val_dataset = BertSeqDataset.from_pickle(f"{train_path}/val.pkl")
test_dataset = BertSeqDataset.from_pickle(f"{train_path}/test.pkl")

collate_seqs = get_bert_seq_collator(max_length=1024, add_special_tokens=False)
val_dataloader_params = {"shuffle": False, "batch_size": 256, "collate_fn":collate_seqs}
train_loader = DataLoader(train_dataset, **val_dataloader_params, num_workers=6)
val_loader = DataLoader(val_dataset, **val_dataloader_params, num_workers=6)
test_loader = DataLoader(test_dataset, **val_dataloader_params, num_workers=6)

from go_metric.models.bert_emb import ProtBertBFDClassifier
import pickle
with open("checkpoints/bert_emb_sample_hparams.pkl", "rb") as f:
    hparams = pickle.load(f)
model = ProtBertBFDClassifier.load_from_checkpoint("checkpoints/bert_emb_sample.ckpt", hparams=hparams)
model.eval()
device = torch.device('cuda:1')
model.to(device)

def get_finetune_embeddings(model, dataset, device):
    collate_seqs = get_bert_seq_collator(max_length=1024, add_special_tokens=True)
    dataloader = DataLoader(dataset, collate_fn=collate_seqs, batch_size=128, shuffle=False)
    prot_ids, emb_l = [], []
    with torch.no_grad():
        for inputs in dataloader:
            prot_ids.extend(inputs['prot_id'])
            tokenized_sequences = inputs["seq"].to(device)
            attention_mask = inputs["mask"].to(device)
            
            word_embeddings = model.ProtBertBFD(tokenized_sequences,
                                           attention_mask)[0]
            embedding = model.pool_strategy({"token_embeddings": word_embeddings,
                                      "cls_token_embeddings": word_embeddings[:, 0],
                                      "attention_mask": attention_mask,
                                      }, pool_max=False, pool_mean_sqrt=False)
            emb_l.append(embedding.cpu())
            if(len(prot_ids) % 1024 == 0):
                print(f"{len(prot_ids)*100 / len(dataset)}%")
    embeddings = torch.cat(emb_l, dim=0)
    return prot_ids, embeddings

train_ids, train_embeddings = get_finetune_embeddings(model, train_dataset, device)
emb_dict = {"prot_id": train_ids, "embedding": train_embeddings}
with open("emb/sample_finetune_train_emb.pkl", "wb") as f:
    pickle.dump(emb_dict, f)
val_ids, val_embeddings = get_finetune_embeddings(model, val_dataset, device)
emb_dict = {"prot_id": val_ids, "embedding": val_embeddings}
with open("emb/sample_finetune_val_emb.pkl", "wb") as f:
    pickle.dump(emb_dict, f)
test_ids, test_embeddings = get_finetune_embeddings(model, test_dataset, device)
emb_dict = {"prot_id": test_ids, "embedding": test_embeddings}
with open("emb/sample_finetune_test_emb.pkl", "wb") as f:
    pickle.dump(emb_dict, f)
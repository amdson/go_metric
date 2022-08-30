import numpy as np
import json, torch
from torch.utils.data import DataLoader
from go_metric.data_utils import *
from transformers import BertModel, BertTokenizer
import re
device = torch.device('cuda:1')

bert_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
def seq_collator(data_dict_list):
    sample = collate_tensors(data_dict_list)
    inputs = bert_tokenizer.batch_encode_plus(sample["seq"],
                                                add_special_tokens=True,
                                                padding='max_length',
                                                truncation=True,
                                                return_attention_mask=True,
                                                max_length=2400)
    return {"input_ids": inputs['input_ids'], "labels": sample['labels'], 
            "prot_ids": sample["prot_id"], 'attention_mask': inputs['attention_mask']}

train_path = "/home/andrew/go_metric/data/go_bench"
train_dataset = BertSeqDataset.from_pickle(f"{train_path}/train.pkl")
val_dataset = BertSeqDataset.from_pickle(f"{train_path}/val.pkl")

model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
model.to(device)
model.eval()

def embed_dataset(dataset, model):
    dl = DataLoader(dataset, collate_fn=seq_collator, batch_size=8, shuffle=False)
    embedding_l = []
    for batch in dl:
        tokenized_sequences = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
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
        if(len(embedding_l) % 1024 == 0):
            print(len(embedding_l)/len(dataset))
            # break
    res_dict = {"prot_ids": dataset.prot_names, "embeddings": np.vstack(embedding_l)}
    return res_dict

train_emb = embed_dataset(train_dataset, model)
import pickle
with open("eval/predictions/protbert_train_emb.pkl", "wb") as f:
    pickle.dump(train_emb, f)
val_emb = embed_dataset(val_dataset, model)
with open("eval/predictions/protbert_val_emb.pkl", "wb") as f:
    pickle.dump(val_emb, f)
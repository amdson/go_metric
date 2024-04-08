import numpy as np
import json, torch
from torch.utils.data import DataLoader
from go_metric.data_utils import *
from transformers import BertModel, BertTokenizer
import re
from Bio import AlignIO
import os

hamas_families = []
hamas_sequences = []
for i, prot_fam in enumerate(os.listdir("data/hamap_alignments/")):
    family_seq = list(AlignIO.read(f"data/hamap_alignments/{prot_fam}", "fasta"))[:512]
    hamas_sequences.extend(family_seq)
    hamas_families.extend([prot_fam]*len(family_seq))
    if(i > 500):
        break

# Gen hamap embeddings
import torch
from go_metric.data_utils import *
from torch.utils.data import DataLoader

class BERTHamasData(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, prot_names, prot_fam, sequences):
        self.prot_names = prot_names
        self.sequences = sequences #A list of strings representing proteins
        self.prot_fam = prot_fam
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, index):
        X = self.sequences[index]
        prot_id = self.prot_names[index]
        prot_fam = self.prot_fam[index]
        return {"seq": X, "prot_id": prot_id, "prot_fam": prot_fam}

hamap_dataset = BERTHamasData([seq.id for seq in hamas_sequences], hamas_families,
                              [" ".join(str(seq.seq).replace('.', '').replace('-', '')) for seq in hamas_sequences])


device = torch.device('cuda:2')

bert_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
def seq_collator(data_dict_list):
    sample = collate_dict(data_dict_list)
    inputs = bert_tokenizer.batch_encode_plus(sample["seq"],
                                                add_special_tokens=True,
                                                padding='max_length',
                                                truncation=True,
                                                return_attention_mask=True,
                                                max_length=2400)
    return {"input_ids": inputs['input_ids'], "prot_ids": sample["prot_id"], 'attention_mask': inputs['attention_mask']}

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
    res_dict = {"prot_id": dataset.prot_names, "embeddings": np.vstack(embedding_l)}
    return res_dict

hamap_dict = embed_dataset(hamap_dataset, model)
with open("emb/base_hamap_emb.pkl", "wb") as f:
    pickle.dump(hamap_dict, f)
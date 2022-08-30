import gzip
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from scipy.sparse import csr_matrix, csc_matrix, dok_matrix, lil_matrix
from torch.utils import data
from torchnlp.utils import collate_tensors
from transformers import BertTokenizer
from go_bench.load_tools import load_GO_tsv_file, load_protein_sequences, convert_to_sparse_matrix

class SequenceDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, prot_names, sequences, labels, max_len=300):
        self.prot_names = prot_names
        self.labels = labels #A csr matrix in which the ith row gives the classifications of the ith protein
        self.sequences = sequences #A list of strings representing proteins
        self.max_len = max_len
        
    @classmethod
    def from_memory(cls, annotation_tsv_path, terms_list_path, sequence_path):
        with open(terms_list_path, "r") as f:
            term_list = json.load(f)
        protein_annotation_dict = load_GO_tsv_file(annotation_tsv_path)
        prot_id_whitelist = [prot_id for prot_id in protein_annotation_dict.keys()]
        sequences, prot_ids = load_protein_sequences(sequence_path, prot_id_whitelist)
        labels = convert_to_sparse_matrix(protein_annotation_dict, term_list, prot_ids)
        return cls(prot_ids, sequences, labels)
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.sequences)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.sequences[index]
        y = torch.squeeze(torch.from_numpy(self.labels[index, :].toarray()), 0)
        return X, y

class BertSeqDataset(SequenceDataset):
    def __getitem__(self, index):
        X = " ".join(self.sequences[index].upper())
        y = torch.squeeze(torch.from_numpy(self.labels[index, :].toarray()), 0)
        prot_id = self.prot_names[index]
        return {"seq": X, "labels": y, "prot_id": prot_id}
    
    def to_pickle(self, fn):
        import pickle
        with open(fn, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, fn):
        import pickle
        with open(fn, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def from_dgp_pickle(cls, term_fn, prot_fn):
        import pickle
        with open(term_fn, 'rb') as f:
            terms = pickle.load(f)['terms'].to_list()
        with open(prot_fn, 'rb') as f:
            df = pickle.load(f)
        prot_names = df['proteins'].to_list()
        sequences = df['sequences'].to_list()
        prot_dict = {}
        for prot_id, annotations in zip(df['proteins'], df['prop_annotations']):
            prot_dict[prot_id] = annotations
        labels = convert_to_sparse_matrix(prot_dict, terms, prot_names)
        return BertSeqDataset(prot_names, sequences, labels, max_len=1000)

def get_bert_seq_collator(max_length=500, add_special_tokens=False):
    bert_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
    def seq_collator(data_dict_list):
        sample = collate_tensors(data_dict_list)
        inputs = bert_tokenizer.batch_encode_plus(sample["seq"],
                                                    add_special_tokens=add_special_tokens,
                                                    padding='max_length',
                                                    truncation=True,
                                                    return_attention_mask=True,
                                                    max_length=max_length)
        return {"seq": torch.tensor(inputs['input_ids']), "labels": sample['labels'], 
                "prot_ids": sample["prot_id"], 'mask': torch.BoolTensor(inputs['attention_mask'])}
    return seq_collator

def get_seq_collator(vocab, device=None):
    def collate_sequences(lst):
        sequences, labels = [], []
        for seq, label in lst:
            sequences.append(seq)
            labels.append(label)
        return vocab.to_input_tensor(sequences, device=device), torch.stack(labels)
    return collate_sequences

def write_sparse(fn, preds, prot_rows, GO_cols, go, min_certainty):
    with open(fn, mode='w') as f:
        f.write("g,t,s\n")
        for row, col in zip(*preds.nonzero()):
            prot_id = prot_rows[row]
            go_id = GO_cols[col]
            val = preds[row, col]
            if(val > min_certainty and go_id in go):
                f.write(f"{prot_id},{go_id},{val}\n")
                
def read_sparse(fn, prot_rows, GO_cols):
    prm = {prot:i for i, prot in enumerate(prot_rows)}
    tcm = {term:i for i, term in enumerate(GO_cols)}
    sparse_probs = dok_matrix((len(prot_rows), len(GO_cols)))
    df = pd.read_csv(fn, skiprows=1)
    for (i, prot, go_id, prob) in df.itertuples():
        if(prot in prm and go_id in tcm):
            sparse_probs[prm[prot], tcm[go_id]] = prob
    return csr_matrix(sparse_probs)

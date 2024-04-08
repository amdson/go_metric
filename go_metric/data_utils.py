import gzip, json, os, pickle
from collections import Counter
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from scipy.sparse import csr_matrix, csc_matrix, dok_matrix, lil_matrix
from torch.utils import data
from transformers import BertTokenizer
from go_bench.load_tools import load_GO_tsv_file, load_protein_sequences, convert_to_sparse_matrix
# from torchnlp.utils import collate_tensors

def stable_hash(text:str):
  hash=0
  for ch in text:
    hash = ( hash*281  ^ ord(ch)*997) & 0xFFFFFFFF
  return hash

class SequenceDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, prot_names, sequences, labels, mini=None):
        self.prot_names = prot_names
        self.labels = labels #A csr matrix in which the ith row gives the classifications of the ith protein
        self.sequences = sequences #A list of strings representing proteins
        self.mini = mini
        
    @classmethod
    def from_memory(cls, annotation_tsv_path, terms_list_path, sequence_path, cache_dir=None):
        if(cache_dir):
            cache_id = str(stable_hash(annotation_tsv_path+terms_list_path+sequence_path))
            cache_path = f"{cache_dir}/{cache_id}.pkl"
            if(os.path.isfile(cache_path)):
                with open(cache_path, 'rb') as f:
                    print("Loading from cache_id:", cache_id)
                    return pickle.load(f)
        with open(terms_list_path, "r") as f:
            term_list = json.load(f)
        protein_annotation_dict = load_GO_tsv_file(annotation_tsv_path)
        prot_id_whitelist = [prot_id for prot_id in protein_annotation_dict.keys()]
        sequences, prot_ids = load_protein_sequences(sequence_path, prot_id_whitelist)
        labels = convert_to_sparse_matrix(protein_annotation_dict, term_list, prot_ids)
        ds = cls(prot_ids, sequences, labels)
        if(cache_dir):
            with open(cache_path, 'wb') as f:
                print("Saving to cache_id:", cache_id)
                pickle.dump(ds, f)
        return ds
        
    def __len__(self):
        'Denotes the total number of samples'
        if(self.mini is not None):
            return self.mini #Good for debugging
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
    def from_pickle(cls, fn, mini=None):
        import pickle
        with open(fn, 'rb') as f:
            s = pickle.load(f)
            s.mini = mini
            return s

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
        return BertSeqDataset(prot_names, sequences, labels)
    
class TermDataset(data.Dataset):
    def __init__(self, seq_dataset):
        self.seq_dataset = seq_dataset
    
    def __getitem__(self, index):
        row, term_col = tuple(index)
        dd = self.seq_dataset[row]
        dd['labels'] = dd['labels'][term_col:term_col+1]
        dd['target_term'] = torch.LongTensor([term_col])
        return dd
    
    @classmethod
    def from_pickle(cls, fn, mini=None):
        import pickle
        with open(fn, 'rb') as f:
            s = pickle.load(f)
            s.mini = mini
            return TermDataset(s)

import torch.utils.data as data
class TermSampler(data.Sampler):
    def __init__(self, labels, negative_ratio=4.0, validation=False):
        self.N, self.C = labels.shape
        self.negative_ratio = negative_ratio
        self.term_frequences = labels.sum(dim=0)
        self.term_weights = 1 / (self.term_frequences**0.5) #Totally made up. Upsample rare classes a bit.  
        self.pos_entries = torch.nonzero(labels) #(N, 2)
        self.pos_weights = self.term_weights[self.pos_entries[:, 1]]
        self.epoch_samples = int(self.pos_entries.shape[0]*0.1*(1+negative_ratio))

        self.validation = validation
        if(validation):
            N, C = self.N, self.C
            nneg = int(self.negative_ratio*N)
            neg_entries = torch.cat([torch.randint(0, N, (nneg, 1)), torch.randint(0, C, (nneg, 1))], dim=1)
            entries = torch.cat([self.pos_entries, neg_entries], dim=0)
            self.entries = entries
            
    def __iter__(self):
        if(self.validation):
            return (self.entries[i, :] for i in range(self.entries.shape[0]))
        N, C = self.N, self.C
        nneg = int(self.negative_ratio/(1+self.negative_ratio)*self.epoch_samples)
        neg_terms = torch.multinomial(self.term_frequences*self.term_weights, nneg, replacement=True).view(-1, 1)
        neg_entries = torch.cat([torch.randint(0, N, (nneg, 1)), neg_terms], dim=1)
        pos_sample_inds = torch.multinomial(self.pos_weights, int(self.pos_entries.shape[0]*0.1), replacement=False)
        pos_samples = self.pos_entries[pos_sample_inds, :]
        entries = torch.cat([pos_samples, neg_entries], dim=0)
        return (entries[i, :] for i in torch.randperm(entries.shape[0]))

def collate_dict(data_dict_l):
    keys = list(data_dict_l[0].keys())
    ex = data_dict_l[0]
    dd = {}
    for k, v in ex.items():
        if(type(v) is torch.Tensor):
            dd[k] = torch.stack([data_dict_l[i][k] for i in range(len(data_dict_l))])
        else:
            dd[k] = [data_dict_l[i][k] for i in range(len(data_dict_l))]                 
    return dd

def get_bert_seq_collator(max_length=500, add_special_tokens=False):
    bert_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
    def seq_collator(data_dict_list):
        sample = collate_dict(data_dict_list)
        inputs = bert_tokenizer.batch_encode_plus(sample["seq"],
                                                    add_special_tokens=add_special_tokens,
                                                    padding='max_length',
                                                    truncation=True,
                                                    return_attention_mask=True,
                                                    max_length=max_length)
        sample['seq'] = torch.tensor(inputs['input_ids'])
        sample['mask'] = torch.BoolTensor(inputs['attention_mask'])
        return sample
    return seq_collator

def get_custom_seq_collator(tokenizer, max_length=500, add_special_tokens=False):
    def seq_collator(data_dict_list):
        sample = collate_dict(data_dict_list)
        inputs = tokenizer.batch_encode_plus(sample["seq"],
                                                    add_special_tokens=add_special_tokens,
                                                    padding='max_length',
                                                    truncation=True,
                                                    return_attention_mask=True,
                                                    max_length=max_length)
        sample['seq'] = torch.tensor(inputs['input_ids'])
        sample['mask'] = torch.BoolTensor(inputs['attention_mask'])
        return sample
    return seq_collator

def get_seq_collator(vocab, device=None):
    def collate_sequences(lst):
        sequences, labels = [], []
        for seq, label in lst:
            sequences.append(seq)
            labels.append(label)
        return vocab.to_input_tensor(sequences, device=device), torch.stack(labels)
    return collate_sequences



class EmbData(data.Dataset):
    def __init__(self, prot_ids, embeddings, labels):
        self.prot_ids = prot_ids
        self.embeddings = embeddings
        self.labels = labels

    def __getitem__(self, index):
        return self.prot_ids[index], self.embeddings[index], torch.squeeze(torch.from_numpy(self.labels[index, :].toarray()), 0)
    def __len__(self):
        return len(self.prot_ids)

    @classmethod
    def from_file(cls, emb_path, label_path, term_list_path):
        import pickle
        with open(emb_path, "rb") as f:
            emb = pickle.load(f)
            embeddings = emb["embeddings"]
            prot_ids = emb["prot_ids"]
        with open(term_list_path, "r") as f:
            term_list = json.load(f)
        protein_annotation_dict = load_GO_tsv_file(label_path)
        labels = convert_to_sparse_matrix(protein_annotation_dict, term_list, prot_ids)
        return EmbData(prot_ids, embeddings, labels)

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

def map_embeddings(train_terms, emb_terms, emb):
    emb_mapping = {go_id: i for i, go_id in enumerate(emb_terms)}
    l = []
    for term in train_terms:
        if(term in emb_mapping):
            l.append(emb[emb_mapping[term], :])
        else:
            print("default zero")
            l.append(np.zeros(emb.shape[1]))
    term_embeddings = np.array(l)
    return term_embeddings
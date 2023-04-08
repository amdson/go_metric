from scipy.sparse import csc_matrix
import numpy as np
import torch
from torch.utils.data import DataLoader
from go_metric.data_utils import TermDataset, TermSampler, BertSeqDataset, get_bert_seq_collator
train_path = "/home/andrew/go_metric/data/go_bench"
train_dataset = TermDataset.from_pickle(f"{train_path}/train.pkl")
val_dataset = TermDataset.from_pickle(f"{train_path}/val.pkl")
train_sampler = TermSampler(torch.BoolTensor(train_dataset.seq_dataset.labels.todense()), 50000)
val_sampler = TermSampler(torch.BoolTensor(val_dataset.seq_dataset.labels.todense()), validation=True)

print(train_dataset.seq_dataset.labels.shape)
samples = [train_dataset[sample] for i, sample in zip(range(1000), train_sampler)]
collate_seqs = get_bert_seq_collator(max_length=1024, add_special_tokens=False)
batch = collate_seqs(samples)
print(batch['labels'])
print(batch['target_term'])

# num_terms = train_dataset.seq_dataset.labels.shape[1]
# print(f"Num terms: {num_terms}")

# collate_seqs = get_bert_seq_collator(max_length=1024, add_special_tokens=False)
# dataloader_params = {"sampler": train_sampler, "batch_size": 7, "collate_fn":collate_seqs}
# val_dataloader_params = {"sampler": val_sampler, "batch_size": 7, "collate_fn":collate_seqs}

# train_loader = DataLoader(train_dataset, **dataloader_params, num_workers=1)
# val_loader = DataLoader(val_dataset, **val_dataloader_params)

# for i, batch in zip(range(10), train_loader):
#     print(batch['labels'])
#     print(batch['target_term'])
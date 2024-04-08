import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from go_metric.models.tars_model import TARSModule
from go_metric.data_utils import *
from go_bench.metrics import calculate_ic, ic_mat
from argparse import ArgumentParser

parser = ArgumentParser()
parser = TARSModule.add_model_specific_args(parser)
model_hparams = parser.parse_known_args()[0]
# parser = pl.Trainer.add_argparse_args(parser)
hparams = parser.parse_args()

print("model hparams", model_hparams)

if __name__ == "__main__":
    train_path = "/home/andrew/go_metric/data/go_bench"
    # train_dataset = TermDataset.from_pickle(f"{train_path}/train.pkl")
    # val_dataset = TermDataset.from_pickle(f"{train_path}/val.pkl")
    train_dataset = BertSeqDataset.from_memory(f"{train_path}/training_molecular_function_annotations.tsv", 
                                            f"{train_path}/subsample_molecular_function_terms.json", 
                                            f"{train_path}/../uniprot_reviewed.fasta", cache_dir='/home/andrew/go_metric/cache')
    val_dataset = BertSeqDataset.from_memory(f"{train_path}/validation_molecular_function_annotations.tsv", 
                                               f"{train_path}/subsample_molecular_function_terms.json", 
                                               f"{train_path}/../uniprot_reviewed.fasta", cache_dir='/home/andrew/go_metric/cache')
    
    train_sampler = TermSampler(torch.BoolTensor(train_dataset.labels.toarray()))
    val_sampler = TermSampler(torch.BoolTensor(val_dataset.labels.toarray()))

    train_labels = train_dataset.labels.toarray()
    term_frequencies = train_labels.sum(axis=0)

    train_samples = np.array(list(a.numpy() for a in train_sampler))
    ispos = (train_labels[train_samples[:, 0], train_samples[:, 1]] > 0).reshape((-1,))
    pos_samples = train_samples[ispos, 1]
    neg_samples = train_samples[~ispos, 1]
    pos_ind, pos_counts = np.unique(pos_samples, return_counts=True)
    n = np.zeros(term_frequencies.shape)
    n[pos_ind] = pos_counts
    pos_counts = n
    neg_ind, neg_counts = np.unique(neg_samples, return_counts=True)
    n = np.zeros(term_frequencies.shape)
    n[neg_ind] = neg_counts
    neg_counts = n

    N, C = train_labels.shape
    base_neg_counts = N - term_frequencies
    neg_weight = (base_neg_counts / (neg_counts + 1)) * (pos_counts / term_frequencies) #Weights for neg samples
    neg_weight = torch.tensor(neg_weight)
    num_terms = train_dataset.labels.shape[1]

    print(neg_weight)

    #Add wrapper for term sampling
    train_dataset, val_dataset = TermDataset(train_dataset), TermDataset(val_dataset)

    import pickle
    import numpy as np

    with open("/home/andrew/go_metric/data/owl_emb/3-L-R-2-50", "rb") as f:
        embeddings = pickle.load(f, encoding="bytes")
    owl_terms = [a[0].split(r'/')[-1].replace("_", ":") for a in embeddings]
    owl_mat = np.array([a[1] for a in embeddings])
    import json
    train_path = "/home/andrew/go_metric/data/go_bench"
    with open(f"{train_path}/molecular_function_terms.json") as f:
        train_terms = json.load(f)
    go_emb = torch.FloatTensor(map_embeddings(train_terms, owl_terms, owl_mat))

    print(f"Num terms: {num_terms}")
    # go_emb = torch.normal(0, 1, (num_terms, 1024)) #Placeholder for GO term embeddings later
    module = TARSModule(term_emb=go_emb, neg_weight=neg_weight, **vars(model_hparams))

    collate_seqs = get_bert_seq_collator(max_length=hparams.max_len, add_special_tokens=False)
    dataloader_params = {"sampler": train_sampler, "batch_size": hparams.batch_size, "collate_fn":collate_seqs, "num_workers": 0}
    val_dataloader_params = {"sampler": val_sampler, "batch_size": hparams.batch_size, "collate_fn":collate_seqs, "num_workers": 0}

    train_loader = DataLoader(train_dataset, **dataloader_params)
    val_loader = DataLoader(val_dataset, **val_dataloader_params)
    
    early_stop_callback = EarlyStopping(monitor="loss/val", min_delta=0.00, patience=10, 
                                        verbose=True, mode='min', check_on_train_epoch_end=True)
    checkpoint_callback = ModelCheckpoint(
        filename="/home/andrew/go_metric/checkpoints/tars-model-subsample",
        verbose=True,
        monitor="loss/val",
        save_on_train_epoch_end=True,
        mode='min'
    )

    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger("logs", name="tars_model_subsample")
    trainer = pl.Trainer(devices=[1], max_epochs=150, callbacks=[early_stop_callback, checkpoint_callback], logger=logger)
    trainer.fit(module, train_loader, val_loader)
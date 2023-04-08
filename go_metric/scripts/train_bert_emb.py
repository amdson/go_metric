import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from go_metric.models.bert_emb import ProtBertBFDClassifier
from go_metric.data_utils import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser = ProtBertBFDClassifier.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
hparams = parser.parse_args()

print("got hparams", hparams)

if __name__ == "__main__":
    # train_dataset = BertSeqDataset.from_dgp_pickle("../dgp_data/data/terms.pkl", "../dgp_data/data/train_data.pkl")
    # val_dataset = BertSeqDataset.from_dgp_pickle("../dgp_data/data/terms.pkl", "../dgp_data/data/test_data.pkl")

    train_path = "/home/andrew/go_metric/data/go_bench"
    train_dataset = BertSeqDataset.from_pickle(f"{train_path}/train.pkl")
    val_dataset = BertSeqDataset.from_pickle(f"{train_path}/val.pkl")

    collate_seqs = get_bert_seq_collator(max_length=hparams.max_length, add_special_tokens=True)
    dataloader_params = {"shuffle": True, "batch_size": 12, "collate_fn":collate_seqs}
    val_dataloader_params = {"shuffle": False, "batch_size": 24, "collate_fn":collate_seqs}

    train_loader = DataLoader(train_dataset, **dataloader_params, num_workers=6)
    val_loader = DataLoader(val_dataset, **val_dataloader_params)

    hparams.num_classes = train_dataset[0]['labels'].shape[0]
    model = ProtBertBFDClassifier(hparams)
    
    early_stop_callback = EarlyStopping(monitor='F1/val', min_delta=0.00, patience=3, verbose=True, mode='max')
    checkpoint_callback = ModelCheckpoint(filename="/home/andrew/go_metric/checkpoints/bert_emb_128", verbose=True, monitor='F1/val', mode='max')
    trainer = pl.Trainer.from_argparse_args(hparams, accelerator='gpu', devices=[1], max_epochs=100, profiler='simple',
                                             callbacks=[early_stop_callback, checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
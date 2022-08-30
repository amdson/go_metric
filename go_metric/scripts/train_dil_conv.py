import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from go_metric.models.dilated_conv import DilatedConvModule
from go_metric.data_utils import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser = DilatedConvModule.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
hparams = parser.parse_args()

print("got hparams", hparams)

train_path = "/home/andrew/go_metric/data/go_bench"

if __name__ == "__main__":
    # train_dataset = BertSeqDataset.from_memory(f"{train_path}/training_molecular_function_annotations.tsv", 
    #                                             f"{train_path}/molecular_function_terms.json",
    #                                             "/home/andrew/go_metric/data/uniprot_reviewed.fasta")
    # val_dataset = BertSeqDataset.from_memory(f"{train_path}/validation_molecular_function_annotations.tsv", 
    #                                         f"{train_path}/molecular_function_terms.json",
    #                                         "/home/andrew/go_metric/data/uniprot_reviewed.fasta")
    train_dataset = BertSeqDataset.from_pickle('/home/andrew/go_metric/data/train_ds.pkl')
    val_dataset = BertSeqDataset.from_pickle('/home/andrew/go_metric/data/val_ds.pkl')

    collate_seqs = get_bert_seq_collator(max_length=hparams.max_len, add_special_tokens=False)
    dataloader_params = {"shuffle": True, "batch_size": hparams.batch_size, "collate_fn":collate_seqs}
    val_dataloader_params = {"shuffle": False, "batch_size": 256, "collate_fn":collate_seqs}

    train_loader = DataLoader(train_dataset, **dataloader_params, num_workers=6)
    val_loader = DataLoader(val_dataset, **val_dataloader_params)

    model = DilatedConvModule(hparams)

    early_stop_callback = EarlyStopping(monitor='F1/val', min_delta=0.00, patience=3, verbose=True, mode='max')
    checkpoint_callback = ModelCheckpoint(filename="/home/andrew/go_metric/checkpoints/dilated_conv", verbose=True, monitor=None)
    trainer = pl.Trainer.from_argparse_args(hparams, accelerator='gpu', devices=[1], max_epochs=100, profiler='simple',
                                             callbacks=[early_stop_callback, checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
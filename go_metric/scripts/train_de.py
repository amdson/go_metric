import os, pickle
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from go_metric.models.deep_emb import EmbMLPModule
from go_metric.data_utils import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser = EmbMLPModule.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
hparams = parser.parse_args()

train_path = "/home/andrew/go_metric/data/GO_bench"


if __name__ == "__main__":
    train_dataset = EmbData.from_file("eval/predictions/protbert_train_emb.pkl", 
                                        "../data/go_bench/training_molecular_function_annotations.tsv", 
                                        "../data/go_bench/molecular_function_terms.json")
    val_dataset = EmbData.from_file("eval/predictions/protbert_val_emb.pkl", 
                                        "../data/go_bench/validation_molecular_function_annotations.tsv", 
                                        "../data/go_bench/molecular_function_terms.json")
    dataloader_params = {"shuffle": True, "batch_size": 128}
    val_dataloader_params = {"shuffle": False, "batch_size": 128}
    train_loader = DataLoader(train_dataset, **dataloader_params, num_workers=6)
    val_loader = DataLoader(val_dataset, **val_dataloader_params)

    model = EmbMLPModule(train_dataset.labels.shape[1])

    early_stop_callback = EarlyStopping(monitor='loss/val', min_delta=0.00, patience=3, verbose=True, mode='min')
    checkpoint_callback = ModelCheckpoint(filename="/home/andrew/go_metric/checkpoints/embmlp", verbose=True, monitor="loss/val")
    trainer = pl.Trainer.from_argparse_args(hparams, accelerator='gpu', devices=[1], max_epochs=100, profiler='simple',
                                             callbacks=[early_stop_callback, checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
import os, pickle
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from go_metric.models.conv_attention import ConvAttentionModule
from go_metric.data_utils import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser = ConvAttentionModule.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
hparams = parser.parse_args()

print("got hparams", hparams)

train_path = "/home/andrew/go_metric/data/non_IEA"

if __name__ == "__main__":
    train_dataset = BertSeqDataset.from_dgp_pickle("../dgp_data/data/terms.pkl", "../dgp_data/data/train_data.pkl")
    val_dataset = BertSeqDataset.from_dgp_pickle("../dgp_data/data/terms.pkl", "../dgp_data/data/test_data.pkl")
    collate_seqs = get_bert_seq_collator(max_length=hparams.max_length, add_special_tokens=False)
    dataloader_params = {"shuffle": True, "batch_size": 16, "collate_fn":collate_seqs}
    val_dataloader_params = {"shuffle": False, "batch_size": 32, "collate_fn":collate_seqs}

    train_loader = DataLoader(train_dataset, **dataloader_params, num_workers=6)
    val_loader = DataLoader(val_dataset, **val_dataloader_params)
    with open("../dgp_data/data/terms.pkl", 'rb') as f:
        terms = pickle.load(f)['terms'].to_list()

    model = ConvAttentionModule(terms, hparams)

    early_stop_callback = EarlyStopping(monitor='F1/val', min_delta=0.00, patience=3, verbose=True, mode='max')
    checkpoint_callback = ModelCheckpoint(filename="/home/andrew/go_metric/checkpoints/dilated_conv", verbose=True, monitor=None)
    trainer = pl.Trainer.from_argparse_args(hparams, accelerator='gpu', devices=[1], max_epochs=100, profiler='simple',
                                             callbacks=[early_stop_callback, checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
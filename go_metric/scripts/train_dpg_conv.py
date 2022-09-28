import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from go_metric.models.bottleneck_dpg_conv import DPGModule
from go_metric.data_utils import *
from go_bench.metrics import calculate_ic, ic_mat
from argparse import ArgumentParser


"""
batch_size: 256
num_filters: 800
bottleneck_layers: 1.0
label_loss_weight: 10
sim_margin: 12
tmargin: 0.9
learning_rate: 5e-4

label_loss_decay: 
"""

parser = ArgumentParser()
parser = DPGModule.add_model_specific_args(parser)
model_hparams = parser.parse_known_args()[0]
parser = pl.Trainer.add_argparse_args(parser)
hparams = parser.parse_args()

print("model hparams", model_hparams)

train_path = "/home/andrew/go_metric/data/go_bench"

if __name__ == "__main__":
    train_dataset = BertSeqDataset.from_pickle(f"{train_path}/train.pkl")
    val_dataset = BertSeqDataset.from_pickle(f"{train_path}/val.pkl")

    collate_seqs = get_bert_seq_collator(max_length=hparams.max_len, add_special_tokens=False)
    dataloader_params = {"shuffle": True, "batch_size": 256, "collate_fn":collate_seqs}
    val_dataloader_params = {"shuffle": False, "batch_size": 256, "collate_fn":collate_seqs}

    train_loader = DataLoader(train_dataset, **dataloader_params, num_workers=6)
    val_loader = DataLoader(val_dataset, **val_dataloader_params)
    with open(f"{train_path}/../ic_dict.json") as f:
        ic_dict = json.load(f)
    with open(f"{train_path}/molecular_function_terms.json") as f:
        terms = json.load(f)
    term_ic = torch.FloatTensor(ic_mat(terms, ic_dict).reshape((1, -1)))

    model = DPGModule(**vars(model_hparams), term_ic=term_ic)
    early_stop_callback = EarlyStopping(monitor="knn_F1/val", min_delta=0.00, patience=5, verbose=True, mode='max')
    checkpoint_callback = ModelCheckpoint(
        filename="/home/andrew/go_metric/checkpoints/bottleneck",
        verbose=True,
        monitor="knn_F1/val"
    )

    trainer = pl.Trainer(gpus=[1,], max_epochs=100, callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(model, train_loader, val_loader)

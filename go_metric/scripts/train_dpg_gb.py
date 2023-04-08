import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from go_metric.models.bottleneck_gb import DPGModule
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


if __name__ == "__main__":
    train_path = "/home/andrew/go_metric/data/go_bench"
    train_dataset = BertSeqDataset.from_pickle(f"{train_path}/train.pkl")
    val_dataset = BertSeqDataset.from_pickle(f"{train_path}/val.pkl")

    # train_dataset = BertSeqDataset.from_dgp_pickle("../dgp_data/data/terms.pkl", "../dgp_data/data/train_data.pkl")
    # val_dataset = BertSeqDataset.from_dgp_pickle("../dgp_data/data/terms.pkl", "../dgp_data/data/test_data.pkl")

    collate_seqs = get_bert_seq_collator(max_length=hparams.max_len, add_special_tokens=False)
    dataloader_params = {"shuffle": True, "batch_size": hparams.batch_size, "collate_fn":collate_seqs}
    val_dataloader_params = {"shuffle": False, "batch_size": hparams.batch_size, "collate_fn":collate_seqs}

    train_loader = DataLoader(train_dataset, **dataloader_params, num_workers=6)
    val_loader = DataLoader(val_dataset, **val_dataloader_params)
    
    # with open(f"/home/andrew/go_metric/data/go_bench/../ic_dict.json") as f:
    #     ic_dict = json.load(f)
    # with open(f"{train_path}/molecular_function_terms.json") as f:
    #     terms = json.load(f)
    # term_ic = torch.FloatTensor(ic_mat(terms, ic_dict).reshape((1, -1)))
    term_ic = torch.ones((1, train_dataset[0]['labels'].shape[0]))
    model_hparams.num_classes = train_dataset[0]['labels'].shape[0]
    model = DPGModule(**vars(model_hparams), term_ic=term_ic)
    early_stop_callback = EarlyStopping(monitor="knn_F1/val", min_delta=0.00, patience=10, 
                                        verbose=True, mode='max', check_on_train_epoch_end=True)
    checkpoint_callback = ModelCheckpoint(
        filename="/home/andrew/go_metric/checkpoints/dpg-gb-bottleneck",
        verbose=True,
        monitor="knn_F1/val",
        save_on_train_epoch_end=True,
        mode='max'
    )

    trainer = pl.Trainer(gpus=[2,], max_epochs=150, callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(model, train_loader, val_loader)
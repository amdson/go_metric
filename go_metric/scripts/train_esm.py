import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import transformers

from go_metric.models.bert_esm_emb import ESMBERTClassifier
from go_metric.data_utils import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser = ESMBERTClassifier.add_model_specific_args(parser)
# parser = pl.Trainer.add_argparse_args(parser)
hparams = parser.parse_args()

# import pickle
# with open('/home/andrew/go_metric/checkpoints/esm_emb_hparams_01.pkl', 'wb') as f:
#     pickle.dump(hparams, f)

print("got hparams", hparams)
train_path = "/home/andrew/go_metric/data/go_bench"
if __name__ == "__main__":
    train_dataset = BertSeqDataset.from_pickle(f"{train_path}/train.pkl")
    val_dataset = BertSeqDataset.from_pickle(f"{train_path}/val.pkl")
    tokenizer = transformers.AutoTokenizer.from_pretrained(hparams.model_name)
    collate_seqs = get_custom_seq_collator(tokenizer, max_length=hparams.max_length, add_special_tokens=True)
    dataloader_params = {"shuffle": True, "batch_size": 4, "collate_fn":collate_seqs}
    val_dataloader_params = {"shuffle": False, "batch_size": 24, "collate_fn":collate_seqs}

    train_loader = DataLoader(train_dataset, **dataloader_params, num_workers=6)
    val_loader = DataLoader(val_dataset, **val_dataloader_params)

    hparams.num_classes = train_dataset[0]['labels'].shape[0]
    model = ESMBERTClassifier.load_from_checkpoint("/home/andrew/go_metric/checkpoints/esm_emb.ckpt")

    early_stop_callback = EarlyStopping(monitor='F1/val', min_delta=0.00, patience=3, verbose=True, mode='max')
    checkpoint_callback = ModelCheckpoint(filename="/home/andrew/go_metric/checkpoints/esm_emb", verbose=True, monitor='F1/val', mode='max')
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger("logs", name="esm_emb")
    trainer = pl.Trainer(devices=[1], max_epochs=100, 
                         callbacks=[early_stop_callback, checkpoint_callback], logger=logger)
    # trainer = pl.Trainer(hparams, accelerator='gpu', devices=[1], max_epochs=100, profiler='simple',
    #                                          callbacks=[early_stop_callback, checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
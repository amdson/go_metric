import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from go_metric.models.bottleneck_dpg_conv import DPGModule
from go_metric.data_utils import *
from go_bench.metrics import calculate_ic, ic_mat

train_path = "/home/andrew/go_metric/data/go_bench"

if __name__ == "__main__":
    MAX_LEN = 1024
    # train_dataset = BertSeqDataset.from_memory(f"{train_path}/training_molecular_function_annotations.tsv", 
    #                                             f"{train_path}/molecular_function_terms.json",
    #                                             "/home/andrew/go_metric/data/uniprot_reviewed.fasta")
    # val_dataset = BertSeqDataset.from_memory(f"{train_path}/validation_molecular_function_annotations.tsv", 
    #                                         f"{train_path}/molecular_function_terms.json",
    #                                         "/home/andrew/go_metric/data/uniprot_reviewed.fasta")
    # train_dataset.to_pickle(f"{train_path}/train.pkl")
    # val_dataset.to_pickle(f"{train_path}/val.pkl")
    train_dataset = BertSeqDataset.from_pickle(f"{train_path}/train.pkl")
    val_dataset = BertSeqDataset.from_pickle(f"{train_path}/val.pkl")
    # train_dataset = BertSeqDataset.from_dgp_pickle("../dgp_data/data/terms.pkl", "../dgp_data/data/train_data.pkl")
    # val_dataset = BertSeqDataset.from_dgp_pickle("../dgp_data/data/terms.pkl", "../dgp_data/data/test_data.pkl")
    collate_seqs = get_bert_seq_collator(max_length=MAX_LEN, add_special_tokens=False)
    dataloader_params = {"shuffle": True, "batch_size": 128, "collate_fn":collate_seqs}
    val_dataloader_params = {"shuffle": False, "batch_size": 128, "collate_fn":collate_seqs}

    train_loader = DataLoader(train_dataset, **dataloader_params, num_workers=6)
    val_loader = DataLoader(val_dataset, **val_dataloader_params)

    VOCAB_SIZE = 30
    NUM_CLASSES= train_dataset[0]['labels'].shape[0]
    with open(f"{train_path}/../ic_dict.json") as f:
        ic_dict = json.load(f)
    with open(f"{train_path}/molecular_function_terms.json") as f:
        terms = json.load(f)
    term_ic = torch.from_numpy(ic_mat(terms, ic_dict).reshape((-1, 1)))
    model = DPGModule(VOCAB_SIZE, NUM_CLASSES, max_len=MAX_LEN, term_ic=None)

    early_stop_callback = EarlyStopping(monitor='loss/val', min_delta=0.00, patience=5, verbose=True, mode='min')
    checkpoint_callback = ModelCheckpoint(
        filename="/home/andrew/go_metric/checkpoints/dgp_bottleneck_conv_dgp_data",
        verbose=True,
        monitor='loss/val'
    )

    trainer = pl.Trainer(gpus=[1,], max_epochs=100, profiler='simple', 
        auto_lr_find=True, callbacks=[early_stop_callback, checkpoint_callback])    # Train the model
    trainer.fit(model, train_loader, val_loader)

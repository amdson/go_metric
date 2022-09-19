import optuna
from optuna.trial import TrialState
from go_metric.optuna_callback import PyTorchLightningPruningCallback

import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from go_metric.models.bottleneck_dpg_conv import DPGModule
from go_metric.data_utils import *
from go_bench.metrics import calculate_ic, ic_mat
from argparse import ArgumentParser

def gen_objective(train_loader, val_loader):
    def objective(trial: optuna.trial.Trial) -> float:
        lr = 5e-4 #trial.suggest_float("lr", 5e-4, 1e-3, log=True)
        sim_margin = 12 #trial.suggest_float("sim_margin", 2, 15)
        tmargin = 0.9 #trial.suggest_float("tmargin", 0.1, 3.0)
        bottleneck_regularization = 0.01 #trial.suggest_float("bottleneck_regularization", 0, 0.3)
        # dropout = trial.suggest_uniform("dropout", 0, 1)
        num_filters = trial.suggest_int("num_filters", 128, 1024)
        bnk_layers = trial.suggest_int("num_bottleneck_layers", 1, 4)
        hidden_size = trial.suggest_int("bnk_size", 512, 4096)
        bnk_dims = [hidden_size]*bnk_layers
        bnk_size = trial.suggest_int("bnk_size", 64, 256)
        dropout = 0.5
        git_hash = get_git_revision_short_hash()
        print(f"Running trial with {num_filters=} {git_hash=} {bnk_size=} {bnk_dims=}")
        model = DPGModule(num_filters=num_filters, sim_margin=sim_margin, tmargin=tmargin, hidden_dims=bnk_dims, bottleneck_dim=bnk_size
                        bottleneck_regularization=bottleneck_regularization, learning_rate=lr, 
                        term_ic=None, git_hash=git_hash)

        callbacks=[PyTorchLightningPruningCallback(trial, monitor="knn_F1/val"),
            EarlyStopping(monitor='knn_F1/val', min_delta=0.00, patience=5, verbose=True, mode='max'),
            ModelCheckpoint(filename="/home/andrew/go_metric/checkpoints/optuna_emb", verbose=True, monitor='knn_F1/val')]

        trainer = pl.Trainer(accelerator='gpu', devices=[1,], max_epochs=100, profiler='simple', callbacks=callbacks)    # Train the model
        trainer.fit(model, train_loader, val_loader)
        return trainer.callback_metrics["knn_F1/val"].item()
    return objective

if __name__ == "__main__":
    parser = ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    train_path = "/home/andrew/go_metric/data/go_bench"
    train_dataset = BertSeqDataset.from_pickle(f"{train_path}/train.pkl")
    val_dataset = BertSeqDataset.from_pickle(f"{train_path}/val.pkl")
    collate_seqs = get_bert_seq_collator(max_length=1024, add_special_tokens=False)
    dataloader_params = {"shuffle": True, "batch_size": 256, "collate_fn":collate_seqs}
    val_dataloader_params = {"shuffle": False, "batch_size": 256, "collate_fn":collate_seqs}
    train_loader = DataLoader(train_dataset, **dataloader_params, num_workers=6)
    val_loader = DataLoader(val_dataset, **val_dataloader_params)
    with open(f"{train_path}/../ic_dict.json") as f:
        ic_dict = json.load(f)
    with open(f"{train_path}/molecular_function_terms.json") as f:
        terms = json.load(f)
    term_ic = torch.from_numpy(ic_mat(terms, ic_dict).reshape((-1, 1)))
    
    pruner: optuna.pruners.BasePruner = (optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner())
    objective = gen_objective(train_loader, val_loader)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=10, timeout=72000)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

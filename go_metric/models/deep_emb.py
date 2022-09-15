import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from go_metric.metric_loss import multilabel_triplet_loss

class EmbMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.l1 = nn.Linear(1024, 1024)
        self.bottleneck = nn.Linear(1024, 128)
        self.l2 = nn.Linear(128, 1024)
        self.classifier_layer = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, return_embedding=False):
        # out = self.embedding(x)
        out = self.l1(x)
        out = self.relu(out)
        embedding = self.bottleneck(out)
        out = self.relu(embedding)
        out = self.l2(out)
        out = self.relu(out)
        out = self.classifier_layer(out)
        if(return_embedding):
            return out, embedding
        return out

class EmbMLPModule(pl.LightningModule):
    def __init__(self, num_classes, lr=3e-4, term_ic=None):
        super().__init__()
        self.save_hyperparameters()
        self.term_ic = term_ic
        self.model = EmbMLP(num_classes)
        self.lr = lr
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, return_embedding=False):
        return self.model(x, return_embedding)

    def on_epoch_start(self):
        print('\n')

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        pids, x, y = batch
        y = y.float()
        logits, embedding = self.model.forward(x, return_embedding=True)
        label_loss = self.loss(logits, y)
        metric_loss = multilabel_triplet_loss(embedding, y, label_weights=self.term_ic, sim_margin=3.0, tmargin=1.0)
        loss = 10*label_loss + metric_loss
        # Logging to TensorBoard by default
        self.log('loss/train', loss)
        self.log('label_loss/train', label_loss)
        self.log('metric_loss/train', metric_loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        pids, x, y = batch
        y = y.float()
        logits, embedding = self.model.forward(x, return_embedding=True)
        label_loss = self.loss(logits, y)
        metric_loss = multilabel_triplet_loss(embedding, y, label_weights=self.term_ic, sim_margin=3.0, tmargin=1.0)
        loss = label_loss + metric_loss
        # Logging to TensorBoard by default
        self.log('loss/val', loss)
        self.log('label_loss/val', label_loss)
        self.log('metric_loss/val', metric_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def on_train_start(self):
        # Proper logging of hyperparams and metrics in TB
        self.logger.log_hyperparams(self.hparams, {"loss/train": 1, "loss/val": 1})

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # parser.add_argument("--input_dim", type=int, default=128)
        # parser.add_argument("--num_layers", type=int, default=8)
        # parser.add_argument('--gradient_clipping_decay', type=float, default=1.0)
        # parser.add_argument('--batch_size', type=int, default=156)
        # parser.add_argument('--dilation_rate', type=float, default=2)
        # parser.add_argument('--num_filters', type=int, default=1600)
        # parser.add_argument('--first_dilated_layer', type=int, default=4)  # This is 0-indexed
        # parser.add_argument('--kernel_size', type=int, default=9)
        # parser.add_argument('--max_len', type=int, default=1024)
        # parser.add_argument('--num_classes', type=int, default=865)
        # parser.add_argument('--vocab_size', type=int, default=30)
        # parser.add_argument('--pooling', type=str, default='max')
        # parser.add_argument('--bottleneck_factor', type=float, default=0.5)
        # parser.add_argument('--lr_decay_rate', type=float, default=0.9997)
        # parser.add_argument('--learning_rate', type=float, default=0.0005)
        return parent_parser
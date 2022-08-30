import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from go_metric.metric_loss import multilabel_triplet_loss

class DPGConvSeq(nn.Module):
    def __init__(self, vocab_size, num_classes, nb_filters, max_kernel, max_len):
        super().__init__()
        self.conv1 = nn.Conv1d(vocab_size, 128, 3, padding='same')
        self.dropout = nn.Dropout(p=0.5)
        kernel_sizes = list(range(8, max_kernel, 8))
        nets = []
        for kernel_size in kernel_sizes:
            bf = BaseFilter(128, kernel_size, nb_filters, max_len)
            nets.append(bf)
        self.nets = nn.ModuleList(nets)
        self.l1 = nn.Linear(nb_filters*len(kernel_sizes), 1024)
        self.bottleneck = nn.Linear(1024, 128)
        self.l2 = nn.Linear(128, 1024)
        self.classifier_layer = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
        self.vocab_size = vocab_size

    def forward(self, x, return_embedding=False):
        # out = self.embedding(x)
        embedding = self.embedding(x)
        out = self.relu(embedding)
        out = self.l2(out)
        out = self.relu(out)
        out = self.classifier_layer(out)
        if(return_embedding):
            return out, embedding
        return out

    def embedding(self, x):
        out = nn.functional.one_hot(x, num_classes=self.vocab_size).float()
        out = torch.transpose(out, 1, 2)
        out = self.conv1(out)
        outputs = [bf(out) for bf in self.nets]
        out = torch.cat(outputs, axis=1)
        out = self.dropout(out)
        out = self.l1(out)
        out = self.relu(out)
        out = self.bottleneck(out)
        return out

class DPGModule(pl.LightningModule):
    def __init__(self, vocab_size, num_classes, max_kernel=129, nb_filters=512, max_len=1024, lr=3e-4, term_ic=None):
        super().__init__()
        self.save_hyperparameters()
        self.term_ic = term_ic
        self.model = DPGConvSeq(vocab_size, num_classes, nb_filters, max_kernel, max_len)
        self.lr = lr
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, return_embedding=False):
        return self.model(x, return_embedding)

    def on_epoch_start(self):
        print('\n')

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        x, y = batch["seq"], batch["labels"] 
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
        x, y = batch["seq"], batch["labels"] 
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

    # def validation_epoch_end(self, outputs):
    #     labels, logits = [], []
    #     for output in outputs:
    #         labels.append(output["labels"])
    #         logits.append(output["logits"])
    #     labels = np.concatenate(labels, axis=0)
    #     logits = np.concatenate(logits, axis=0)
    #     thresholds = [-3, -1, -0.5, 0, 0.5, 1, 3]
    #     f1 = 0
    #     for threshold in thresholds:
    #         preds = logits > threshold
    #         f1 = max(f1, f1_score(labels, preds, average='micro'))
    #     self.log('F1/val', f1, prog_bar=True)
    #     return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def on_train_start(self):
        # Proper logging of hyperparams and metrics in TB
        self.logger.log_hyperparams(self.hparams, {"loss/train": 1, "loss/val": 1})
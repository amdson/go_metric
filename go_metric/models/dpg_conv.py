import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import f1_score

#Possible improvements
#Different pooling for small convs, more layers for small convs, better

def conv1d_xiaver_init(conv):
    nn.init.xavier_normal_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv

class BaseFilter(nn.Module):
    def __init__(self, vocab_size, kernel_size, nb_filters, max_len):
        super().__init__()
        self.conv = nn.Conv1d(vocab_size, nb_filters, kernel_size, padding='valid')
        conv1d_xiaver_init(self.conv)
        self.pool = nn.MaxPool1d(max_len - kernel_size + 1)
        self.relu = nn.ReLU()

    def forward(self, one_hot):
        x = self.conv(one_hot)
        x = self.pool(x)
        x = self.relu(x)
        return torch.flatten(x, start_dim=1)


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
        self.classifier_layer = nn.Linear(nb_filters*len(kernel_sizes), num_classes)
        self.vocab_size = vocab_size

    def forward(self, x):
        # out = self.embedding(x)
        out = nn.functional.one_hot(x, num_classes=self.vocab_size).float()
        out = torch.transpose(out, 1, 2)
        out = self.conv1(out)
        outputs = [bf(out) for bf in self.nets]
        out = torch.cat(outputs, axis=1)
        out = self.dropout(out)
        out = self.classifier_layer(out)
        return out

class DPGModule(pl.LightningModule):
    def __init__(self, vocab_size, num_classes, max_kernel=129, nb_filters=512, max_len=1024, lr=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = DPGConvSeq(vocab_size, num_classes, nb_filters, max_kernel, max_len)
        self.lr = lr
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def on_epoch_start(self):
        print('\n')

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch["seq"], batch["labels"] 
        logits = self.model(x)
        loss = self.loss(logits, y.float())
        # Logging to TensorBoard by default
        self.log('loss/train', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['seq'], batch['labels']
        logits = self.model(x)
        loss = self.loss(logits, y.float())

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('loss/val', loss, prog_bar=True)
        #self.log('val_acc', acc, prog_bar=True)
        return {"logits": logits.detach().cpu().numpy(), "labels": y.detach().cpu().numpy()}

    def validation_epoch_end(self, outputs):
        labels, logits = [], []
        for output in outputs:
            labels.append(output["labels"])
            logits.append(output["logits"])
        labels = np.concatenate(labels, axis=0)
        logits = np.concatenate(logits, axis=0)
        thresholds = [-3, -1, -0.5, 0, 0.5, 1, 3]
        f1 = 0
        for threshold in thresholds:
            preds = logits > threshold
            f1 = max(f1, f1_score(labels, preds, average='micro'))
        self.log('F1/val', f1, prog_bar=True)
        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def on_train_start(self):
        # Proper logging of hyperparams and metrics in TB
        self.logger.log_hyperparams(self.hparams, {"loss/train": 1, "loss/val": 1, "F1/val": 0})


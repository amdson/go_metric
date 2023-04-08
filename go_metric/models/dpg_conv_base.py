import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from go_metric.utils import tuple_type
from go_metric.metric_loss import metric_logits_loss, multilabel_triplet_loss
from go_metric.multilabel_knn import embedding_knn
from scipy import sparse

#Possible improvements
#Different pooling for small convs, more layers for small convs

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

def get_MLP(layer_dim, dropout=0.5):
    mlp = nn.Sequential()
    for i in range(len(layer_dim)-1):
        mlp.append(nn.Linear(layer_dim[i], layer_dim[i+1]))
        if(i < len(layer_dim)-2):
            mlp.append(nn.Dropout(p=dropout))
            mlp.append(nn.ReLU())
    return mlp

class DPGConvSeq(nn.Module):
    def __init__(self, vocab_size, bottleneck_dim, hidden_dims, num_classes, nb_filters, max_kernel, max_len):
        super().__init__()
        self.conv1 = nn.Conv1d(vocab_size, 128, 3, padding='same')
        self.dropout = nn.Dropout(p=0.5)
        kernel_sizes = list(range(8, max_kernel, 8))
        nets = []
        for kernel_size in kernel_sizes:
            bf = BaseFilter(128, kernel_size, nb_filters, max_len)
            nets.append(bf)
        self.nets = nn.ModuleList(nets)
        bottleneck_layers = [nb_filters*len(kernel_sizes)] + hidden_dims + [bottleneck_dim]
        self.bottleneck = get_MLP(bottleneck_layers, dropout=0)
        # print("bottleneck layers", self.bottleneck)
        self.l2 = nn.Linear(bottleneck_dim, 1024)
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
        out = self.bottleneck(out)
        return out

class DPGModule(pl.LightningModule):
    def __init__(self, vocab_size=30, num_classes=865, max_len=1024, 
                    max_kernel=129, num_filters=512, bottleneck_dim=128, hidden_dims=(2048, 2048), bottleneck_regularization=0.0, 
                    bottleneck_layers=1, classification_layers=1, label_loss_weight=10.0, label_loss_decay=0.94,
                    sim_margin=3.0, tmargin=1.0, sim_type='dot', gradient_clipping_decay=1.0, batch_size=256,
                    learning_rate=5e-4, lr_decay_rate= 0.9997, term_ic=None, git_hash=None):
        super().__init__()
        self.save_hyperparameters()
        self.term_ic = term_ic
        self.model = DPGConvSeq(vocab_size, bottleneck_dim, list(hidden_dims), num_classes, num_filters, max_kernel, max_len)
        self.lr = learning_rate
        self.bottleneck_regularization = bottleneck_regularization
        self.label_loss_weight = label_loss_weight
        self.label_loss_decay = label_loss_decay
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
        emb_loss = self.bottleneck_regularization * torch.square(embedding).sum(axis=1).mean()
        loss = label_loss + emb_loss
        # loss = label_loss
        # Logging to TensorBoard by default
        self.log('loss/train', loss)

        # self.log('batch_triplets', num_triplets)
        return loss
        
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        x, y = batch["seq"], batch["labels"] 
        y = y.float()
        logits = self.model.forward(x)
        loss = self.loss(logits, y)
        # Logging to TensorBoard by default
        self.log('loss/val', loss)
        output = {"loss": loss, "labels": batch["labels"], "logits": logits}
        return output

    def validation_epoch_end(self, outputs):
        labels, logits = [], []
        for output in outputs:
            labels.append(output["labels"])
            logits.append(output["logits"])
        labels = torch.cat(labels, dim=0).cpu().numpy()
        logits = torch.cat(logits, dim=0)

        preds = (logits > 0).cpu().numpy()
        f1 = f1_score(labels, preds, average='micro')
        self.log('F1/val', f1, prog_bar=True)

        self.val_labels = labels
        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-7)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=5)
        return optimizer
    
    def on_train_start(self):
        # Proper logging of hyperparams and metrics in TB
        self.logger.log_hyperparams(self.hparams, {"loss/train": 1, "loss/val": 1})

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument('--vocab_size', type=int, default=30)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--max_len', type=int, default=1024)

        parser.add_argument('--num_classes', type=int, default=865)
        parser.add_argument('--num_filters', type=int, default=800)
        parser.add_argument('--max_kernel', type=int, default=129)
        parser.add_argument("--bottleneck_dim", type=int, default=128)
        parser.add_argument("--bottleneck_regularization", type=float, default=0.01)

        # parser.add_argument("--bottleneck_layers", type=int, default=1)
        parser.add_argument("--hidden_dims", type=tuple_type, default=(2048, 2048))
        parser.add_argument("--classification_layers", type=int, default=1)

        parser.add_argument('--sim_type', type=str, default='dot')
        parser.add_argument('--sim_margin', type=float, default=3.0)
        parser.add_argument('--tmargin', type=float, default=0.95)

        parser.add_argument('--label_loss_weight', type=float, default=10.0)
        parser.add_argument("--label_loss_decay", type=float, default=1.0)
        parser.add_argument('--learning_rate', type=float, default=5e-4)
        parser.add_argument('--lr_decay_rate', type=float, default=0.9997)

        parser.add_argument('--gradient_clipping_decay', type=float, default=1.0)
        
        return parent_parser


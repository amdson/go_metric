import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from go_metric.metric_loss import multilabel_triplet_loss
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
    def __init__(self, vocab_size=30, num_classes=865, max_len=1024, 
                    max_kernel=129, num_filters=512, batch_size=128,
                    bottleneck_size=128, bottleneck_regularization=0.0, 
                    bottleneck_layers=1, classification_layers=1, label_loss_weight=10.0,
                    sim_margin=3.0, tmargin=1.0, gradient_clipping_decay=1.0, 
                    learning_rate=3e-4, lr_decay_rate= 0.9997, term_ic=None):
        super().__init__()
        self.save_hyperparameters()
        self.term_ic = term_ic
        self.model = DPGConvSeq(vocab_size, num_classes, num_filters, max_kernel, max_len)
        self.lr = learning_rate
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
        return {"loss": loss, "embeddings":embedding.detach(), "labels": sparse.csr_matrix(batch["labels"].cpu().numpy())}
    
    def training_epoch_end(self, outputs):
        labels, embeddings = [], []
        for output in outputs:
            labels.append(output["labels"])
            embeddings.append(output["embeddings"])
        self.train_labels = sparse.vstack(labels)
        self.train_embeddings = torch.cat(embeddings, dim=0)
        return super().training_epoch_end(outputs)
        
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
        output = {"loss": loss, "label_loss": label_loss, "metric_loss": metric_loss, 
                "labels": batch["labels"], "logits": logits, "embeddings": embedding}
        return output

    def validation_epoch_end(self, outputs):
        labels, logits, embeddings = [], [], []
        for output in outputs:
            labels.append(output["labels"])
            logits.append(output["logits"])
            embeddings.append(output["embeddings"])
        labels = torch.cat(labels, dim=0).cpu().numpy()
        logits = torch.cat(logits, dim=0)

        preds = (logits > 0).cpu().numpy()
        f1 = f1_score(labels, preds, average='micro')
        self.log('F1/val', f1, prog_bar=True)

        val_embeddings = torch.cat(embeddings, dim=0)
        self.val_embeddings = val_embeddings
        self.val_labels = labels
        return super().validation_epoch_end(outputs)

    def on_train_epoch_end(self):
        train_embeddings = self.train_embeddings.to(self.device)
        val_embeddings = self.val_embeddings.to(self.device)
        val_preds = embedding_knn(train_embeddings, val_embeddings, self.train_labels, k=3).toarray() >= 0.3
        knn_f1 = f1_score(self.val_labels, val_preds, average='micro')
        self.log('knn_F1/val', knn_f1, prog_bar=True)
        self.train_embeddings, self.train_labels, self.val_embeddings, self.val_labels = None, None, None, None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def on_train_start(self):
        # Proper logging of hyperparams and metrics in TB
        self.logger.log_hyperparams(self.hparams, {"loss/train": 1, "loss/val": 1})

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument('--vocab_size', type=int, default=30)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--max_len', type=int, default=1024)

        parser.add_argument('--num_classes', type=int, default=865)
        parser.add_argument('--num_filters', type=int, default=512)
        parser.add_argument('--max_kernel', type=int, default=129)
        parser.add_argument("--bottleneck_size", type=int, default=128)
        parser.add_argument("--bottleneck_regularization", type=float, default=0.0)

        parser.add_argument("--bottleneck_layers", type=int, default=1)
        parser.add_argument("--classification_layers", type=int, default=1)

        parser.add_argument('--sim_margin', type=float, default=3.0)
        parser.add_argument('--tmargin', type=float, default=1.0)

        parser.add_argument('--label_loss_weight', type=float, default=10.0)
        parser.add_argument('--learning_rate', type=float, default=0.005)
        parser.add_argument('--lr_decay_rate', type=float, default=0.9997)

        parser.add_argument('--gradient_clipping_decay', type=float, default=1.0)
        
        return parent_parser


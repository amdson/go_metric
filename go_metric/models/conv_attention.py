import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import f1_score

#Model notes
"""
https://arxiv.org/pdf/2107.11626.pdf
Loss 1: BCE
Loss 2: 
Label Level Embedding Network
x_i: Input (AxS)
r_i: Enc(x_i) (C x S)
U: (L x C)
g_i: Label level representations MultiAttBlock(U, r, r) (L x D)
f^j_c: Label classifier
s_ij: sig(f^j_c*g_ij))
Proj: Projection network
z_ij: Proj(g_ij), label level embeddings

"""


def conv1d_xiaver_init(conv):
    nn.init.xavier_normal_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv

class BaseFilter(nn.Module):
    def __init__(self, vocab_size, kernel_size, nb_filters):
        super().__init__()
        self.conv = nn.Conv1d(vocab_size, nb_filters, kernel_size, padding='same')
        conv1d_xiaver_init(self.conv)
        self.relu = nn.ReLU()

    def forward(self, one_hot):
        x = self.conv(one_hot)
        x = self.relu(x)
        return x


class ParLinear(nn.Module):
    def __init__(self, embed_size, num_classes):
        super().__init__()
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.mat = nn.Parameter(torch.normal(0, 1. / np.sqrt(self.embed_size), (1, self.num_classes, self.embed_size)))
        self.bias = nn.Parameter(torch.zeros((1, self.num_classes)))

    #x is (batch_size x num_classes x embed_size)
    def forward(self, x):
        a = torch.mul(self.mat, x) #(B x L x Emb)
        a = torch.sum(a, axis=2)
        return a + self.bias

class ConvAttention(nn.Module):
    def __init__(self, go_term_list, vocab_size, nb_filters, max_kernel, embed_dim):
        super().__init__()
        #Conv Embedding
        self.conv1 = nn.Conv1d(vocab_size, 128, 3, padding='same')
        self.dropout = nn.Dropout(p=0.25)
        kernel_sizes = list(range(3, max_kernel, 8))
        nets = []
        for kernel_size in kernel_sizes:
            bf = BaseFilter(128, kernel_size, nb_filters)
            nets.append(bf)
        self.nets = nn.ModuleList(nets)
        self.relu = nn.ReLU()
        self.convf = nn.Conv1d(len(kernel_sizes)*nb_filters, 512, 1)
        self.repr_dim = 512
        self.embed_dim = embed_dim
        self.num_classes = len(go_term_list)

        print(f"repr_dim={self.repr_dim}, embed_dim={self.embed_dim}, num_classes={self.num_classes}")

        self.class_query = nn.Parameter(torch.normal(0, 1/np.sqrt(self.repr_dim), (1, self.num_classes, self.repr_dim)))

        self.cross_attention = nn.MultiheadAttention(self.repr_dim, 8, batch_first=True)
        self.classifier_layer = ParLinear(self.repr_dim, self.num_classes)

        self.proj1 = nn.Linear(self.repr_dim, self.repr_dim)
        self.proj2 = nn.Linear(self.repr_dim, self.embed_dim)

        self.vocab_size = vocab_size

        self.go_term_list = go_term_list

    def forward_label_repr(self, x, return_attention=False):
        batch_size = x.shape[0]
        out = nn.functional.one_hot(x, num_classes=self.vocab_size).float() #(N, S, C)
        out = out.transpose(1, 2) #(N, C, S)
        out = self.conv1(out) 
        out = self.relu(out) #(N, 128, S)
        outputs = [bf(out) for bf in self.nets]
        out = torch.cat(outputs, axis=1) #(N, repr_dim, S)
        out = self.dropout(out) 
        out = self.convf(out)
        out = self.relu(out)
        g = out.transpose(1, 2) #(N, S, repr_dim)
        U = self.class_query.expand((batch_size, -1, -1))
        z = self.cross_attention(U, g, g, need_weights=return_attention) #(N, L, repr_dim)
        if(return_attention):
            return z
        return z[0]

    def forward(self, x, return_label_emb=False):
        z = self.forward_label_repr(x) #(N, L, repr_dim)
        s = self.classifier_layer(z).squeeze() #(N, L)
        if(return_label_emb):
            batch_size, L, repr_dim = z.shape
            z = torch.reshape(z, (-1, repr_dim))
            label_emb = self.proj1(z)
            label_emb = self.relu(label_emb)
            label_emb = self.proj2(z)
            label_emb = torch.reshape(label_emb, (batch_size, L, -1)) #(N, L, emb_size)
            return (s, label_emb)
        else:
            return s

class ConvAttentionModule(pl.LightningModule):
    def __init__(self, go_term_list, hparams):
        super(ConvAttentionModule, self).__init__()
        self.save_hyperparameters(hparams)
        self.h = hparams
        h = hparams
        self.model = ConvAttention(go_term_list, h.vocab_size, h.nb_filters, h.max_kernel, h.embed_dim)
        self.bce_loss = nn.BCEWithLogitsLoss()

    @classmethod
    def from_memory(cls, go_term_path, hparams):
        import json
        with open(go_term_path) as f:
            go_terms = json.load(f)
        return ConvAttentionModule(go_terms, hparams) 

    def forward(self, input_ids):
        """ Usual pytorch forward function.
        :param input_ids: text sequences [batch_size x src_seq_len]
        :param go_labels: (Integer Tensor) go_id for each sequence [batch_size]
        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        return self.model(input_ids)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch["seq"], batch["labels"] 
        # go_class = batch["go_class"][0]

        logits = self.model(x)
        bce_loss = self.bce_loss(logits, y.float())
        return bce_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["seq"], batch["labels"] 
        logits = self.model(x)
        loss = self.bce_loss(logits, y.float())

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('loss_val', loss, prog_bar=True)
        #self.log('val_acc', acc, prog_bar=True)
        return {"logits": logits.detach().cpu().numpy(), "labels": y.detach().cpu().numpy()}

    def validation_epoch_end(self, outputs):
        labels, logits = [], []
        for output in outputs:
            labels.append(output["labels"])
            logits.append(output["logits"])
        labels = np.concatenate(labels, axis=0)
        logits = np.concatenate(logits, axis=0)
        thresholds = [0]
        f1 = 0
        for threshold in thresholds:
            preds = logits > threshold
            f1 = max(f1, f1_score(labels, preds, average='micro'))
        self.log('F1/val', f1, prog_bar=True)
        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.h.learning_rate)
        return optimizer
    
    def on_train_start(self):
        # Proper logging of hyperparams and metrics in TB
        self.logger.log_hyperparams(self.h, {"loss_train": 1, "loss_val": 1})

    @classmethod
    def add_model_specific_args(
        cls, parser):
        """ Parser for Estimator specific arguments/hyperparameters. 
        :param parser: HyperOptArgumentParser obj
        Returns:
            - updated parser
        """
        parser.add_argument("--max_length", default=1024, type=int)
        parser.add_argument("--vocab_size", default=30, type=int)
        parser.add_argument("--nb_filters", default=256, type=int)
        parser.add_argument("--max_kernel", default=128, type=int)
        parser.add_argument("--embed_dim", default=64, type=int)
        parser.add_argument("--learning_rate", default=3e-04, type=float)
        parser.add_argument("--contrastive_weight", default=0.05, type=float)
        parser.add_argument("--max_contrastive_weight", default=0.2, type=float)
        parser.add_argument("--bce_pretraining_epochs", default=12, type=int)
        return parser
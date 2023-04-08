import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from go_metric.utils import tuple_type
from go_metric.metric_loss import metric_logits_loss, multilabel_triplet_loss
from scipy import sparse

class AddPositionalEncoding(nn.Module):
    def __init__(self, d_model=256, input_dropout=0.1, timing_dropout=0.1,
                max_len=2048):
        super().__init__()
        self.timing_table = nn.Parameter(torch.FloatTensor(max_len, d_model))
        nn.init.normal_(self.timing_table)
        self.input_dropout = nn.Dropout(input_dropout)
        self.timing_dropout = nn.Dropout(timing_dropout)
  
    def forward(self, x):
        """
        Args:
            x: A tensor of shape [batch size, length, d_model]
        """
        x = self.input_dropout(x)
        timing = self.timing_table[None, :x.shape[1], :]
        timing = self.timing_dropout(timing)
        return x + timing

class TARSModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, term_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.term_proj = nn.Linear(term_size, embedding_dim)
        self.pos_encoding = AddPositionalEncoding(d_model=embedding_dim)
        num_layers = 6
        n_head = 4
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, n_head, dim_feedforward=embedding_dim*4, 
                                         dropout=0.1, activation='relu', batch_first=True, norm_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier_layer = nn.Linear(2*embedding_dim, 2)
        self.vocab_size = vocab_size

    def forward(self, x, mask, term_emb):
        embedding = self.emb(x) # (B, L, D)
        embedding = self.pos_encoding(embedding)
        B, L, D = embedding.shape
        term_emb = self.term_proj(term_emb).view(B, 1, D) # (B, D)
        embedding = torch.cat((term_emb, embedding), dim=1)
        mask = torch.cat([torch.zeros((B, 1), device=mask.device, dtype=torch.bool), mask], dim=1)
        embedding = self.transformer_encoder(embedding, src_key_padding_mask=mask)
        # print(embedding.shape, mask.shape)
        # embedding_mean = (embedding*(~mask).view(B, L+1, 1)).sum(dim=1) / (~mask).sum(dim=1, keepdim=True)
        embedding_mean = torch.mean(embedding, dim=1)
        embedding_cls = embedding[:, 0, :]
        out_emb = torch.cat([embedding_mean, embedding_cls], dim=1) #(B, 2*D)
        logits = self.classifier_layer(out_emb)
        return logits
    
class TARSModule(pl.LightningModule):
    def __init__(self, term_emb=None, vocab_size=30, embedding_dim=256, term_size = 50, max_len = 1024, batch_size=256,
                    learning_rate=5e-4, git_hash=None):
        super().__init__()
        self.save_hyperparameters()
        self.term_emb = term_emb
        self.model = TARSModel(vocab_size, embedding_dim, term_size)
        self.lr = learning_rate
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, mask, term_emb):
        return self.model(x, mask, term_emb)
    
    def on_fit_start(self) -> None:
        self.term_emb = self.term_emb.to(self.device)

    def on_train_epoch_start(self):
        print('\n')

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        x, y, mask = batch["seq"], batch["labels"].long(), batch['mask']
        target_term = batch['target_term'] #(B,)
        term_emb = self.term_emb[target_term, :]

        logits = self.model.forward(x, ~mask, term_emb) #(B, 2)
        # print(logits.shape, y.shape)
        # print("y", y)
        loss = self.loss(logits, y)
        # Logging to TensorBoard by default
        self.log('loss/train', loss, prog_bar=True)
        return {"loss": loss}
        
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        x, y, mask = batch["seq"], batch["labels"].long(), batch['mask']
        target_term = batch['target_term'] #(B,)
        term_emb = self.term_emb[target_term, :]

        logits = self.model.forward(x, ~mask, term_emb) #(B, 2)
        # print(logits.shape, y.shape)
        # print("y", y)
        loss = self.loss(logits, y)
        # loss = 0
        # Logging to TensorBoard by default
        self.log('loss/val', loss, prog_bar=True)
        return {"loss": loss}

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
        parser.add_argument('--embedding_dim', type=int, default=256)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--max_len', type=int, default=1024)
        parser.add_argument('--learning_rate', type=float, default=5e-4)
        # parser.add_argument('--lr_decay_rate', type=float, default=0.9997)
        return parent_parser
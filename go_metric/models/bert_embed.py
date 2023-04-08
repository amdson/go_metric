from typing import Tuple
import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import f1_score
import pytorch_lightning as pl
from transformers import BertTokenizer, BertModel

import pandas as pd
import os
import re
from collections import OrderedDict
import logging as log
import numpy as np

class ProtBertBFDRegression(pl.LightningModule):
    """
    # https://github.com/minimalist-nlp/lightning-text-classification.git
    
    Sample model to show how to use BERT to classify sentences.
    """

    def __init__(self, hparams) -> None:
        super(ProtBertBFDRegression, self).__init__()
        self.h = hparams
        self.loss = nn.MSELoss()

        self.model_name = "Rostlab/prot_bert_bfd"

        # build model
        self.__build_model()

        self.freeze_encoder()
        # if self.h.nr_frozen_epochs <= 0:
        #     self.unfreeze_encoder() #Unfreeze layers that should be trained
        # else:
        #     self._frozen = False
        # self.nr_frozen_epochs = self.h.nr_frozen_epochs

    def __build_model(self) -> None:
        """ Init BERT model + tokenizer + classification head."""
        self.ProtBertBFD = BertModel.from_pretrained(self.model_name)
        self.encoder_features = 1024
        
        # Classification head
        self.regression_head = nn.Sequential(
            nn.Linear(self.encoder_features*1, self.h.num_classes),
        )

    # def unfreeze_encoder(self, frozen_layers=14) -> None:
    #     """ un-freezes the encoder layer. """
    #     if self._frozen:
    #         log.info(f"\n-- Encoder model fine-tuning")
    #         # for param in self.ProtBertBFD.parameters():
    #         #     param.requires_grad = True
    #         for n, param in self.ProtBertBFD.named_parameters():
    #             if("layer" in n):
    #                 if(int(re.search("layer\.(\d+)", n).group(1)) >= frozen_layers):
    #                     param.requires_grad = True
    #             elif("pooler" in n):
    #                 param.requires_grad = True
    #         self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.ProtBertBFD.parameters():
            param.requires_grad = False
        self._frozen = True
    
    def pool_strategy(self, features):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pool_mean = sum_embeddings / sum_mask
        return pool_mean

        # output_vector = torch.cat([cls_token, pool_mean], 1)
        # return output_vector
    
    def forward(self, input_ids, attention_mask):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]
        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        # input_ids = torch.tensor(input_ids, device=self.device)
        # attention_mask = torch.tensor(attention_mask,device=self.device)

        word_embeddings = self.ProtBertBFD(input_ids,
                                           attention_mask)[0]

        pooling = self.pool_strategy({"token_embeddings": word_embeddings,
                                      "cls_token_embeddings": word_embeddings[:, 0],
                                      "attention_mask": attention_mask,
                                      })
        return self.regression_head(pooling)

    def training_step(self, batch) -> dict:
        """ 
        Runs one training step. This usually consists in the forward function followed
            by the loss function.
        
        :param batch: The output of your dataloader. 
        :param batch_nb: Integer displaying which batch this is
        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        input_ids, mask, y = batch['seq'], batch['mask'], batch['rnc']
        y_hat = self.forward(input_ids, mask)
        loss_val = self.loss(y_hat, y)

        self.log("loss/train", loss_val, prog_bar=True)

        # current_lr = self.scheduler.get_last_lr()[0]
        # self.log("lr", current_lr, prog_bar=True, on_step=True)

        # can also return just a scalar instead of a dict (return loss_val)
        return loss_val

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, mask, y = batch['seq'], batch['mask'], batch['rnc']
        y_hat = self.forward(inputs, mask)
        loss_val = self.loss(y_hat, y.float())
        self.log("loss/val", loss_val, prog_bar=True)
        output = OrderedDict({'logits':y_hat.detach().cpu(), 'labels':y.detach().cpu()})
        return output

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.regression_head.parameters()},
            {
                "params": self.ProtBertBFD.parameters(),
                "lr": self.h.encoder_learning_rate,
            },
        ]
        optimizer = optim.Adam(parameters, lr=self.h.learning_rate, weight_decay=self.h.weight_decay)
        # def lr_lambda(step):
        #     warmup_steps = 7000
        #     pr = step / warmup_steps
        #     if(pr < 1):
        #         return pr
        #     else: 
        #         return max(0.001, 1-pr*0.1)
        # lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        # self.scheduler = lr_scheduler
        # lr_scheduler_config = {
        #     "scheduler": lr_scheduler,
        #     "interval": "step",
        #     "frequency": 1,
        #     "strict": False,
        # }
        # return {"optimizer": optimizer, "lr_scheduler":lr_scheduler_config}
        return optimizer

    # def on_epoch_end(self):
    #     """ Pytorch lightning hook """
    #     if self.current_epoch + 1 >= self.nr_frozen_epochs:
    #         self.unfreeze_encoder()
    #     print("Unfroze layers")
    #     print(self.ProtBertBFD)

    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument(
            "--max_length",
            default=1024,
            type=int,
            help="Maximum sequence length.",
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default=1e-04,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=3e-05,
            type=float,
            help="regression head learning rate.",
        )
        parser.add_argument(
            "--nr_frozen_epochs",
            default=0,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen."
        )
        parser.add_argument("--gradient_checkpointing", default=True, type=bool, 
            help="Enable or disable gradient checkpointing which use the cpu memory \
                with the gpu memory to store the model.",
        )
        parser.add_argument(
            "--gradient_clipping", default=1.0, type=float, help="Global norm gradient clipping"
        )
        parser.add_argument(
            "--weight_decay", default=0.0, type=float, help="Weight decay per train step."
        )
        return parser
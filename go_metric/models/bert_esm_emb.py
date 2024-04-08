from typing import Tuple
import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import f1_score
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel

import pandas as pd
import os
import re
from collections import OrderedDict
import logging as log
import numpy as np

class ESMBERTClassifier(pl.LightningModule):
    """
    # https://github.com/minimalist-nlp/lightning-text-classification.git
    
    Sample model to show how to use BERT to classify sentences.
    
    :param hparams: ArgumentParser containing the hyperparameters.
    hparams: {
        nr_frozen_epochs,
        batch_size,
        num_classes,
        gradient_checkpointing,
    }
    """
    def __init__(self, hparams) -> None:
        super(ESMBERTClassifier, self).__init__()
        self.h = hparams
        self.model_name = self.h.model_name
        self.save_hyperparameters()
        # build model
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()
        # self.freeze_encoder()
        # if self.h.nr_frozen_epochs <= 0:
        #     self.unfreeze_encoder() #Unfreeze layers that should be trained
        # else:
        #     self._frozen = False
        self.nr_frozen_epochs = self.h.nr_frozen_epochs
        self._train_dataset_generated = False

    def __build_model(self) -> None:
        """ Init BERT model + tokenizer + classification head."""
        self.model = AutoModel.from_pretrained(self.model_name)
        print(self.model)
        self.encoder_features = self.model.config.hidden_size
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False)
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.encoder_features, self.h.num_classes),
        )
        
    def __build_loss(self):
        """ Initializes the loss function/s. """
        self._loss = nn.BCEWithLogitsLoss()

    # https://github.com/UKPLab/sentence-transformers/blob/eb39d0199508149b9d32c1677ee9953a84757ae4/sentence_transformers/models/Pooling.py
    def pool_strategy(self, features,
                      pool_cls=True, pool_max=True, pool_mean=True,
                      pool_mean_sqrt=True):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        if pool_cls:
            output_vectors.append(cls_token)
        if pool_max:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if pool_mean or pool_mean_sqrt:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if pool_mean:
                output_vectors.append(sum_embeddings / sum_mask)
            if pool_mean_sqrt:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.stack(output_vectors, -1).sum(dim=-1)
        # output_vector = torch.cat(output_vectors, 1)
        return output_vector
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]
        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        # input_ids = torch.tensor(input_ids, device=self.device)
        # attention_mask = torch.tensor(attention_mask,device=self.device)

        word_embeddings = self.model(input_ids,
                                           attention_mask)[0]

        pooling = self.pool_strategy({"token_embeddings": word_embeddings,
                                      "cls_token_embeddings": word_embeddings[:, 0],
                                      "attention_mask": attention_mask,
                                      }, pool_max=False, pool_mean_sqrt=False)
        return self.classification_head(pooling)
    
    def forward_emb(self, input_ids, token_type_ids, attention_mask):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]
        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        word_embeddings = self.model(input_ids,
                                           attention_mask)[0]
        pooling = self.pool_strategy({"token_embeddings": word_embeddings,
                                      "cls_token_embeddings": word_embeddings[:, 0],
                                      "attention_mask": attention_mask,
                                      }, pool_max=False, pool_mean_sqrt=False)
        return self.classification_head(pooling), pooling

    def loss(self, predictions, targets) -> torch.tensor:
        return self._loss(predictions, targets)

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ 
        Runs one training step. This usually consists in the forward function followed
            by the loss function.
        
        :param batch: The output of your dataloader. 
        :param batch_nb: Integer displaying which batch this is
        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs, mask, y = batch['seq'], batch['mask'], batch['labels']
        y_hat = self.forward(inputs, None, mask)
        loss_val = self.loss(y_hat, y.float())

        current_lr = self.scheduler.get_last_lr()[0]
        self.log("lr", current_lr, prog_bar=True, on_step=True)

        tqdm_dict = {"train_loss": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict})

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def on_validation_epoch_start(self) -> None:
        self.val_outputs = []
        return super().on_validation_epoch_start()

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, mask, y = batch['seq'], batch['mask'], batch['labels']
        y_hat = self.forward(inputs, None, mask)
        loss_val = self.loss(y_hat, y.float())
        self.log("loss/val", loss_val)
        output = OrderedDict({'logits':y_hat.detach().cpu(), 'labels':y.detach().cpu()})
        self.val_outputs.append(output)
        return output
        
    def on_validation_epoch_end(self) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        
        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        outputs = self.val_outputs
        labels = torch.cat([x['labels'] for x in outputs])
        logits = torch.cat([x['logits'] for x in outputs])
        thresholds = [-3, -1, -0.5, 0, 0.5, 1, 3]
        f1 = 0
        for threshold in thresholds:
            preds = logits > threshold
            f1 = max(f1, f1_score(labels, preds, average='micro'))
        print("max f1", f1)
        self.log('F1/val', f1, prog_bar=True)

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.classification_head.parameters()
             },
            {
                "params": self.model.parameters(),
                "lr": self.h.encoder_learning_rate,
            },
        ]
        optimizer = optim.Adam(parameters, lr=self.h.learning_rate, weight_decay=self.h.weight_decay)
        def lr_lambda(step):
            warmup_steps = 7000
            pr = step / warmup_steps
            if(pr < 1):
                return pr
            else: 
                return max(0.01, 1-pr*0.1)
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        self.scheduler = lr_scheduler
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": False,
        }
        return {"optimizer": optimizer, "lr_scheduler":lr_scheduler_config}

    # def on_epoch_end(self):
    #     """ Pytorch lightning hook """
    #     if self.current_epoch + 1 >= self.nr_frozen_epochs:
    #         self.unfreeze_encoder()
    #     print("Unfroze layers")
    #     print(self.model)

    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument(
            "--model_name",
            default="facebook/esm2_t33_650M_UR50D",
            type=str,
            help="Maximum sequence length.",
        )
        parser.add_argument(
            "--max_length",
            default=1024,
            type=int,
            help="Maximum sequence length.",
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default=1e-05,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=9e-05,
            type=float,
            help="Classification head learning rate.",
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
            "--weight_decay", default=0.00, type=float, help="Weight decay per train step."
        )
        return parser
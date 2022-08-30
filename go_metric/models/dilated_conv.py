import torch
import torch.nn as nn
import math
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import f1_score

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, dilation=dilation, padding='same')

    def forward(self, x, mask):
        mask = mask.unsqueeze(1)
        x = self.conv1(x)
        x = x.masked_fill(~mask, 0)
        return x
        
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, dilation_rate, bottleneck_factor, kernel_size,
                    activation_fn):
        super().__init__()
        self.activation_fn = activation_fn
        num_bottleneck_units = math.floor(in_channels*bottleneck_factor)
        self.b1 = nn.BatchNorm1d(in_channels) #N, C, L
        self.b2 = nn.BatchNorm1d(num_bottleneck_units)
        self.conv1 = ConvLayer(in_channels, num_bottleneck_units, dilation_rate, kernel_size)
        self.conv2 = ConvLayer(num_bottleneck_units, in_channels, 1, 1)

    def forward(self, x, mask):
        input_x = x
        x = self.b1(x)
        x = self.activation_fn(x)
        x = self.conv1(x, mask)
        x = self.b2(x)
        x = self.activation_fn(x)
        x = self.conv2(x, mask)
        x = x + input_x
        return x

class DilatedResNet(nn.Module):
    def __init__(self, input_dim, num_filters, kernel_size, 
                    bottleneck_factor, dilation_rate, 
                    num_layers, first_dilated_layer):
        super().__init__()
        self.conv1 = ConvLayer(input_dim, num_filters, dilation=1, kernel_size=kernel_size)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            shifted_layer_index = i - first_dilated_layer + 1
            dilation = max(1, dilation_rate**shifted_layer_index)
            self.layers.append(ResidualBlock(num_filters, dilation, 
                                bottleneck_factor, kernel_size,
                                activation_fn=nn.ReLU()))
    def forward(self, x, mask):
        x = self.conv1(x, mask)
        for m in self.layers:
            x = m.forward(x, mask)
        return x

class DilatedConvSeq(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.embedding = nn.Embedding(hparams.vocab_size, hparams.input_dim)
        self.resnet = DilatedResNet(hparams.input_dim, hparams.num_filters, hparams.kernel_size, 
                                    hparams.bottleneck_factor, hparams.dilation_rate, 
                                    hparams.num_layers, hparams.first_dilated_layer)
        self.classifier_layer = nn.Linear(hparams.num_filters, hparams.num_classes)

    def forward(self, seq, mask):
        x = self.embedding(seq).transpose(1, 2)
        x = self.resnet(x, mask)
        x = torch.max(x, dim=2, keepdim=False)[0]
        logits = self.classifier_layer(x)
        return logits

class DilatedConvModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = DilatedConvSeq(hparams)
        self.lr = hparams.learning_rate
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def on_epoch_start(self):
        print('\n')

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, mask, y = batch["seq"], batch['mask'], batch["labels"] 
        logits = self.model(x, mask)
        loss = self.loss(logits, y.float())
        # Logging to TensorBoard by default
        self.log('loss/train', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, mask, y = batch["seq"], batch['mask'], batch["labels"] 
        logits = self.model(x, mask)
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
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--input_dim", type=int, default=128)
        parser.add_argument("--num_layers", type=int, default=8)
        parser.add_argument('--gradient_clipping_decay', type=float, default=1.0)
        parser.add_argument('--batch_size', type=int, default=156)
        parser.add_argument('--dilation_rate', type=float, default=2)
        parser.add_argument('--num_filters', type=int, default=1600)
        parser.add_argument('--first_dilated_layer', type=int, default=4)  # This is 0-indexed
        parser.add_argument('--kernel_size', type=int, default=9)
        parser.add_argument('--max_len', type=int, default=1024)
        parser.add_argument('--num_classes', type=int, default=865)
        parser.add_argument('--vocab_size', type=int, default=30)
        parser.add_argument('--pooling', type=str, default='max')
        parser.add_argument('--bottleneck_factor', type=float, default=0.5)
        parser.add_argument('--lr_decay_rate', type=float, default=0.9997)
        parser.add_argument('--learning_rate', type=float, default=0.0005)
        return parent_parser
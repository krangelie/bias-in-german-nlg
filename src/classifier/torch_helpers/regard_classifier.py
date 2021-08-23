import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from pytorch_lightning.metrics.functional import accuracy, f1


class RegardClassifier(pl.LightningModule):
    def __init__(self, n_embed, n_hidden_lin, n_output, lr, weight_vector, drop_p):
        super(RegardClassifier, self).__init__()
        self.save_hyperparameters()
        self.n_embed = n_embed
        self.n_hidden_lin = n_hidden_lin
        self.n_output = n_output
        self.lr = lr
        self.weight_vector = weight_vector

    def forward(self, *args, **kwargs):
        pass

    def training_step(self, batch, batch_idx):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # training_step defined the train loop. It is independent of forward
        inputs, labels = batch
        inputs, labels = inputs.float().to(device), labels.long().to(device)
        output = self(inputs)
        if isinstance(output, tuple):
            output = output[0]
        loss = F.cross_entropy(output, labels, weight=self.weight_vector)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # training_step defined the train loop. It is independent of forward
        inputs, labels = batch
        inputs, labels = inputs.float().to(device), labels.long().to(device)
        outputs = self(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = F.cross_entropy(outputs, labels)

        probs = F.log_softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        acc = accuracy(preds, labels)
        f1_score = f1(preds, labels, num_classes=self.n_output, average="macro")

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1_score, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # training_step defined the train loop. It is independent of forward
        inputs, labels = batch
        inputs, labels = inputs.float().to(device), labels.long().to(device)
        outputs = self(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = F.cross_entropy(outputs, labels)

        probs = F.log_softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        acc = accuracy(preds, labels)
        f1_score = f1(preds, labels, num_classes=self.n_output, average="macro")

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1", f1_score, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, eps=1e-8)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    optimizer, mode="min", patience=5
        # )
        return {
            "optimizer": optimizer,
            #     "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

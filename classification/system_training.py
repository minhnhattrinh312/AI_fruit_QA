import numpy as np
import torch
import lightning
import timm.optim
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Classifier(lightning.LightningModule):

    def __init__(self, model, learning_rate, factor_lr, patience_lr):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        # torch 2.3 => compile to make faster
        self.model = torch.compile(self.model, mode="reduce-overhead")

        self.learning_rate = learning_rate
        self.factor_lr = factor_lr
        self.patience_lr = patience_lr
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, label = batch
        pred = self.model(data)
        loss_mse = MSELoss()(pred, label)
        metrics = {'train_loss': loss_mse}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return loss_mse

    def validation_step(self, batch, batch_idx):
        data, label = batch
        y_pred = self.model(data)
        output_dict = {
            "batch_y_true": label,
            "batch_y_pred": y_pred.clone(),
        }
        self.validation_step_outputs.append(output_dict)
    def on_validation_epoch_end(self):
        y_true = torch.cat([x["batch_y_true"] for x in self.validation_step_outputs], dim=0)
        prediction = torch.cat([x["batch_y_pred"] for x in self.validation_step_outputs], dim=0)
        loss_mse = MSELoss()(prediction, y_true)
        metrics = {'val_loss': loss_mse}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def configure_optimizers(self):
        optimizer = timm.optim.Nadam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=self.factor_lr, patience=self.patience_lr)
        lr_schedulers = {
            "scheduler": scheduler,
            "monitor": "val_loss",
            "strict": False,
            
        }

        return [optimizer], lr_schedulers

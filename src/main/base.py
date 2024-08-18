import torch
from torch import nn
import pytorch_lightning as pl

__all__ = ['Main', ]


class Main(pl.LightningModule):

    def __init__(
        self,
        model,
        optimizer_cls,
        optimizer_args=None,
        scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args=None,
        scheduler_monitor='val_loss',
        loss_fn=nn.CrossEntropyLoss(),
        target=None,
        bias_no_w_decay=False
    ):
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_args
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_args
        self.schedule_monitor = scheduler_monitor
        self.loss_fn = loss_fn
        self.target = target
        self.bias_no_w_decay = bias_no_w_decay
        self.output_file = "test_table"
        self.output_predict_result = False

        self.lr = optimizer_args["lr"]
        self.save_hyperparameters()

    def forward(self, inputs):
        results = self.model(inputs)
        return results

    def training_step(self, batch, batch_idx):
        target = batch[self.target]

        pred = self(batch)
        loss = self.loss_fn(pred, target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        target = batch[self.target]

        pred = self(batch)
        loss = self.loss_fn(pred, target)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        target = batch[self.target]

        pred = self(batch)
        loss = self.loss_fn(pred, target)

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return {"test_loss": loss}

    def configure_optimizers(self):
        if self.optimizer_cls:
            if self.bias_no_w_decay:
                weight_p, bias_p = [], []

                for p in self.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)

                for name, p in self.named_parameters():
                    if 'bias' in name:
                        bias_p += [p]
                    else:
                        weight_p += [p]

                weight_decay = self.optimizer_kwargs['weight_decay']
                self.optimizer_kwargs.pop('weight_decay', None)
                optimizer = self.optimizer_cls(
                    params=[
                                {'params': weight_p, 'weight_decay': weight_decay},
                                {'params': bias_p, 'weight_decay': 0}
                            ],
                    **self.optimizer_kwargs
                )
            else:
                optimizer = self.optimizer_cls(
                    params=self.parameters(), **self.optimizer_kwargs
                )

        if self.scheduler_cls:
            schedulers = []
            schedule = self.scheduler_cls(optimizer=optimizer, **self.scheduler_kwargs)
            optimconf = {"scheduler": schedule, "name": "lr_schedule"}
            if self.schedule_monitor:
                optimconf["monitor"] = self.schedule_monitor
            schedulers.append(optimconf)
            return [optimizer], schedulers
        else:
            return optimizer

    def save_model(self, path: str):
        torch.save(self.model, path)

    def output_result_file(self, name: str = None):
        self.output_predict_result = True
        if name:
            self.output_file = name

        self.test_step_outputs = []


from torch import nn
from torchmetrics import Metric
import torch
from typing import List
from typing import Optional
from typing import Dict
import warnings
from typing import Type
from typing import Any
import pandas as pd
from src.model.schnetpack import AtomisticModel
import pytorch_lightning as pl
from torchmetrics.regression import R2Score
from src.model.schnetpack import BsAtomisticModel

__all__ = ['ModelOutput', 'AtomisticTask', 'UnsupervisedModelOutput', 'BsAtomisticTask', ]


class ModelOutput(nn.Module):
    """
    Defines an output of a model, including mappings to a loss function and weight for training
    and metrics to be logged.
    """

    def __init__(
        self,
        name: str,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        constraints: Optional[List[torch.nn.Module]] = None,
        target_property: Optional[str] = None,
    ):
        """
        Args:
            name: name of output in results dict
            target_property: Name of target in training batch. Only required for supervised training.
                If not given, the output name is assumed to also be the target name.
            loss_fn: function to compute the loss
            loss_weight: loss weight in the composite loss: $l = w_1 l_1 + \dots + w_n l_n$
            metrics: dictionary of metrics with names as keys
            constraints:
                constraint class for specifying the usage of model output in the loss function and logged metrics,
                while not changing the model output itself. Essentially, constraints represent postprocessing transforms
                that do not affect the model output but only change the loss value. For example, constraints can be used
                to neglect or weight some atomic forces in the loss function. This may be useful when training on
                systems, where only some forces are crucial for its dynamics.
        """
        super().__init__()
        self.name = name
        self.target_property = target_property or name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.metrics = nn.ModuleDict(metrics)
        self.constraints = constraints or []

    def calculate_loss(self, pred, target):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0

        loss = self.loss_weight * self.loss_fn(
            pred[self.name], target[self.target_property]
        )
        return loss

    def calculate_metrics(self, pred, target):
        metrics = {
            metric_name: metric(pred[self.name], target[self.target_property])
            for metric_name, metric in self.metrics.items()
        }
        return metrics



class UnsupervisedModelOutput(ModelOutput):
    """
    Defines an unsupervised output of a model, i.e. an unsupervised loss or a regularizer
    that do not depend on label data. It includes mappings to the loss function,
    a weight for training and metrics to be logged.
    """

    def calculate_loss(self, pred, target=None):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0
        loss = self.loss_weight * self.loss_fn(pred[self.name])
        return loss

    def calculate_metrics(self, pred, target=None):
        metrics = {
            metric_name: metric(pred[self.name])
            for metric_name, metric in self.metrics.items()
        }
        return metrics




class AtomisticTask(pl.LightningModule):
    """
    The basic learning task in SchNetPack, which ties model, loss and optimizer together.

    """

    def __init__(
        self,
        model: AtomisticModel,
        outputs: List[ModelOutput],
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
        warmup_steps: int = 0,
    ):
        """
        Args:
            model: the neural network model
            outputs: list of outputs an optional loss functions
            optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
            optimizer_args: dict of optimizer keyword arguments
            scheduler_cls: type of torch learning rate scheduler
            scheduler_args: dict of scheduler keyword arguments
            scheduler_monitor: name of metric to be observed for ReduceLROnPlateau
            warmup_steps: number of steps used to increase the learning rate from zero
              linearly to the target learning rate at the beginning of training
        """
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_args
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_args
        self.schedule_monitor = scheduler_monitor
        self.outputs = nn.ModuleList(outputs)
        self.output_file = "test_table"
        self.output_predict_result = False
        self.rerun_start_on_epoch = None
        self.rerun_when_loss = None

        self.r2_score = R2Score()

        self.grad_enabled = len(self.model.required_derivatives) > 0
        self.lr = optimizer_args["lr"]
        self.warmup_steps = warmup_steps
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit":
            self.model.initialize_transforms(self.trainer.datamodule)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        results = self.model(inputs)
        return results

    def loss_fn(self, pred, batch):
        loss = 0.0
        for output in self.outputs:
            loss += output.calculate_loss(pred, batch)
        return loss

    def log_metrics(self, pred, targets, subset):
        for output in self.outputs:
            for metric_name, metric in output.calculate_metrics(pred, targets).items():
                if (subset == "val") and (metric_name == "mae"):
                    self.log(
                        f"{subset}_{output.name}_{metric_name}",
                        metric,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )
                else:
                    self.log(
                        f"{subset}_{output.name}_{metric_name}",
                        metric,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )

    def apply_constraints(self, pred, targets):
        for output in self.outputs:
            for constraint in output.constraints:
                pred, targets = constraint(pred, targets, output)
        return pred, targets

    def training_step(self, batch, batch_idx):

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)
        if self.rerun_start_on_epoch and self.rerun_when_loss:
            if self.current_epoch >= self.rerun_start_on_epoch and loss >= self.rerun_when_loss:
                self.trainer.should_stop = True
                self.trainer.should_rerun = True

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log_metrics(pred, targets, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pred, targets, "val")

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)
        if self.output_predict_result:
            self.test_step_outputs.append(
                {
                    "targets": targets[next(iter(pred.keys()))],
                    "pred": pred[next(iter(pred.keys()))],
                }
            )
        loss = self.loss_fn(pred, targets)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pred, targets, "test")
        return {"test_loss": loss}

    def predict_without_postprocessing(self, batch):
        pp = self.model.do_postprocessing
        self.model.do_postprocessing = False
        pred = self(batch)
        self.model.do_postprocessing = pp
        return pred

    def on_test_epoch_end(self):
        if self.output_predict_result:
            target, pred = None, None
            for out in self.test_step_outputs:
                if target is None:
                    target = out["targets"]
                else:
                    target = torch.cat((target, out["targets"]))
                if pred is None:
                    pred = out['pred']
                else:
                    pred = torch.cat((pred, out['pred']))
            output = [['targets', 'predicts']]
            r2 = self.r2_score(pred, target)
            print(f"\nR2 = {r2}\n")
            for i in range(target.size()[0]):
                output.append(
                    [float(target[i]), float(pred[i])]
                )
            pd.DataFrame(output).to_csv(f'results/{self.output_file}.csv', header=False, index=False)
            self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            params=self.parameters(), **self.optimizer_kwargs
        )

        if self.scheduler_cls:
            schedulers = []
            schedule = self.scheduler_cls(optimizer=optimizer, **self.scheduler_kwargs)
            optimconf = {"scheduler": schedule, "name": "lr_schedule"}
            if self.schedule_monitor:
                optimconf["monitor"] = self.schedule_monitor
            # incase model is validated before epoch end (not recommended use of val_check_interval)
            if self.trainer.val_check_interval < 1.0:
                warnings.warn(
                    "Learning rate is scheduled after epoch end. To enable scheduling before epoch end, "
                    "please specify val_check_interval by the number of training epochs after which the "
                    "model is validated."
                )
            # incase model is validated before epoch end (recommended use of val_check_interval)
            if self.trainer.val_check_interval > 1.0:
                optimconf["interval"] = "step"
                optimconf["frequency"] = self.trainer.val_check_interval
            schedulers.append(optimconf)
            return [optimizer], schedulers
        else:
            return optimizer

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer=None,
        optimizer_closure=None,
    ):
        if self.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def save_model(self, path: str, do_postprocessing: Optional[bool] = None):
        pp_status = self.model.do_postprocessing
        if do_postprocessing is not None:
            self.model.do_postprocessing = do_postprocessing

        torch.save(self.model, path)

        self.model.do_postprocessing = pp_status

    def output_result_file(self, name: str = None):
        self.output_predict_result = True
        if name:
            self.output_file = name

        self.test_step_outputs = []



class BsAtomisticTask(AtomisticTask):
    """
    The basic learning task in SchNetPack, which ties model, loss and optimizer together.

    """

    def __init__(
        self,
        model: BsAtomisticModel,
        outputs: List[ModelOutput],
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
        warmup_steps: int = 0,
    ):
        """
        Args:
            model: the neural network model
            outputs: list of outputs an optional loss functions
            optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
            optimizer_args: dict of optimizer keyword arguments
            scheduler_cls: type of torch learning rate scheduler
            scheduler_args: dict of scheduler keyword arguments
            scheduler_monitor: name of metric to be observed for ReduceLROnPlateau
            warmup_steps: number of steps used to increase the learning rate from zero
              linearly to the target learning rate at the beginning of training
        """
        super().__init__(
            model=model,
            outputs=outputs,
            optimizer_cls=optimizer_cls,
            optimizer_args=optimizer_args,
            scheduler_cls=scheduler_cls,
            scheduler_args=scheduler_args,
            scheduler_monitor=scheduler_monitor,
            warmup_steps=warmup_steps,
        )




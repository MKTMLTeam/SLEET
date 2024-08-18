import logging
from lightning.pytorch.callbacks import Callback
from typing import Union
from typing import List
from typing import Optional
from typing import Dict
from lightning.pytorch.trainer.connectors.accelerator_connector import _PRECISION_INPUT
from datetime import timedelta
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.plugins import PLUGIN_INPUT
from typing import Iterable
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning.pytorch.trainer.connectors.accelerator_connector import _LITERAL_WARN
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from pytorch_lightning.callbacks import ModelSummary
from lightning.pytorch.loggers import Logger
import os
from lightning.pytorch.strategies import Strategy
from lightning.fabric.utilities.types import _PATH
from pytorch_lightning import Trainer
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS

__all__ = ['log', 'BaseTrainer', ]


log = logging.getLogger(__name__)



class BaseTrainer(Trainer):
    def __init__(
        self,
        *,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        num_nodes: int = 1,
        precision: _PRECISION_INPUT = "32-true",
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        fast_dev_run: Union[int, bool] = False,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: int = -1,
        min_steps: Optional[int] = None,
        max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
        limit_test_batches: Optional[Union[int, float]] = None,
        limit_predict_batches: Optional[Union[int, float]] = None,
        overfit_batches: Union[int, float] = 0.0,
        val_check_interval: Optional[Union[int, float]] = None,
        check_val_every_n_epoch: Optional[int] = 1,
        num_sanity_val_steps: Optional[int] = None,
        log_every_n_steps: Optional[int] = None,
        enable_checkpointing: Optional[bool] = None,
        enable_progress_bar: Optional[bool] = None,
        enable_model_summary: Optional[bool] = None,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        benchmark: Optional[bool] = None,
        inference_mode: bool = True,
        use_distributed_sampler: bool = True,
        profiler: Optional[Union[Profiler, str]] = None,
        detect_anomaly: bool = False,
        barebones: bool = False,
        plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None,
        sync_batchnorm: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        default_root_dir: Optional[_PATH] = None,
        ckpt_path: Optional[_PATH] = None,
        ckpt_dir: Optional[_PATH] = None,
        enable_autorun: bool = False,
        retry_times_total_limit: int = 20,
        retry_times_sub_limit: int = 5,
        rerun_start_on_epoch: int = 10,
        rerun_when_loss: float = 5.,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            logger=logger,
            callbacks=callbacks,
            fast_dev_run=fast_dev_run,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_steps=max_steps,
            min_steps=min_steps,
            max_time=max_time,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,
            overfit_batches=overfit_batches,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            num_sanity_val_steps=num_sanity_val_steps,
            log_every_n_steps=log_every_n_steps,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            deterministic=deterministic,
            benchmark=benchmark,
            inference_mode=inference_mode,
            use_distributed_sampler=use_distributed_sampler,
            profiler=profiler,
            detect_anomaly=detect_anomaly,
            barebones=barebones,
            plugins=plugins,
            sync_batchnorm=sync_batchnorm,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            default_root_dir=default_root_dir,
        )
        self.ckpt_path = ckpt_path
        self.ckpt_dir = ckpt_dir
        self.enable_autorun = enable_autorun
        self.retry_times_total_limit = retry_times_total_limit
        self.retry_times_sub_limit = retry_times_sub_limit
        self.rerun_start_on_epoch = rerun_start_on_epoch
        self.rerun_when_loss = rerun_when_loss
        self.should_rerun = False

    def hyper_fit(
        self,
        model: "pl.LightningModule",
        train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional[LightningDataModule] = None,
        ckpt_path: Optional[str] = None,
    ) -> None:
        if self.enable_autorun:
            model.rerun_start_on_epoch = self.rerun_start_on_epoch
            model.rerun_when_loss = self.rerun_when_loss
            self.fit(
                model=model,
                train_dataloaders=train_dataloaders,
                val_dataloaders=val_dataloaders,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
            )

            current_sub_retry_times = 1
            current_total_retry_times = 0
            rerun = self.should_rerun
            last_epoch = None
            while rerun:
                for i, callback in enumerate(self.callbacks):
                    if type(callback) == ModelSummary:
                        self.callbacks.pop(i)
                ckpt_list = os.listdir(self.ckpt_dir)
                for ckpt_tmp in ckpt_list:
                    if ckpt_tmp.replace("epoch=", "").replace(".ckpt", "").isnumeric():
                        ckpt = ckpt_tmp
                        ckpt_resume = f"{self.ckpt_dir}/{ckpt}"
                        epoch = int(ckpt.replace("epoch=", "").replace(".ckpt", ""))
                        break
                if epoch == last_epoch:
                    current_sub_retry_times += 1
                else:
                    current_sub_retry_times = 1
                current_total_retry_times += 1
                last_epoch = epoch

                self.should_stop = False
                log.info("Train loss is explored. Try to rerun from best epoch.")
                log.info(f"Sub rerun times: {current_sub_retry_times} / {self.retry_times_sub_limit} .")
                log.info(f"Total rerun times: {current_total_retry_times} / {self.retry_times_total_limit} .")
                self.fit_loop.epoch_progress.current.processed = epoch
                self.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_resume)
                rerun = (
                    (
                        (current_total_retry_times != self.retry_times_total_limit) and
                        (current_sub_retry_times != self.retry_times_sub_limit)
                    ) and
                    self.should_rerun
                )
                self.should_rerun = False
        else:
            self.fit(
                model=model,
                train_dataloaders=train_dataloaders,
                val_dataloaders=val_dataloaders,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
            )


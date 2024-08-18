import tempfile
from omegaconf import OmegaConf
from .utils import to_get_datetime
from .utils import to_get_uuid
from .utils import str2class
import logging
from pytorch_lightning import Callback
from .utils import print_config
from pytorch_lightning import Trainer
import hydra
from pytorch_lightning import LightningModule
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import Logger
import torch
from pytorch_lightning import LightningDataModule
import os
from omegaconf import DictConfig
from .utils import log_hyperparameters
from typing import List
from .train import PredictionWriter

__all__ = ['log', 'test', 'train', 'predict', ]

OmegaConf.register_new_resolver("tmpdir", tempfile.mkdtemp, use_cache=True)
OmegaConf.register_new_resolver("datetime", to_get_datetime)
OmegaConf.register_new_resolver("uuid", to_get_uuid)
OmegaConf.register_new_resolver("str2class", str2class)





log = logging.getLogger(__name__)



@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def train(config: DictConfig):

    config.run.output_root_dir = os.path.join(config.run_dir, config.run.id)

    if OmegaConf.is_missing(config, "run.data_dir"):
        log.error(
            "Config incomplete! You need to specify the data directory `data_dir`."
        )
        return

    if not ("model" in config and "data" in config):
        log.error(
            """
        Config incomplete! You have to specify at least `data` and `model`!
        For an example, try one of our pre-defined experiments:
        > spktrain data_dir=/data/will/be/here +experiment=qm9
        """
        )
        return

    if not os.path.exists(os.path.join(config.run.execute_dir, 'slurm_out')):
        os.makedirs(os.path.join(config.run.execute_dir, 'slurm_out'))

    if not os.path.exists(config.run.config_dir):
        os.makedirs(config.run.config_dir)

    config_save_file = config.run.id.replace(".", "_")

    with open(f"{config.run.config_dir}/{config_save_file}.yaml", "w") as f:
        OmegaConf.save(config, f, resolve=False)

    if config.get("print_config"):
        print_config(config, resolve=False)

    if not os.path.exists(config.run.data_dir):
        os.makedirs(config.run.data_dir)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.data)

    # Init model
    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)
    if "from_model_path" in config.globals:
        checkpoint = torch.load(config.globals.from_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Init LightningModule
    log.info(f"Instantiating main <{config.main._target_}>")
    scheduler_cls = (
        str2class(config.main.scheduler_cls) if config.main.scheduler_cls else None
    )

    main: LightningModule = hydra.utils.instantiate(
        config.main,
        model=model,
        optimizer_cls=str2class(config.main.optimizer_cls),
        scheduler_cls=scheduler_cls,
    )
    if config.globals.compile:
        main = torch.compile(main)

    if config.get("output_result_file"):
        main.output_result_file(config.outputFileName)

    # Init Lightning callbacks
    callbacks: List[Callback] = []

    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    if "seed" in config:
        seed_everything(config.seed, workers=True)
    else:
        seed_everything(workers=True)

    # Init Lightning loggers
    logger: List[Logger] = []

    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                lg = hydra.utils.instantiate(lg_conf)

                logger.append(lg)

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=config.run.output_root_dir,
        ckpt_dir=f"{config.run.output_root_dir}/checkpoints",
        _convert_="partial",
    )

    log.info("Logging hyperparameters.")
    log_hyperparameters(config=config, model=main, trainer=trainer)

    # Train the model
    log.info("Starting training.")
    trainer.hyper_fit(model=main, datamodule=datamodule, ckpt_path=trainer.ckpt_path)

    # Evaluate model on test set after training
    log.info("Starting testing.")
    trainer.test(model=main, datamodule=datamodule, ckpt_path="best")

    # Store best model
    best_path = trainer.checkpoint_callback.best_model_path
    log.info(f"Best checkpoint path:\n{best_path}")

    log.info("Store best model.")
    best_result = type(main).load_from_checkpoint(best_path)
    best_result.save_model(config.globals.model_path)
    if config.autorun:
        with open('trainstat', 'w') as f:
            f.write('finish')



@hydra.main(config_path="configs", config_name="test", version_base="1.2")
def test(config: DictConfig):

    if config.get("print_config"):
        print_config(
            config,
            fields=(
                "test_data_name",
                "modeldir",
                "globals",
                "data",
                "main",
                "trainer",
            ),
            resolve=False
        )

    config_save_file = config.run.id.replace(".", "_")

    if not os.path.exists(config.run.config_dir):
        os.makedirs(config.run.config_dir)

    with open(f"{config.run.config_dir}/{config_save_file}.yaml", "w") as f:
        OmegaConf.save(config, f, resolve=False)

    datamodule: LightningDataModule = hydra.utils.instantiate(config.data)

    model = torch.load(f"{config.modeldir}/best_model")

    if "outputs" in config.main:
        outputs = hydra.utils.instantiate(config.main.outputs)
        main: LightningModule = hydra.utils.instantiate(
            config.main,
            model=model,
            outputs=outputs,
            optimizer_cls=None,
            scheduler_cls=None,
        )
    else:
        main: LightningModule = hydra.utils.instantiate(
            config.main,
            model=model,
            optimizer_cls=None,
            scheduler_cls=None,
        )

    if config.get("output_result_file"):
        main.output_result_file(config.outputFileName)

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=[
            PredictionWriter(
                output_dir=config.data.outputdir, write_interval=config.write_interval
            )
        ],
        default_root_dir=config.modeldir,
        _convert_="partial",
    )
    ckpt = os.listdir(f"{config.modeldir}/checkpoints")[0]

    log.info("Logging hyperparameters.")
    log_hyperparameters(config=config, model=main, trainer=trainer)

    trainer.test(model=main, dataloaders=datamodule, ckpt_path=f"{config.modeldir}/checkpoints/{ckpt}")



@hydra.main(config_path="configs", config_name="predict", version_base="1.2")
def predict(config: DictConfig):

    if config.get("print_config"):
        print_config(
            config,
            fields=(
                "test_data_name",
                "modeldir",
                "outputdir",
                "globals",
                "data",
                "main",
                "trainer",
            ),
            resolve=False
        )

    config_save_file = config.run.id.replace(".", "_")

    with open(f"{config.run.config_dir}/{config_save_file}.yaml", "w") as f:
        OmegaConf.save(config, f, resolve=False)

    datamodule: LightningDataModule = hydra.utils.instantiate(config.data)

    model = torch.load(f"{config.modeldir}/best_model")

    if "outputs" in config.main:
        outputs = hydra.utils.instantiate(config.main.outputs)
        main: LightningModule = hydra.utils.instantiate(
            config.main,
            model=model,
            outputs=outputs,
            optimizer_cls=None,
            scheduler_cls=None,
        )
    else:
        main: LightningModule = hydra.utils.instantiate(
            config.main,
            model=model,
            optimizer_cls=None,
            scheduler_cls=None,
        )

    if config.get("output_result_file"):
        main.output_result_file(config.outputFileName)

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=[
            PredictionWriter(
                output_dir=config.outputdir, write_interval=config.write_interval
            )
        ],
        default_root_dir=".",
        _convert_="partial",
    )
    ckpt = os.listdir(f"{config.modeldir}/checkpoints")[0]
    trainer.predict(model=main, dataloaders=datamodule, ckpt_path=f"{config.modeldir}/checkpoints/{ckpt}")





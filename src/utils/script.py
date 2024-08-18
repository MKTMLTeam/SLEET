from typing import Type
import importlib
from rich.tree import Tree
from omegaconf import OmegaConf
import rich
from typing import Sequence
from rich.syntax import Syntax
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import DictConfig
import os
import datetime
import time
import torch
import uuid

__all__ = ['end_get_uuid', 'get_datetime', 'as_dtype', 'log_hyperparameters', 'str2class', 'print_config', 'init_datetime', 'get_uuid', 'init_uuid', 'uuid_str', 'datetime_str', 'to_get_datetime', 'to_get_uuid', 'TORCH_DTYPES', 'conn', 'empty', 'end_get_datetime', ]


uuid_str = None


def str2class(class_path: str) -> Type:
    """
    Obtain a class type from a string

    Args:
        class_path: module path to class, e.g. ``module.submodule.classname``

    Returns:
        class type
    """
    class_path = class_path.split(".")
    class_name = class_path[-1]
    module_name = ".".join(class_path[:-1])
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls



@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "run",
        "globals",
        "data",
        "model",
        "main",
        "trainer",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Config.
        fields (Sequence[str], optional): Determines which main fields from config will be printed
        and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = Tree(
        ":gear: Running with the following config:", style=style, guide_style=style
    )

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(Syntax(branch_content, "yaml"))

    rich.print(tree)



get_uuid = True


get_datetime = True


@rank_zero_only
def end_get_uuid():
    with open('scripts/conn', 'w') as f:
        f.write("True")
    os.remove('uuidStore.txt')



@rank_zero_only
def end_get_datetime():
    with open('scripts/conn', 'w') as f:
        f.write("True")
    os.remove('datetimeStore.txt')




def empty(*args, **kwargs):
    pass




@rank_zero_only
def log_hyperparameters(config, model, trainer) -> None:
    """
    This saves Hydra config using Lightning loggers.
    """

    # send hparams to all loggers
    trainer.logger.log_hyperparams(config)

    # disable logging any more hyperparameters for all loggers
    trainer.logger.log_hyperparams = empty




datetime_str = None


@rank_zero_only
def init_datetime():
    global get_datetime
    global datetime_str
    datetime_str = str(datetime.datetime.now()).replace(' ', '_').replace('-', '_').replace(':', '_')
    with open('datetimeStore.txt', 'w') as f:
        f.write(datetime_str)
    with open('scripts/conn', 'w') as f:
        f.write("False")



conn = True




def to_get_datetime(x):
    global get_datetime
    global datetime_str
    global conn
    if get_datetime:
        init_datetime()
        while conn:
            time.sleep(0.5)
            with open('scripts/conn', 'r') as f:
                conn_str = f.read()
                conn = conn_str == "True"
        with open('datetimeStore.txt', 'r') as f:
            datetime_str = f.read()
        get_datetime = False
        end_get_datetime()
        return datetime_str
    elif not get_datetime:
        return datetime_str

TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float": torch.float,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "half": torch.half,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "short": torch.short,
    "int32": torch.int32,
    "int": torch.int,
    "int64": torch.int64,
    "long": torch.long,
    "complex64": torch.complex64,
    "cfloat": torch.cfloat,
    "complex128": torch.complex128,
    "cdouble": torch.cdouble,
    "quint8": torch.quint8,
    "qint8": torch.qint8,
    "qint32": torch.qint32,
    "bool": torch.bool,
}


def as_dtype(dtype_str: str) -> torch.dtype:
    """Convert a string to torch.dtype"""
    return TORCH_DTYPES[dtype_str]




@rank_zero_only
def init_uuid():
    global get_uuid
    global uuid_str
    uuid_str = str(uuid.uuid1())
    with open('uuidStore.txt', 'w') as f:
        f.write(uuid_str)
    with open('scripts/conn', 'w') as f:
        f.write("False")



def to_get_uuid(x):
    global get_uuid
    global uuid_str
    global conn
    if get_uuid:
        init_uuid()
        while conn:
            time.sleep(0.5)
            with open('scripts/conn', 'r') as f:
                conn_str = f.read()
                conn = conn_str == "True"
        with open('uuidStore.txt', 'r') as f:
            uuid_str = f.read()
        get_uuid = False
        end_get_uuid()
        return uuid_str
    elif not get_uuid:
        return uuid_str





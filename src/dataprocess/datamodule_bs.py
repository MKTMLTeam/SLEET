from .datamodule import AtomsDataModuleError
from .atoms import AtomsDataFormat
from .loader import _atoms_collate_fn
from typing import List
from .bondsteps import load_bond_steps_no_pad_dataset
import numpy as np
import shutil
from typing import Union
from .datamodule import AtomsDataModule
from typing import Optional
import os
import torch
from .splitting import SplittingStrategy
from torch.utils.data.dataloader import _collate_fn_t
import logging
from typing import Dict
import fasteners

__all__ = ['BondStepsNoPadDataModule', ]


class BondStepsNoPadDataModule(AtomsDataModule):
    """
    A general ``LightningDataModule`` for SchNetPack bond steps datasets.

    """

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        outputdir: str,
        num_train: Union[int, float] = None,
        num_val: Union[int, float] = None,
        num_test: Optional[Union[int, float]] = None,
        split_file: Optional[str] = "split.npz",
        format: Optional[AtomsDataFormat] = None,
        load_properties: Optional[List[str]] = None,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        train_transforms: Optional[List[torch.nn.Module]] = None,
        val_transforms: Optional[List[torch.nn.Module]] = None,
        test_transforms: Optional[List[torch.nn.Module]] = None,
        num_workers: int = 8,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        property_units: Optional[Dict[str, str]] = None,
        data_workdir: Optional[str] = None,
        cleanup_workdir_stage: Optional[str] = "test",
        splitting: Optional[SplittingStrategy] = None,
        pin_memory: Optional[bool] = False,
        split: Optional[str] = "test",
        collate_fn: Optional[_collate_fn_t] = _atoms_collate_fn,
        regressionTransformer: Optional[bool] = False,
    ):
        """
        Args:
            datapath: path to dataset
            batch_size: (train) batch size
            num_train: number of training examples (absolute or relative)
            num_val: number of validation examples (absolute or relative)
            num_test: number of test examples (absolute or relative)
            split_file: path to npz file with data partitions
            format: dataset format
            load_properties: subset of properties to load
            val_batch_size: validation batch size. If None, use test_batch_size, then
                batch_size.
            test_batch_size: test batch size. If None, use val_batch_size, then
                batch_size.
            transforms: Preprocessing transform applied to each system separately before
                batching.
            train_transforms: Overrides transform_fn for training.
            val_transforms: Overrides transform_fn for validation.
            test_transforms: Overrides transform_fn for testing.
            num_workers: Number of data loader workers.
            num_val_workers: Number of validation data loader workers
                (overrides num_workers).
            num_test_workers: Number of test data loader workers
                (overrides num_workers).
            property_units: Dictionary from property to corresponding unit as a string
                (eV, kcal/mol, ...).
            distance_unit: Unit of the atom positions and cell as a string
                (Ang, Bohr, ...).
            data_workdir: Copy data here as part of setup, e.g. to a local file
                system for faster performance.
            cleanup_workdir_after: Determines after which stage to remove the data
                workdir
            splitting: Method to generate train/validation/test partitions
                (default: RandomSplit)
            pin_memory: If true, pin memory of loaded data to GPU. Default: Will be
                set to true, when GPUs are used.
        """
        super().__init__(
            datapath=datapath,
            batch_size=batch_size,
            outputdir=outputdir,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            split_file=split_file,
            format=format,
            load_properties=load_properties,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            transforms=transforms,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            num_workers=num_workers,
            num_val_workers=num_val_workers,
            num_test_workers=num_test_workers,
            property_units=property_units,
            data_workdir=data_workdir,
            cleanup_workdir_stage=cleanup_workdir_stage,
            splitting=splitting,
            pin_memory=pin_memory,
            split=split,
            collate_fn=collate_fn,
        )
        self.rt = regressionTransformer

    def setup(self, stage: Optional[str] = None):
        # check whether data needs to be copied
        if self.data_workdir is None:
            datapath = self.datapath
        else:
            datapath = self._copy_to_workdir()

        # copy split file to output dir
        if self.split_file != f"{self.outputdir}/split.npz":
            if os.path.exists(self.split_file):
                shutil.copy(self.split_file, f"{self.outputdir}/split.npz")
                self.split_file = f"{self.outputdir}/split.npz"

        # (re)load datasets
        if self.dataset is None:
            self.dataset = load_bond_steps_no_pad_dataset(
                datapath,
                self.format,
                property_units=self.property_units,
                regressionTransformer=self.rt,
            )

            # load and generate partitions if needed
            if self.train_idx is None:
                self._load_partitions()

            # partition dataset
            self._train_dataset = self.dataset.subset(self.train_idx)
            self._val_dataset = self.dataset.subset(self.val_idx)
            if self.split == "test":
                self._test_dataset = self.dataset.subset(self.test_idx)
            elif self.split == "train":
                self._test_dataset = self.dataset.subset(self.train_idx)
            elif self.split == "validation":
                self._test_dataset = self.dataset.subset(self.val_idx)
            else:
                raise ValueError(f"Unacceptable split: {self.split}")
            self._setup_transforms()

    def _load_partitions(self):
        # split dataset
        if self.split_file is not None:
            lock = fasteners.InterProcessLock(f"{self.split_file.replace('split.npz', '')}/splitting.lock")
        else:
            lock = fasteners.InterProcessLock("splitting.lock")

        with lock:
            self._log_with_rank("Enter splitting lock")

            if self.split_file is not None and os.path.exists(self.split_file):
                self._log_with_rank("Load split")

                S = np.load(self.split_file)
                self.train_idx = S["train_idx"].tolist()
                self.val_idx = S["val_idx"].tolist()
                self.test_idx = S["test_idx"].tolist()
                if self.num_train and self.num_train != len(self.train_idx):
                    logging.warning(
                        f"Split file was given, but `num_train ({self.num_train})"
                        + f" != len(train_idx)` ({len(self.train_idx)})!"
                    )
                if self.num_val and self.num_val != len(self.val_idx):
                    logging.warning(
                        f"Split file was given, but `num_val ({self.num_val})"
                        + f" != len(val_idx)` ({len(self.val_idx)})!"
                    )
                if self.num_test and self.num_test != len(self.test_idx):
                    logging.warning(
                        f"Split file was given, but `num_test ({self.num_test})"
                        + f" != len(test_idx)` ({len(self.test_idx)})!"
                    )
            else:
                self._log_with_rank("Create split")

                if not self.num_train or not self.num_val:
                    raise AtomsDataModuleError(
                        "If no `split_file` is given, the sizes of the training and"
                        + " validation partitions need to be set!"
                    )
                if hasattr(self.splitting, "input_idx"):
                    self.splitting = self.splitting()
                    self.splitting.input_idx(
                        self.dataset.metadata["_property_has_value_dict"][self.property]
                    )
                    self.train_idx, self.val_idx, self.test_idx = self.splitting.split(
                        self.num_train,
                        self.num_val,
                        self.num_test,
                    )
                else:
                    self.train_idx, self.val_idx, self.test_idx = self.splitting.split(
                        self.dataset, self.num_train, self.num_val, self.num_test
                    )

                if self.split_file is not None:
                    self._log_with_rank("Save split")
                    np.savez(
                        self.split_file,
                        train_idx=self.train_idx,
                        val_idx=self.val_idx,
                        test_idx=self.test_idx,
                    )

        self._log_with_rank("Exit splitting lock")


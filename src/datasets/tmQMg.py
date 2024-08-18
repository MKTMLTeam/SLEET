import shutil
from typing import List
import logging
from typing import Optional
from typing import Dict
import tempfile
from src.dataprocess import _ligands_collate_fn
from tqdm import tqdm
from src.transform import analysis_ligands
import numpy as np
from src.transform import get_atom_numbers_from_smi
import torch
from src.dataprocess import AtomsDataFormat
from src.dataprocess import create_bond_steps_no_pad_dataset
from src import properties as structure
from src.dataprocess import BondStepsNoPadDataModule
import os
from src.dataprocess import load_bond_steps_no_pad_dataset
from torch.utils.data.dataloader import _collate_fn_t
from src.transform import tmQM_smiles_fix
from src.dataprocess import BaseAtomsData

__all__ = ['tmQMg_obabel_bondstep_no_pad_ligands', ]


class tmQMg_obabel_bondstep_no_pad_ligands(BondStepsNoPadDataModule):

    # properties
    lumo = "tzvp_lumo_energy"
    homo = "tzvp_homo_energy"
    gap = "tzvp_homo_lumo_gap"
    gap_d = "homo_lumo_gap_delta"
    e = "tzvp_electronic_energy"
    e_d = "electronic_energy_delta"
    dis = "tzvp_dispersion_energy"
    dis_d = "dispersion_energy_delta"
    H = "enthalpy_energy"
    H_c = "enthalpy_energy_correction"
    G = "gibbs_energy"
    G_c = "gibbs_energy_correction"
    zpe_c = "zpe_correction"
    Cv = "heat_capacity"
    S = "entropy"
    mu = "tzvp_dipole_moment"
    mu_d = "dipole_moment_delta"
    alpha = "polarizability"
    lovib = "lowest_vibrational_frequency"
    hivib = "highest_vibrational_frequency"

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        outputdir: str,
        smi_datapath: str,
        y_datapath: str,
        num_train: Optional[int] = None,
        num_val: Optional[int] = None,
        num_test: Optional[int] = None,
        split_file: Optional[str] = "split.npz",
        format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
        load_properties: Optional[List[str]] = None,
        remove_uncharacterized: bool = False,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        train_transforms: Optional[List[torch.nn.Module]] = None,
        val_transforms: Optional[List[torch.nn.Module]] = None,
        test_transforms: Optional[List[torch.nn.Module]] = None,
        num_workers: int = 2,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        data_workdir: Optional[str] = None,
        collate_fn: Optional[_collate_fn_t] = _ligands_collate_fn,
        regressionTransformer: Optional[bool] = False,
        separated_ligands: Optional[bool] = False,
        only_pos: Optional[bool] = False,
        **kwargs
    ):
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
            collate_fn=collate_fn,
            regressionTransformer=regressionTransformer,
            **kwargs
        )
        self.y_datapath = y_datapath
        self.smi_datapath = smi_datapath
        self.rt = regressionTransformer
        self.separated_ligands = separated_ligands
        self.only_pos = only_pos

    def prepare_data(self):
        if not os.path.exists(self.datapath):
            property_unit_dict = {
                tmQMg_obabel_bondstep_no_pad_ligands.lumo: "Ha",
                tmQMg_obabel_bondstep_no_pad_ligands.homo: "Ha",
                tmQMg_obabel_bondstep_no_pad_ligands.gap: "Ha",
                tmQMg_obabel_bondstep_no_pad_ligands.gap_d: "Ha",
                tmQMg_obabel_bondstep_no_pad_ligands.e: "Ha",
                tmQMg_obabel_bondstep_no_pad_ligands.e_d: "Ha",
                tmQMg_obabel_bondstep_no_pad_ligands.dis: "Ha",
                tmQMg_obabel_bondstep_no_pad_ligands.dis_d: "Ha",
                tmQMg_obabel_bondstep_no_pad_ligands.H: "Ha",
                tmQMg_obabel_bondstep_no_pad_ligands.H_c: "Ha",
                tmQMg_obabel_bondstep_no_pad_ligands.G: "Ha",
                tmQMg_obabel_bondstep_no_pad_ligands.G_c: "Ha",
                tmQMg_obabel_bondstep_no_pad_ligands.zpe_c: "Ha",
                tmQMg_obabel_bondstep_no_pad_ligands.Cv: "cal/mol/K",
                tmQMg_obabel_bondstep_no_pad_ligands.S: "cal/mol/K",
                tmQMg_obabel_bondstep_no_pad_ligands.mu: "D",
                tmQMg_obabel_bondstep_no_pad_ligands.mu_d: "D",
                tmQMg_obabel_bondstep_no_pad_ligands.alpha: "a0^3",
                tmQMg_obabel_bondstep_no_pad_ligands.lovib: "invcm",
                tmQMg_obabel_bondstep_no_pad_ligands.hivib: "invcm",
                structure.total_charge: None,
                structure.n_elec: None,
                structure.M: None,
                structure.MGp: None,
                structure.MPd: None,
                structure.L: None,
                structure.n_L: None,
            }

            tmpdir = tempfile.mkdtemp("tmQMg_bs")

            dataset = create_bond_steps_no_pad_dataset(
                datapath=self.datapath,
                format=self.format,
                property_unit_dict=property_unit_dict,
                regressionTransformer=self.rt
            )
            try:
                self._download_data(tmpdir, dataset)
            except Exception:
                os.remove(self.datapath)
                raise
            shutil.rmtree(tmpdir)
        else:
            dataset = load_bond_steps_no_pad_dataset(
                self.datapath, self.format, regressionTransformer=self.rt
            )

    def _download_data(
        self, tmpdir, dataset: BaseAtomsData
    ):
        logging.info("Loading tmQMg smiles data...")
        with open(self.smi_datapath) as f:
            lines = f.readlines()
            new_lines = []
            for line in lines:
                line = line.split('\t')
                new_lines.append(line)
            smiless = sorted(new_lines)
        logging.info("Done.")

        logging.info("Parse y files...")
        with open(self.y_datapath) as f:
            y_lines = sorted(f.readlines()[1:])

        irange = np.arange(len(smiless), dtype=int)

        char_table = np.load('datasource/char.npy', allow_pickle=True)
        char_plus = np.load('datasource/char_plus_fix.npy', allow_pickle=True)

        property_list = []

        n_L_list = []

        for i in tqdm(irange):
            properties = {}

            smiles = smiless[i][1]
            smiles = tmQM_smiles_fix(smiles)
            ats = get_atom_numbers_from_smi(smiles)
            metal, ligands = analysis_ligands(smiles, char_table, char_plus, self.separated_ligands, self.only_pos)
            # if len(ligands) > 8:
            #     raise Exception("lens of ligands: ", len(ligands))
            lp = y_lines[i].replace("\n", "").split(",")
            if not self.separated_ligands:
                real_n_L = int(np.sum(li[structure.LB] for li in ligands) + len(ligands))
            else:
                real_n_L = len(ligands)
            n_L_list.append(real_n_L)
            for pn, p in zip(dataset.available_properties, lp[5:]):
                properties[pn] = np.array([float(p)])

            properties[structure.Z] = ats
            properties[structure.M] = metal
            properties[structure.MGp] = np.array([char_plus[metal[0]][1]])
            properties[structure.MPd] = np.array([char_plus[metal[0]][2]])
            properties[structure.L] = ligands
            properties[structure.n_L] = np.array([real_n_L])
            properties[structure.total_charge] = np.array([float(lp[1])])
            properties[structure.n_elec] = np.array([float(lp[4])])
            property_list.append(properties)

        max_n_L = max(n_L_list)

        logging.info("Done.")
        logging.info("Write atoms to db...")
        dataset.update_metadata(max_nligands=max_n_L)
        dataset.add_systems(property_list=property_list)
        logging.info("Done.")


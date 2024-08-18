import logging
from typing import Dict
from ase.db import connect
from ase import Atoms
from typing import Iterable
from .atoms import BaseAtomsData
from src import properties as structure
from typing import Union
from typing import Any
import os
import torch
from typing import List
from typing import Optional
from src.units import convert_units
from src.transform import bond_step_gen
from .atoms import AtomsDataFormat

__all__ = ['BondStepsDataError', 'logger', 'BondStepsData', 'load_bond_steps_no_pad_dataset', 'BondStepsNoPadData', 'create_bond_steps_no_pad_dataset', ]


logger = logging.getLogger(__name__)

class BondStepsDataError(Exception):
    pass




class BondStepsData(BaseAtomsData):
    def __init__(
        self,
        datapath: str,
        load_properties: Optional[List[str]] = None,
        load_structure: bool = True,
        transforms: Optional[List[torch.nn.Module]] = None,
        subset_idx: Optional[List[int]] = None,
        property_units: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            datapath: Path to ASE DB.
            load_properties: Set of properties to be loaded and returned.
                If None, all properties in the ASE dB will be returned.
            load_properties: If True, load structure properties.
            transforms: preprocessing torch.nn.Module (see schnetpack.data.transforms)
            subset_idx: List of data indices.
            units: property-> unit string dictionary that overwrites the native units
                of the dataset. Units are converted automatically during loading.
        """
        self.datapath = datapath

        BaseAtomsData.__init__(
            self,
            load_properties=load_properties,
            load_structure=load_structure,
            transforms=transforms,
            subset_idx=subset_idx,
        )

        self._check_db()
        self.conn = connect(self.datapath)

        # initialize units
        md = self.metadata
        if "_property_unit_dict" not in md.keys():
            raise BondStepsDataError(
                "Dataset does not have a property units set. Please add units to the "
                + "dataset using `spkconvert`!"
            )

        self._units = md["_property_unit_dict"]
        self.conversions = {prop: 1.0 for prop in self._units}
        if property_units is not None:
            for prop, unit in property_units.items():
                self.conversions[prop] = convert_units(
                    self._units[prop], unit
                )
                self._units[prop] = unit

    def __len__(self) -> int:
        if self.subset_idx is not None:
            return len(self.subset_idx)

        with connect(self.datapath, use_lock_file=False) as conn:
            return conn.count()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.subset_idx is not None:
            idx = self.subset_idx[idx]

        props = self._get_properties(
            self.conn, idx, self.load_properties, self.load_structure
        )
        props = self._apply_transforms(props)

        return props

    def _apply_transforms(self, props):
        if self._transform_module is not None:
            props = self._transform_module(props)
        return props

    def _check_db(self):
        if not os.path.exists(self.datapath):
            raise BondStepsDataError(f"ASE DB does not exists at {self.datapath}")

        if self.subset_idx:
            with connect(self.datapath, use_lock_file=False) as conn:
                n_structures = conn.count()

            assert max(self.subset_idx) < n_structures

    def iter_properties(
        self,
        indices: Union[int, Iterable[int]] = None,
        load_properties: List[str] = None,
        load_structure: Optional[bool] = None,
    ):
        """
        Return property dictionary at given indices.

        Args:
            indices: data indices
            load_properties (sequence or None): subset of available properties to load
            load_structure: load and return structure

        Returns:
            properties (dict): dictionary with molecular properties

        """
        load_properties = load_properties or self.load_properties
        load_structure = load_structure or self.load_structure

        if self.subset_idx:
            if indices is None:
                indices = self.subset_idx
            elif type(indices) is int:
                indices = [self.subset_idx[indices]]
            else:
                indices = [self.subset_idx[i] for i in indices]
        else:
            if indices is None:
                indices = range(len(self))
            elif type(indices) is int:
                indices = [indices]

        # read from ase db
        with connect(self.datapath, use_lock_file=False) as conn:
            for i in indices:
                yield self._get_properties(
                    conn,
                    i,
                    load_properties=load_properties,
                    load_structure=load_structure,
                )

    def atomrefs(self) -> Dict[str, torch.Tensor]:
        """Single-atom reference values for properties"""
        pass

    def _get_properties(
        self, conn, idx: int, load_properties: List[str], load_structure: bool
    ):
        row = conn.get(idx + 1)

        # extract properties
        # TODO: can the copies be avoided?
        properties = {}
        properties[structure.idx] = torch.tensor([idx])
        for pname in load_properties:
            properties[pname] = (
                    torch.tensor(row.data[pname].copy()) * self.conversions[pname]
                )

        Z = row["numbers"].copy()
        properties[structure.n_atoms] = torch.tensor([Z.shape[0]])

        if load_structure:
            properties[structure.Z] = torch.tensor(Z, dtype=torch.long)
            if structure.sm in row.data.keys():
                properties[structure.bond_step] = torch.tensor(
                    bond_step_gen(row.data[structure.sm])
                ).float()
            if structure.stereo in row.data.keys():
                properties[structure.stereo] = (
                        torch.tensor(row.data[structure.stereo].copy())
                    )
        if structure.sol in row.data.keys():
            properties[structure.sol] = (
                    torch.tensor(row.data[structure.sol].copy())
                )

        return properties

    # Metadata

    @property
    def metadata(self):
        with connect(self.datapath) as conn:
            return conn.metadata

    def _set_metadata(self, val: Dict[str, Any]):
        with connect(self.datapath) as conn:
            conn.metadata = val

    def update_metadata(self, **kwargs):
        assert all(
            key[0] != 0 for key in kwargs
        ), "Metadata keys starting with '_' are protected!"

        md = self.metadata
        md.update(kwargs)
        self._set_metadata(md)

    @property
    def available_properties(self) -> List[str]:
        md = self.metadata
        return list(md["_property_unit_dict"].keys())

    @property
    def units(self) -> Dict[str, str]:
        """Dictionary of properties to units"""
        return self._units

    ## Creation

    @staticmethod
    def create(
        datapath: str,
        property_unit_dict: Dict[str, str],
        property_has_value_dict: Optional[Dict[str, List]] = None,
        **kwargs,
    ) -> "BondStepsData":
        """

        Args:
            datapath: Path to ASE DB.
            property_unit_dict: Defines the available properties of the datasetseta and
                provides units for ALL properties of the dataset. If a property is
                unit-less, you can pass "arb. unit" or `None`.
            kwargs: Pass arguments to init.

        Returns:
            newly created BondStepsData

        """
        if not datapath.endswith(".db"):
            raise BondStepsDataError(
                "Invalid datapath! Please make sure to add the file extension '.db' to "
                "your dbpath."
            )

        if os.path.exists(datapath):
            raise BondStepsDataError(f"Dataset already exists: {datapath}")
        with connect(datapath, use_lock_file=False) as conn:
            conn.metadata = {
                "_property_unit_dict": property_unit_dict,
                "_property_has_value_dict": property_has_value_dict,
            }

        return BondStepsData(datapath, **kwargs)

    # add systems
    def add_system(self, **properties):
        """
        Add atoms data to the dataset.

        Args:
            atoms: System composition and geometry. If Atoms are None,
                the structure needs to be given as part of the property dict
                (using structure.Z, structure.R, structure.cell, structure.pbc)
            **properties: properties as key-value pairs. Keys have to match the
                `available_properties` of the dataset.

        """
        with connect(self.datapath, use_lock_file=False) as conn:
            self._add_system(conn, **properties)

    def add_systems(
        self,
        property_list: List[Dict[str, Any]],
    ):
        """
        Add atoms data to the dataset.

        Args:
            atoms_list: System composition and geometry. If Atoms are None,
                the structure needs to be given as part of the property dicts
                (using structure.Z, structure.R, structure.cell, structure.pbc)
            property_list: Properties as list of key-value pairs in the same
                order as corresponding list of `atoms`.
                Keys have to match the `available_properties` of the dataset
                plus additional structure properties, if atoms is None.
        """

        with connect(self.datapath, use_lock_file=False) as conn:
            for prop in property_list:
                self._add_system(conn, **prop)

    def _add_system(self, conn, **properties):
        """Add systems to DB"""

        # add available properties to database
        atoms = Atoms(numbers=properties[structure.Z])
        valid_props = set().union(
            conn.metadata["_property_unit_dict"].keys(),
            [
                structure.Z,
            ],
        )
        for prop in properties:
            if prop not in valid_props:
                logger.warning(
                    f"Property `{prop}` is not a defined property for this dataset and "
                    + "will be ignored. If it should be included, it has to be "
                    + "provided together with its unit when calling "
                    + "BondStepsData.create()."
                )

        data = {}
        for pname in conn.metadata["_property_unit_dict"].keys():
            try:
                data[pname] = properties[pname]
            except Exception:
                raise BondStepsDataError("Required property missing:" + pname)

        conn.write(atoms, data=data)



class BondStepsNoPadData(BondStepsData):
    def __init__(
        self,
        datapath: str,
        load_properties: Optional[List[str]] = None,
        load_structure: bool = True,
        transforms: Optional[List[torch.nn.Module]] = None,
        subset_idx: Optional[List[int]] = None,
        property_units: Optional[Dict[str, str]] = None,
        regressionTransformer: Optional[Dict[str, str]] = False,
    ):
        """
        Args:
            datapath: Path to ASE DB.
            load_properties: Set of properties to be loaded and returned.
                If None, all properties in the ASE dB will be returned.
            load_properties: If True, load structure properties.
            transforms: preprocessing torch.nn.Module (see schnetpack.data.transforms)
            subset_idx: List of data indices.
            units: property-> unit string dictionary that overwrites the native units
                of the dataset. Units are converted automatically during loading.
        """
        super().__init__(
            datapath=datapath,
            load_properties=load_properties,
            load_structure=load_structure,
            transforms=transforms,
            subset_idx=subset_idx,
            property_units=property_units,
        )
        self.rt = regressionTransformer

    def _get_properties(
        self, conn, idx: int, load_properties: List[str], load_structure: bool
    ):
        row = conn.get(idx + 1)

        # extract properties
        # TODO: can the copies be avoided?
        properties = {}
        properties[structure.idx] = torch.tensor([idx])
        special_properties = {
            structure.L,
            structure.n_L,
            structure.M,
            structure.MGp,
            structure.MPd,
        }
        if (conn.metadata["max_nligands"] != 0) and (structure.L in row.data.keys()):
            max_nligands = conn.metadata["max_nligands"]
        elif (conn.metadata["max_nligands"] == 0) and (structure.L in row.data.keys()):
            raise Exception("Didn't get max_nligands in database!")
        for pname in load_properties:
            if pname in special_properties:
                continue
            properties[pname] = (
                    torch.tensor(row.data[pname].copy()) * self.conversions[pname]
                )
            if self.rt:
                tmp = [*str(properties[pname][0]).replace(" ", "")[7:-21]]
                tmp_new = tmp.copy()
                tmp_digit = []
                point_w = None
                point_w_offset = False
                for i in range(len(tmp)):
                    if tmp[i] == ".":
                        point_w = i
                for i, t in enumerate(tmp):
                    if t == "-":
                        point_w_offset = True
                        continue
                    if i == point_w:
                        continue
                    elif point_w > i:
                        x = point_w - i - 1
                    else:
                        x = point_w - i
                    tmp_digit.append(x)
                for i, t in enumerate(tmp):
                    try:
                        if t == "0":
                            tmp_new[i] = 16
                        else:
                            tmp_new[i] = int(t) + 6 + i * 10
                    except ValueError:
                        if t == ".":
                            point_w = i
                        elif t == "-":
                            tmp_new[i] = 12
                if point_w:
                    tmp_new.pop(point_w)
                    if point_w_offset:
                        tmp_new.insert(1, point_w - 1)
                    else:
                        tmp_new.insert(0, point_w)
                if "-" not in tmp:
                    tmp_new.insert(0, 13)
                tmp_new.insert(0, 14)
                properties["label_regression"] = torch.tensor(tmp_new, dtype=torch.long)
                properties["digit"] = torch.tensor(tmp_digit, dtype=torch.long)

        Z = row["numbers"].copy()
        properties[structure.n_atoms] = torch.tensor([Z.shape[0]])

        if load_structure:
            properties[structure.Z] = torch.tensor(Z, dtype=torch.long)
        if structure.sol in row.data.keys():
            properties[structure.sol] = (
                    torch.tensor(row.data[structure.sol].copy())
                )
        if structure.n_L in row.data.keys():
            properties[structure.n_L] = (
                    torch.tensor(row.data[structure.n_L].copy(), dtype=torch.long)
                )
        if structure.M in row.data.keys():
            properties[structure.M] = (
                    torch.tensor(row.data[structure.M].copy(), dtype=torch.long)
                )
        if structure.MGp in row.data.keys():
            properties[structure.MGp] = (
                    torch.tensor(row.data[structure.MGp].copy(), dtype=torch.long)
                )
        if structure.MPd in row.data.keys():
            properties[structure.MPd] = (
                    torch.tensor(row.data[structure.MPd].copy(), dtype=torch.long)
                )
        if structure.L in row.data.keys():
            n_L = len(row.data[structure.L])
            # n = random.randint(0, n_L - 1)
            # rearange_ligands = row.data[structure.L][n:] + row.data[structure.L][:n]
            rearange_ligands = row.data[structure.L]
            properties[structure.L] = [{} for i in range(max_nligands)]
            if n_L > max_nligands:
                raise Exception(n_L)
            properties[structure.L].append(n_L)
            for i, li in enumerate(rearange_ligands):
                properties[structure.L][i][structure.n_atoms] = torch.tensor([li[structure.Z].shape[0]])
                properties[structure.L][i][structure.Z] = torch.tensor(li[structure.Z].copy(), dtype=torch.long)
                properties[structure.L][i][structure.LB] = torch.tensor(li[structure.LB].copy(), dtype=torch.long)
                properties[structure.L][i][structure.LMBP] = torch.tensor(li[structure.LMBP].copy(), dtype=torch.long)
                properties[structure.L][i][structure.Gp] = torch.tensor(li[structure.Gp].copy(), dtype=torch.long)
                properties[structure.L][i][structure.Pd] = torch.tensor(li[structure.Pd].copy(), dtype=torch.long)
                properties[structure.L][i][structure.bond_step] = torch.tensor(li[structure.bond_step].copy())
                properties[structure.L][i][structure.bond_order] = torch.tensor(li[structure.bond_order].copy())
                properties[structure.L][i][structure.R] = torch.tensor(li[structure.R].copy())
                properties[structure.L][i][structure.stereo] = torch.tensor(li[structure.stereo].copy())
            for i in range(n_L, max_nligands):
                properties[structure.L][i][structure.n_atoms] = torch.tensor([0])
                properties[structure.L][i][structure.Z] = torch.tensor([], dtype=torch.long)
                properties[structure.L][i][structure.LB] = torch.tensor([], dtype=torch.long)
                properties[structure.L][i][structure.LMBP] = torch.tensor([], dtype=torch.long)
                properties[structure.L][i][structure.Gp] = torch.tensor([], dtype=torch.long)
                properties[structure.L][i][structure.Pd] = torch.tensor([], dtype=torch.long)
                properties[structure.L][i][structure.bond_step] = torch.tensor([])
                properties[structure.L][i][structure.bond_order] = torch.tensor([])
                properties[structure.L][i][structure.R] = torch.tensor([])
                properties[structure.L][i][structure.stereo] = torch.tensor([])

        return properties

    @staticmethod
    def create(
        datapath: str,
        property_unit_dict: Dict[str, str],
        property_has_value_dict: Optional[Dict[str, List]] = None,
        **kwargs,
    ) -> "BondStepsData":
        """

        Args:
            datapath: Path to ASE DB.
            property_unit_dict: Defines the available properties of the datasetseta and
                provides units for ALL properties of the dataset. If a property is
                unit-less, you can pass "arb. unit" or `None`.
            kwargs: Pass arguments to init.

        Returns:
            newly created BondStepsData

        """
        if not datapath.endswith(".db"):
            raise BondStepsDataError(
                "Invalid datapath! Please make sure to add the file extension '.db' to "
                "your dbpath."
            )

        if os.path.exists(datapath):
            raise BondStepsDataError(f"Dataset already exists: {datapath}")
        with connect(datapath, use_lock_file=False) as conn:
            conn.metadata = {
                "_property_unit_dict": property_unit_dict,
                "_property_has_value_dict": property_has_value_dict,
                "max_nligands": 0
            }

        return BondStepsNoPadData(datapath, **kwargs)




def load_bond_steps_no_pad_dataset(
    datapath: str,
    format: AtomsDataFormat,
    regressionTransformer: Optional[bool] = False,
    **kwargs
) -> BaseAtomsData:
    """
    Load dataset.

    Args:
        datapath: file path
        format: atoms data format
        **kwargs: arguments for passed to AtomsData init

    """
    if format is AtomsDataFormat.ASE:
        dataset = BondStepsNoPadData(datapath=datapath, regressionTransformer=regressionTransformer, **kwargs)
    else:
        raise BondStepsDataError(f"Unknown format: {format}")
    return dataset



def create_bond_steps_no_pad_dataset(
    datapath: str,
    format: AtomsDataFormat,
    property_unit_dict: Dict[str, str],
    property_has_value_dict: Optional[Dict[str, List]] = None,
    regressionTransformer: Optional[bool] = False,
    **kwargs,
) -> BaseAtomsData:
    """
    Create a new atoms dataset.

    Args:
        datapath: file path
        format: atoms data format
        property_unit_dict: dictionary that maps properties to units,
            e.g. {"energy": "kcal/mol"}
        **kwargs: arguments for passed to AtomsData init

    Returns:

    """
    if format is AtomsDataFormat.ASE:
        dataset = BondStepsNoPadData.create(
            datapath=datapath,
            property_unit_dict=property_unit_dict,
            property_has_value_dict=property_has_value_dict,
            regressionTransformer=regressionTransformer,
            **kwargs,
        )
    else:
        raise BondStepsDataError(f"Unknown format: {format}")
    return dataset



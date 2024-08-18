import torch
from src import properties
from typing import Dict
from .base import Transform
import numpy as np

__all__ = ['BondStepLigandNeighborNoPadListTransform', 'BondStepLigandNeighborNoPadList', ]


class BondStepLigandNeighborNoPadListTransform(Transform):
    """
    Base class for neighbor lists.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
    ):
        """
        Args:
            cutoff: Cutoff radius for neighbor search.
        """
        super().__init__()

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        max_nligands = 14
        n_ligands = inputs[properties.L][max_nligands]
        for i in range(n_ligands):
            Z = inputs[properties.L][i][properties.Z]
            idx_i, idx_j, idx_o = self._build_neighbor_list(Z)

            inputs[properties.L][i][properties.idx_i] = idx_i.detach()
            inputs[properties.L][i][properties.idx_j] = idx_j.detach()
            inputs[properties.L][i][properties.idx_o] = idx_o.detach()
        for i in range(max_nligands - n_ligands):
            i = i + n_ligands
            inputs[properties.L][i][properties.idx_i] = torch.tensor([], dtype=torch.long)
            inputs[properties.L][i][properties.idx_j] = torch.tensor([], dtype=torch.long)
            inputs[properties.L][i][properties.idx_o] = torch.tensor([], dtype=torch.long)
        return inputs

    def _build_neighbor_list(
        self,
        Z: torch.Tensor,
    ):
        """Override with specific neighbor list implementation"""
        raise NotImplementedError



class BondStepLigandNeighborNoPadList(BondStepLigandNeighborNoPadListTransform):
    """
    Calculate neighbor list using Numpy.
    """

    def _build_neighbor_list(self, Z):
        n_atoms = len(Z)
        if n_atoms == 1:
            idx_i = np.zeros((1), dtype=int)
            idx_j = np.zeros((1), dtype=int)
        else:
            idx_i = np.tile(
                np.arange(n_atoms, dtype=int)[:, np.newaxis], (n_atoms)
            )
            idx_j = np.tile(
                np.arange(n_atoms, dtype=int)[np.newaxis], (n_atoms, 1)
            )

            idx_i = idx_i[
                ~np.eye(n_atoms, dtype=bool)
            ]
            idx_j = idx_j[
                ~np.eye(n_atoms, dtype=bool)
            ]

        idx_i = torch.from_numpy(idx_i).long()
        idx_j = torch.from_numpy(idx_j).long()
        idx_o = torch.arange(n_atoms, dtype=int)
        return idx_i, idx_j, idx_o


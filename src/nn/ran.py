from .scatter import scatter_prod
from torch import nn
from .base import Dense
import torch
from .scatter import scatter_add
from math import e as exponential

__all__ = ['RAN_no_pad', ]


class RAN_no_pad(nn.Module):
    def __init__(self, type: str = "direct_ran", scale: float = 1.0):
        super().__init__()
        self.type = type
        self.register_buffer("scale", torch.tensor(scale))
        if (self.type == "direct_ran") or (self.type == "rane"):
            pass
        elif (self.type == "sumbatch_ran") or (self.type == "summolecule_ran"):
            self.atom_env = Dense(1, 10, bias=True, activation=None)

    def ran(self, atomic_numbers, idx_i, idx_j, idx_m, batch_size, r_ij, type: str = "direct"):
        # edit distances with atomic_numbers
        a_i = atomic_numbers[idx_i]
        a_j = atomic_numbers[idx_j]
        dist_edit = torch.exp(
            -0.05 * exponential * torch.pow(
                (
                    a_i * a_j
                ), 1.01
            ) * torch.reciprocal(
                torch.pow(
                    (
                        a_i + a_j
                    ), 1.1
                )
            )
        )
        if type == 'sumbatch':
            dist_edit_sum = torch.sum(dist_edit).view(-1, 1)
            atom_mod = self.atom_env(dist_edit_sum)
            atom_mod = torch.exp(-0.005 * torch.sum(atom_mod))
            dist_edit = dist_edit + atom_mod
        elif type == 'summolecule':
            dist_edit_sum = scatter_add(dist_edit, idx_m, batch_size).view(batch_size, 1)
            atom_mod = self.atom_env(dist_edit_sum)
            atom_mod = torch.exp(-0.005 * torch.sum(atom_mod, dim=1))
            atom_mod = torch.gather(atom_mod, 0, idx_m)
            dist_edit = dist_edit + atom_mod

        # edit distances with atomic_numbers
        r_ij = r_ij + dist_edit * self.scale
        return r_ij

    def rane(self, atomic_numbers, idx_i, idx_j, idx_m, batch_size, r_ij):
        a_i = atomic_numbers[idx_i]
        a_j = atomic_numbers[idx_j]

        # atomic numbers modify with bond steps and then edit bond steps
        atom_mod = a_i * (a_j + r_ij * 0.01)
        dist_edit = torch.exp(
            -0.01 * exponential * torch.log10(
                torch.pow(
                    scatter_prod(
                        atom_mod, idx_i, atomic_numbers.shape[0]
                    ), 1.01
                ) * torch.reciprocal(
                    torch.pow(
                        scatter_add(
                            atom_mod, idx_i, atomic_numbers.shape[0]
                        ), 1.1
                    )
                )
            )
        )
        atom_num_exp = atomic_numbers + dist_edit
        a_i = atom_num_exp[idx_i]
        a_j = atom_num_exp[idx_j]
        dist_edit = torch.exp(
            -0.4 * torch.exp(
                torch.pow(
                    a_i * a_j, 1.01
                ) * torch.reciprocal(
                    torch.pow(
                        a_i + a_j, 1.1
                    )
                )
            )
        )

        # edit distances with atomic_numbers
        r_ij = r_ij + dist_edit * self.scale
        return r_ij

    def forward(self, atomic_numbers, idx_i, idx_j, idx_m, batch_size, r_ij):
        if self.type == 'direct_ran':
            outputs = self.ran(atomic_numbers, idx_i, idx_j, idx_m, batch_size, r_ij)
        elif self.type == 'sumbatch_ran':
            outputs = self.ran(atomic_numbers, idx_i, idx_j, idx_m, batch_size, r_ij, 'sumbatch')
        elif self.type == 'summolecule_ran':
            outputs = self.ran(atomic_numbers, idx_i, idx_j, idx_m, batch_size, r_ij, 'summolecule')
        elif self.type == 'rane':
            outputs = self.rane(atomic_numbers, idx_i, idx_j, idx_m, batch_size, r_ij)

        return outputs




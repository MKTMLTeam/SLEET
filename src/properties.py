from typing import Final

__all__ = ['position', 'idx_k_triples', 'bond_order', 'masses', 'shielding', 'cell', 'pred_Z', 'sm', 'offsets_lr', 'n_atoms', 'MPd', 'R', 'cell_offset', 'relative_atomic_energy', 'neighbor_mask', 'pred_idx_j', 'nbh_prediction', 'nbh_model', 'nuclear_spin_coupling', 'n_nbh_model', 'cell_strained', 'partial_charges', 'magnetic_moments', 'R_strained', 'polarizability', 'lidx_i', 'required_grad', 'spin_multiplicity', 'hessian', 'n_elec', 'strain', 'pred_r_ij', 'idx_i_triples', 'seg_m', 'pred_r_ij_idcs', 'L', 'forces', 'lidx_j', 'total_charge', 'n_L', 'total_dipole_moment', 'idx_o', 'pbc', 'Gp', 'idx_m', 'idx_j_lr', 'electric_field', 'next_Z', 'Rij_lr', 'atom_mask', 'dipole_moment', 'idx_j', 'energy', 'polarizability_derivatives', 'required_external_fields', 'LMBP', 'idx_i_lr', 'MGp', 'idx_j_triples', 'dipole_derivatives', 'distribution_r_ij', 'idx_i', 'n_nbh', 'idx_t', 'nuclear_magnetic_moments', 'Rij', 'LB', 'sol', 'n_nbh_prediction', 'r_ij', 'magnetic_field', 'Pd', 'pred_idx_m', 'nbh_placement', 'composition', 'M', 'neighbors', 'idx', 'Z', 'stress', 'n_nbh_placement', 'bond_step', 'stereo', 'distribution_Z', 'n_pred_nbh', 'offsets', ]


"""
Keys to access structure properties.

Note: Had to be moved out of Structure class for TorchScript compatibility

"""

idx: Final[str] = "_idx"

## structure
atom_mask: Final[str] = "_atom_mask"
Z: Final[str] = "_atomic_numbers"  #: nuclear charge
position: Final[str] = "_positions"  #: atom positions
R: Final[str] = position  #: atom positions
bond_step: Final[str] = "_bond_steps"
bond_order: Final[str] = "_bond_orders"
stereo: Final[str] = "_stereos"
neighbors: Final[str] = "_neighbors"
neighbor_mask: Final[str] = "_neighbor_mask"

cell: Final[str] = "_cell"  #: unit cell
cell_offset: Final[str] = "_cell_offset"
strain: Final[str] = "strain"
pbc: Final[str] = "_pbc"  #: periodic boundary conditions

seg_m: Final[str] = "_seg_m"  #: start indices of systems
idx_m: Final[str] = "_idx_m"  #: indices of systems
idx_i: Final[str] = "_idx_i"  #: indices of center atoms
idx_j: Final[str] = "_idx_j"  #: indices of neighboring atoms
idx_o: Final[str] = "_idx_o"  #: indices of max number of atoms
idx_i_lr: Final[str] = "_idx_i_lr"  #: indices of center atoms for long-range
idx_j_lr: Final[str] = "_idx_j_lr"  #: indices of neighboring atoms for long-range

lidx_i: Final[str] = "_idx_i_local"  #: local indices of center atoms (within system)
lidx_j: Final[str] = "_idx_j_local"  #: local indices of neighboring atoms (within system)
Rij: Final[str] = "_Rij"  #: vectors pointing from center atoms to neighboring atoms
Rij_lr: Final[str] = "_Rij_lr"  #: vectors pointing from center atoms to neighboring atoms for long range
n_atoms: Final[str] = "_n_atoms"  #: number of atoms
offsets: Final[str] = "_offsets"  #: cell offset vectors
offsets_lr: Final[str] = "_offsets_lr"  #: cell offset vectors for long range

R_strained: Final[str] = (
    position + "_strained"
)  #: atom positions with strain-dependence
cell_strained: Final[str] = cell + "_strained"  #: atom positions with strain-dependence

n_nbh: Final[str] = "_n_nbh"  #: number of neighbors

#: indices of center atom triples
idx_i_triples: Final[str] = "_idx_i_triples"

#: indices of first neighboring atom triples
idx_j_triples: Final[str] = "_idx_j_triples"

#: indices of second neighboring atom triples
idx_k_triples: Final[str] = "_idx_k_triples"

## chemical properties
energy: Final[str] = "energy"
forces: Final[str] = "forces"
stress: Final[str] = "stress"
masses: Final[str] = "masses"
dipole_moment: Final[str] = "dipole_moment"
total_dipole_moment: Final[str] = "total_dipole_moment"
polarizability: Final[str] = "polarizability"
hessian: Final[str] = "hessian"
dipole_derivatives: Final[str] = "dipole_derivatives"
polarizability_derivatives: Final[str] = "polarizability_derivatives"
total_charge: Final[str] = "total_charge"
partial_charges: Final[str] = "partial_charges"
spin_multiplicity: Final[str] = "spin_multiplicity"
electric_field: Final[str] = "electric_field"
magnetic_field: Final[str] = "magnetic_field"
magnetic_moments: Final[str] = "magnetic_moments"
nuclear_magnetic_moments: Final[str] = "nuclear_magnetic_moments"
shielding: Final[str] = "shielding"
nuclear_spin_coupling: Final[str] = "nuclear_spin_coupling"

## external fields needed for different response properties
required_external_fields = {
    dipole_moment: [electric_field],
    dipole_derivatives: [electric_field],
    partial_charges: [electric_field],
    polarizability: [electric_field],
    polarizability_derivatives: [electric_field],
    shielding: [magnetic_field],
    nuclear_spin_coupling: [magnetic_field],
}


required_grad = {
    energy: [],
    forces: [position],
    hessian: [position],
    dipole_moment: [electric_field],
    polarizability: [electric_field],
    dipole_derivatives: [electric_field, position],
    polarizability_derivatives: [electric_field, position],
    shielding: [magnetic_field, magnetic_moments],
}


'GSchnet property'
## structure
#: absolute pairwise distances between center atom i and neighbor atom j
r_ij: Final[str] = "_rij"
#: indices of trajectories (i.e. all atom placement steps of one system share the index)
idx_t: Final[str] = "_idx_t"

#: lists of indices to extract different neighborhoods from idx_i and idx_j given by
# different cutoffs
#: neighborhood from SchNet model cutoff
nbh_model: Final[str] = "_nbh_model"
#: neighborhood from prediction cutoff (i.e. atoms used to predict distances)
nbh_prediction: Final[str] = "_nbh_prediction"
#: neighborhood from placement cutoff (i.e. atoms considered as neighbors during
# placement)
nbh_placement: Final[str] = "_nbh_placement"

#: number of neighbors for each atom in model cutoff
n_nbh_model: Final[str] = "_n_nbh_model"
#: number of neighbors for each atom in prediction cutoff
n_nbh_prediction: Final[str] = "_n_nbh_prediction"
#: number of neighbors for each atom in placement cutoff
n_nbh_placement: Final[str] = "_n_nbh_placement"

## information required for prediction
# the type is predicted at every step (half of the time, the stop type is predicted to
# mark the focus as finished)
#: the types to predict (in n_atoms*2 steps)
pred_Z: Final[str] = "_pred_Z"
#: indices of the focus atom and its neighbors inside the prediction cutoff
pred_idx_j: Final[str] = "_pred_idx_j"
#: indices of the prediction step the atoms in pred_idx_j belong to
pred_idx_m: Final[str] = "_pred_idx_m"
#: number of atoms used for prediction in each step (i.e. count of pred_idx_m)
n_pred_nbh: Final[str] = "_n_pred_nbh"
# distances are only predicted in half of the steps (when the predicted type is not
# the stop type)
#: indices in pred_idx_j that are used to predict distances
pred_r_ij_idcs: Final[str] = "_pred_r_ij_idcs"
#: the distances between the corresponding atoms and the new atom
pred_r_ij: Final[str] = "_pred_r_ij"

#: like pred_Z, but each type is repeated n_pred_nbh times (for embedding of next types)
next_Z: Final[str] = "_next_Z"

#: distribution predicted by the model (type of the next atom)
distribution_Z: Final[str] = "_distribution_Z"
#: distribution predicted by the model (pairwise distances)
distribution_r_ij: Final[str] = "_distribution_r_ij"

## properties (for conditioning)
composition: Final[str] = "composition"  #: the atomic composition
relative_atomic_energy: Final[str] = "relative_atomic_energy"


# smiles
sm: Final[str] = "smiles"
sol: Final[str] = "solvents"

# tmQM
M: Final[str] = "metals"
MGp: Final[str] = "_metals_groups"
MPd: Final[str] = "_metals_period"
Gp: Final[str] = "_groups"
Pd: Final[str] = "_period"
L: Final[str] = "ligands"
n_L: Final[str] = "n_ligands"
LB: Final[str] = "_ligand_bonds"
LMBP: Final[str] = "_ligands_metal_binding_positions"
# tmQMg
n_elec: Final[str] = "n_electrons"

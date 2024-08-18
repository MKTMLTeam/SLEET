import numpy as np
from rdkit import Chem

__all__ = ['get_atom_numbers_from_smi', 'bond_step_gen', 'del_eye_no_reshape', 'skip_diag_strided_no_reshape', 'del_eye', 'bond_order_gen_no_pad', 'skip_diag_strided', ]


def skip_diag_strided_no_reshape(A, m):
    strided = np.lib.stride_tricks.as_strided
    s0, s1 = A.strides
    return strided(A.ravel()[1:], shape=(m-1, m), strides=(s0+s1, s1)).reshape(-1)



def skip_diag_strided(A, m):
    strided = np.lib.stride_tricks.as_strided
    s0, s1 = A.strides
    return strided(A.ravel()[1:], shape=(m-1, m), strides=(s0+s1, s1)).reshape(m, -1)




def get_atom_numbers_from_smi(smiles):
    m = Chem.MolFromSmiles(smiles)
    m2 = Chem.AddHs(m)
    atoms = None
    for atom in m2.GetAtoms():
        if atoms is not None:
            atoms = np.concatenate((atoms, np.array([atom.GetAtomicNum()])), axis=0)
        else:
            atoms = np.array([atom.GetAtomicNum()])
    atom_numbers = atoms.astype('int32')
    return atom_numbers



def del_eye_no_reshape(A, m):
    return A[
        ~np.eye(m, dtype=bool)
    ]




def del_eye(A, m):
    return A[
        ~np.eye(m, dtype=bool)
    ].reshape(m, m - 1)




def bond_step_gen(smiles):
    m = Chem.MolFromSmiles(smiles)
    m2 = Chem.AddHs(m)
    array = Chem.rdmolops.GetDistanceMatrix(m2)
    m = array.shape[0]
    if m < 75:
        array = del_eye(array, m)
    else:
        array = skip_diag_strided(array, m)
    return array




def bond_order_gen_no_pad(m):
    bos = {
        Chem.BondType.SINGLE: 1.0,
        Chem.BondType.DOUBLE: 2.0,
        Chem.BondType.TRIPLE: 3.0,
        Chem.BondType.AROMATIC: 1.5,
        Chem.BondType.UNSPECIFIED: 0.0
    }
    N_atoms = m.GetNumAtoms()
    if N_atoms != 1:
        array = np.zeros((N_atoms, N_atoms))
        bondorder = np.array([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bos[bond.GetBondType()]] for bond in m.GetBonds()]).transpose()
        indice0 = np.concatenate([bondorder[0].astype('int32'), bondorder[1].astype('int32')])
        indice1 = np.concatenate([bondorder[1].astype('int32'), bondorder[0].astype('int32')])
        array[indice0, indice1] = np.concatenate([bondorder[2], bondorder[2]])[np.arange(bondorder[2].shape[0] * 2)]
        m = array.shape[0]
        if m < 75:
            array = del_eye_no_reshape(array, m)
        else:
            array = skip_diag_strided_no_reshape(array, m)
    else:
        array = np.zeros((N_atoms))
    return array



from rdkit import RDLogger
from rdkit import Chem
import numpy as np
from rdkit.Chem.rdchem import KekulizeException
from src.transform import bond_order_gen_no_pad
from src.transform import build_3d_from_bondstep_bondpath_stereo
from rdkit.Chem import rdmolops
from src import properties as structure
import traceback

__all__ = ['analysis_ligands', 'tmQM_smiles_fix', 'metals', 'cal_stereo_from_Mol', ]


def tmQM_smiles_fix(smiles):
    RDLogger.DisableLog('rdApp.*')
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        smiles = smiles.replace("[NH3]", "[NH3+]")
        smiles = smiles.replace("[NH2]", "[NH2+]")
        smiles = smiles.replace("[NH]", "[NH+]")
        smiles = smiles.replace("[N@@H]", "[N@@H+]")
        smiles = smiles.replace("[N@H]", "[N@H+]")
        smiles = smiles.replace("[N]", "[N+]")
        smiles = smiles.replace("[N@]", "[N@+]")
        smiles = smiles.replace("[N@@]", "[N@@+]")
        smiles = smiles.replace("[BH2]", "[BH2-]")
        smiles = smiles.replace("[B@@H]", "[B@@H-]")
        smiles = smiles.replace("[B@H]", "[B@H-]")
        smiles = smiles.replace("[BH]", "[BH-]")
        smiles = smiles.replace("[BH4]", "[BH4-]")
        smiles = smiles.replace("[BH3]", "[BH3-]")
        smiles = smiles.replace("[B]", "[B-]")
        smiles = smiles.replace("[B@@]", "[B@@-]")
        smiles = smiles.replace("[B@]", "[B@-]")
        m = Chem.MolFromSmiles(smiles)
    if m is None:
        smiles = smiles.replace("[C]", "[C+]")
        smiles = smiles.replace("[CH]", "[CH+]")
        smiles = smiles.replace("[CH2]", "[CH2+]")
        m = Chem.MolFromSmiles(smiles)
    if m is None:
        smiles = smiles.replace("N1", "[N+2]1")
        m = Chem.MolFromSmiles(smiles)
    if m is None:
        smiles = smiles.replace("[O]", "[O+]")
        m = Chem.MolFromSmiles(smiles)
    if m is None:
        smiles = smiles.replace("N4", "[N+2]4")
        m = Chem.MolFromSmiles(smiles)
    try:
        Chem.AddHs(m)
    except Exception:
        raise TypeError(smiles)
    return smiles



metals = {
    'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc',
    'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
    'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
    'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm',
    'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
    'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'Fr', 'Ra',
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf',
    'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
    'Nh', 'Fl', 'Mc', 'Lv'
}




def cal_stereo_from_Mol(m):
    stereo = np.zeros(m.GetNumAtoms(), dtype='int32')
    si = Chem.FindPotentialStereo(m)
    for element in si:
        idx = element.centeredOn
        if str(element.descriptor) == 'Bond_Trans':
            stereo[idx:idx+2] = 1
        elif str(element.descriptor) == 'Bond_Cis':
            stereo[idx:idx+2] = 2
    for chi in Chem.FindMolChiralCenters(m, force=True, includeUnassigned=True, useLegacyImplementation=True):
        if chi[1] == 'R':
            stereo[chi[0]] = 3
        elif chi[1] == 'S':
            stereo[chi[0]] = 4
        elif chi[1] == 'r':
            stereo[chi[0]] = 5
        elif chi[1] == 's':
            stereo[chi[0]] = 6

    return stereo



def analysis_ligands(smiles, char_table, char_plus, separated_ligands, only_pos):
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    for atom in m.GetAtoms():
        if atom.GetSymbol() in metals:
            conn = {nbr.GetIdx() for nbr in atom.GetNeighbors()}
            center_idx = atom.GetIdx()
    em = Chem.EditableMol(m)
    for i in conn:
        em.RemoveBond(center_idx, i)
    p = em.GetMol()
    while True:
        try:
            p = Chem.RemoveHs(p)
        except KekulizeException:
            tb = traceback.format_exc()
            unk_list = tb[tb.find("Unkekulized atoms:")+19:].split(" ")
            tb1_len = len(tb)
            p.GetAtomWithIdx(int(unk_list[0])).SetNumExplicitHs(1)
            try:
                p = Chem.RemoveHs(p)
            except KekulizeException:
                tb = traceback.format_exc()
                tb2_len = len(tb)
                if tb[tb.find("Unkekulized atoms:", tb1_len)+19:].split(" ") == unk_list:
                    unk_list = tb[tb.find("Unkekulized atoms:", tb1_len)+19:].split(" ")
                    if '\n' in unk_list[0]:
                        unk_list[0] = unk_list[0].split("\n")[0]
                    p.GetAtomWithIdx(int(unk_list[0])).SetNumExplicitHs(2)
                    try:
                        p = Chem.RemoveHs(p)
                    except KekulizeException:
                        tb = traceback.format_exc()
                        if tb[tb.find("Unkekulized atoms:", tb2_len)+19:].split(" ") == unk_list:
                            unk_list = tb[tb.find("Unkekulized atoms:", tb2_len)+19:].split(" ")
                            p.GetAtomWithIdx(int(unk_list[0])).SetNumExplicitHs(3)

        else:
            tb = None
        if tb is None:
            break
    Chem.SanitizeMol(p)
    smis = [x for x in Chem.GetMolFrags(p, asMols=True)]
    # insert complexing agent information
    smis_conn = {i: 0 for i in rdmolops.GetMolFrags(p)}
    smis_conn_array = {i: np.zeros_like(np.arange(len(i))) for i in rdmolops.GetMolFrags(p)}
    if separated_ligands:
        smis_conn_idx = {i: [] for i in rdmolops.GetMolFrags(p)}
    for con in conn:
        for k in smis_conn.keys():
            if con in k:
                smis_conn[k] += 1
                smis_conn_array[k][k.index(con)] = 1
                if separated_ligands:
                    smis_conn_idx[k].append(k.index(con))
    list_conn = list(smis_conn.values())
    list_conn_array = list(smis_conn_array.values())
    if separated_ligands:
        list_conn_idx = list(smis_conn_idx.values())

    ligands = []
    for i, smi in enumerate(smis):
        m2 = Chem.AddHs(smi)
        syms = [m2.GetAtomWithIdx(j).GetSymbol() for j in rdmolops.GetMolFrags(m2)[0]]
        if len(syms) == 1 and (syms[0] in metals):
            at = np.array([syms[0]])
            metal = np.array(np.where(at == char_table[:, 0])[0], dtype='int32')
            continue
        ligand = {}

        try:
            stereo = cal_stereo_from_Mol(m2)
        except ValueError:
            atoms = None
            for atom in m2.GetAtoms():
                if atoms is not None:
                    atoms = np.concatenate((atoms, np.array([atom.GetAtomicNum()])), axis=0)
                else:
                    atoms = np.array([atom.GetAtomicNum()])
            target_number = atoms.astype('int64')
            m2 = Chem.MolFromSmiles(Chem.MolToSmiles(m2))
            m2 = Chem.AddHs(m2)
            atoms = None
            for atom in m2.GetAtoms():
                if atoms is not None:
                    atoms = np.concatenate((atoms, np.array([atom.GetAtomicNum()])), axis=0)
                else:
                    atoms = np.array([atom.GetAtomicNum()])
            atoms = atoms.astype('int64')
            stereo = cal_stereo_from_Mol(m2)
            if any(atoms != target_number):
                tm = np.zeros((atoms.shape[0], atoms.shape[0]))
                for k in range(1, 99):
                    t = np.where(target_number == k)
                    n = np.where(atoms == k)
                    tm[n, t] = 1
                stereo = np.dot(stereo, tm).astype('int64')
        xyz = build_3d_from_bondstep_bondpath_stereo(m2)

        array = Chem.rdmolops.GetDistanceMatrix(m2)
        bond_order = bond_order_gen_no_pad(m2)
        if len(array) != 1:
            array = array[array != 0]
        else:
            array = np.array([255.])

        atoms = None
        for atom in m2.GetAtoms():
            if atoms is not None:
                atoms = np.concatenate((atoms, np.array([atom.GetAtomicNum()])), axis=0)
            else:
                atoms = np.array([atom.GetAtomicNum()])
        atom_numbers = atoms.astype('int32')

        if separated_ligands and (list_conn[i] > 0):
            for conn_idx in range(list_conn[i]):
                conn_tmp_array = np.zeros_like(np.arange(len(syms)))
                if only_pos:
                    conn_tmp_array[list_conn_idx[i][conn_idx]] = 1
                else:
                    conn_tmp_array[list_conn_idx[i][conn_idx]] = atom_numbers[list_conn_idx[i][conn_idx]]
                ligand[structure.Z] = atom_numbers
                ligand[structure.Gp] = char_plus[atom_numbers][:, 1]
                ligand[structure.Pd] = char_plus[atom_numbers][:, 2]
                ligand[structure.R] = xyz
                ligand[structure.LB] = np.array([list_conn[i] - 1])
                ligand[structure.LMBP] = conn_tmp_array
                ligand[structure.bond_step] = array
                ligand[structure.bond_order] = bond_order
                ligand[structure.stereo] = stereo
                ligands.append(ligand)
        else:
            conn_tmp_array = np.zeros_like(np.arange(len(syms)))
            conn_tmp_array[:len(list_conn_array[i])] = list_conn_array[i]
            ligand[structure.Z] = atom_numbers
            ligand[structure.Gp] = char_plus[atom_numbers][:, 1]
            ligand[structure.Pd] = char_plus[atom_numbers][:, 2]
            ligand[structure.R] = xyz
            if list_conn[i] > 0:
                ligand[structure.LB] = np.array([list_conn[i] - 1])
            else:
                ligand[structure.LB] = np.array([list_conn[i]])
            ligand[structure.LMBP] = conn_tmp_array
            ligand[structure.bond_step] = array
            ligand[structure.bond_order] = bond_order
            ligand[structure.stereo] = stereo
            ligands.append(ligand)
    return metal, ligands


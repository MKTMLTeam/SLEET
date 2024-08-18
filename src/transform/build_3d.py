from rdkit.Chem.rdmolops import GetDistanceMatrix

__all__ = ['build_3d_from_bondstep_bondpath_stereo', ]


def build_3d_from_bondstep_bondpath_stereo(m):
    # atoms_list = [atom.GetIdx() for atom in m.GetAtoms()]
    # n_atoms = m.GetNumAtoms()
    #
    # stereo = np.ones(m.GetNumAtoms(), dtype=int)
    # si = Chem.FindPotentialStereo(m)
    # for element in si:
    #     idx = element.centeredOn
    #     if str(element.descriptor) == 'Bond_Trans':
    #         stereo[idx:idx+2] = 2
    #     elif str(element.descriptor) == 'Bond_Cis':
    #         stereo[idx:idx+2] = 3
    # for x in Chem.FindMolChiralCenters(m, force=True, includeUnassigned=True, useLegacyImplementation=True):
    #     if x[1] == 'R':
    #         stereo[x[0]] = 4
    #     elif x[1] == 'S':
    #         stereo[x[0]] = 5
    #     elif x[1] == 'r':
    #         stereo[x[0]] = 6
    #     elif x[1] == 's':
    #         stereo[x[0]] = 7
    #
    # total_stereo_list = np.zeros((n_atoms, n_atoms))
    # total_path_list = np.zeros((n_atoms, n_atoms))
    # for i in atoms_list:
    #     both_list = [
    #         [
    #             pow(prod(x[0]), 1.01) / sum(x[0]), exp(-0.01 * exp(1) * log10(pow(prod(x[1]), 1.01) / sum(x[1])))] for x in [
    #                 [
    #                     [
    #                         stereo[a] for a in GetShortestPath(m, atoms_list[i], atoms_list[j])
    #                     ], [
    #                         m.GetAtomWithIdx(a).GetAtomicNum() for a in GetShortestPath(m, atoms_list[i], atoms_list[j])
    #                     ]
    #                 ] for j in atoms_list if j != i
    #         ]
    #     ]
    #     for j in atoms_list:
    #         if j > i:
    #             total_stereo_list[i, j] = both_list[j-1][0]
    #             total_path_list[i, j] = both_list[j-1][1]
    #         elif j < i:
    #             total_stereo_list[i, j] = both_list[j][0]
    #             total_path_list[i, j] = both_list[j][1]
    #         else:
    #             continue

    bondstep = GetDistanceMatrix(m)
    # new_xyz = np.concatenate([bondstep, np.array(total_path_list), np.array(total_stereo_list)], axis=1)
    return bondstep


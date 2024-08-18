import torch
from src.nn import Dense
from torch import nn
from typing import Callable
from src.nn import scatter_add
from src.nn.activations import shifted_softplus
from src.nn import replicate_module
from src import properties as structure
from src.atomistic import SubAtomwise
from typing import Dict
from src.nn import RAN_no_pad
from src.nn import MultiHeadAttentionLayer

__all__ = ['InteractionBlock', 'MetalEmbedding', 'SLEET', 'SubSchNet_bs', ]


class MetalEmbedding(nn.Module):
    def __init__(
        self,
        max_z: int,
        n_metal_basis: int,
        n_embed_out: int,
    ):
        super().__init__()
        self.src_embedding = nn.Embedding(max_z, n_metal_basis, padding_idx=0)
        self.gp_embedding = nn.Embedding(20, n_metal_basis, padding_idx=0)
        self.pd_embedding = nn.Embedding(8, n_metal_basis, padding_idx=0)
        self.out = nn.Sequential(
            Dense(n_metal_basis * 3, n_metal_basis * 3, activation=nn.ReLU()),
            Dense(n_metal_basis * 3, n_embed_out)
        )

    def forward(self, metals: torch.Tensor, mgp: torch.Tensor, mpd: torch.Tensor):
        metals = self.src_embedding(metals)
        mgp = self.gp_embedding(mgp)
        mpd = self.pd_embedding(mpd)
        metals = torch.cat([metals, mgp, mpd], dim=-1)
        metals = self.out(metals)
        return metals



class InteractionBlock(nn.Module):
    r"""SchNet interaction block for modeling interactions of atomistic systems."""

    def __init__(
        self,
        n_atom_basis: int,
        n_rbf: int,
        n_filters: int,
        activation: Callable = shifted_softplus,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            n_rbf (int): number of radial basis functions.
            n_filters: number of filters used in continuous-filter convolution.
            activation: if None, no activation function is used.
        """
        super().__init__()
        self.in2f = Dense(n_atom_basis, n_filters, bias=False, activation=None)
        self.f2out = nn.Sequential(
            Dense(n_filters, n_atom_basis, activation=activation),
            Dense(n_atom_basis, n_atom_basis, activation=None),
        )
        self.filter_network = nn.Sequential(
            Dense(n_rbf, n_filters, activation=activation), Dense(n_filters, n_filters)
        )

    def forward(
        self,
        x: torch.Tensor,
        f_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rcut_ij: torch.Tensor,
    ):
        """Compute interaction output.

        Args:
            x: input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            atom features after interaction
        """
        x = self.in2f(x)
        Wij = self.filter_network(f_ij)
        Wij = Wij * rcut_ij[:, None]

        # continuous-filter convolution
        x_j = x[idx_j]
        x_ij = x_j * Wij
        x = scatter_add(x_ij, idx_i, dim_size=x.shape[0])

        x = self.f2out(x)
        return x



class SubSchNet_bs(nn.Module):
    """SchNet architecture for learning representations of atomistic systems

    References:

    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum
       interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.

    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        n_out: int,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        aggregation_mode: str,
        n_filters: int = None,
        shared_interactions: bool = False,
        max_z: int = 100,
        activation: Callable = shifted_softplus,
        use_stereos: bool = False,
        use_ran: bool = False,
        ran: nn.Module = None,
        embedding: nn.Module = None,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            n_filters: number of filters used in continuous-filter convolution
            shared_interactions: if True, share the weights across
                interaction blocks and filter-generating networks.
            max_z: maximal nuclear charge
            activation: activation function
        """
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.n_filters = n_filters or self.n_atom_basis
        self.use_stereos = use_stereos
        self.use_ran = use_ran
        self.ran = ran
        self.stereo_ran = RAN_no_pad("rane", 0.1)
        # layer for expanding interatomic distances in a basis
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.n_out = n_out

        # layers
        self.embedding = embedding

        self.binding_sides_embed = nn.Embedding(2, n_atom_basis, padding_idx=0)

        self.interactions = replicate_module(
            lambda: InteractionBlock(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=activation,
            ),
            n_interactions,
            shared_interactions,
        )
        self.interactions_bo = replicate_module(
            lambda: InteractionBlock(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=activation,
            ),
            n_interactions,
            shared_interactions,
        )
        self.interactions_out = replicate_module(
            lambda: Dense(n_atom_basis * 2, n_atom_basis, activation=nn.ReLU()),
            n_interactions,
            shared_interactions,
        )
        self.out_net = SubAtomwise(n_atom_basis, n_out, aggregation_mode=aggregation_mode)

    def forward(self, inputs: Dict[str, torch.Tensor], batch_size: int):
        atomic_numbers = inputs[structure.Z]
        bond_steps = inputs[structure.bond_step]
        bond_orders = inputs[structure.bond_order]
        ligand_bonds = inputs[structure.LB]
        binding_sides = inputs[structure.LMBP]
        idx_i = inputs[structure.idx_i]
        idx_j = inputs[structure.idx_j]
        idx_m = inputs[structure.idx_m]
        n_atoms = inputs[structure.n_atoms]

        if self.use_ran:
            B = n_atoms.shape[0]
            idx_m_nbh = torch.repeat_interleave(
                torch.arange(len(n_atoms), device=n_atoms.device), n_atoms * (n_atoms - 1), dim=0
            )
            if self.use_stereos:
                stereos = inputs[structure.stereo] + 1
                bond_steps = self.stereo_ran(stereos, idx_i, idx_j, idx_m_nbh, B, bond_steps)
            bond_steps = self.ran(atomic_numbers, idx_i, idx_j, idx_m_nbh, B, bond_steps)
        else:
            bond_steps = self.ran(bond_steps)

        # compute atom and pair features
        x = self.embedding(atomic_numbers)
        x_binding = self.binding_sides_embed(binding_sides)
        x += x_binding
        f_ij = self.radial_basis(bond_steps)
        f_bo_ij = self.radial_basis(bond_orders)
        bond_stepscut_ij = self.cutoff_fn(bond_steps)
        bocut_ij = self.cutoff_fn(bond_orders)

        # compute interaction block to update atomic embeddings
        for interaction, interaction_bo, out in zip(self.interactions, self.interactions_bo, self.interactions_out):
            v = interaction(x, f_ij, idx_i, idx_j, bond_stepscut_ij)
            v_bo = interaction_bo(x, f_bo_ij, idx_i, idx_j, bocut_ij)
            x = torch.cat([x] * 2, dim=1) + torch.cat([v, v_bo], dim=1)
            x = out(x)

        x = self.out_net(x, idx_m, n_atoms)
        x[x == 0] = 1e-8

        dim1 = torch.arange(0, x.size(0), device=x.device)
        max_lb = max(ligand_bonds) + 1
        tmp = torch.zeros(max_lb, x.size(0), self.n_out, device=x.device)
        dim_list = [dim1[ligand_bonds >= i] for i in range(max_lb)]
        idx_d0 = torch.repeat_interleave(
            torch.arange(max_lb, device=x.device), torch.tensor([len(t) for t in dim_list], device=x.device)
        )
        idx_d1 = torch.cat(dim_list)
        tmp[idx_d0, idx_d1] = torch.repeat_interleave(
            x, max_lb, dim=0
        ).view(x.size(0), max_lb, x.size(1)).permute(1, 0, 2)[idx_d0, idx_d1]
        x = tmp.to(x.device)

        if x.shape[1] != batch_size:
            x = torch.cat(
                [x, torch.zeros((x.size(0), batch_size - x.size(1), self.n_out), dtype=x.dtype, device=x.device)],
                dim=1
            )

        return x



class SLEET(nn.Module):
    def __init__(
        self,
        n_atom_basis: int,
        n_metal_basis: int,
        n_interactions: int,
        n_embed_out: int,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        n_lembed: int = 8,
        aggregation_mode: str = "avg",
        same_embed: bool = False,
        n_filters: int = None,
        shared_interactions: bool = False,
        max_z: int = 100,
        activation: Callable = shifted_softplus,
        use_stereos: bool = False,
        ran: nn.Module = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        if ran is None:
            self.ran = nn.Identity()
            self.use_ran = False
        else:
            self.use_ran = True
            self.ran = ran
        self.n_embed_out = n_embed_out
        self.embedding = nn.Embedding(100, n_atom_basis, padding_idx=0)
        self.ligand_embeddings = replicate_module(
            lambda: SubSchNet_bs(
                shared_interactions=shared_interactions,
                embedding=self.embedding,
                aggregation_mode=aggregation_mode,
                n_atom_basis=n_atom_basis,
                n_interactions=n_interactions,
                radial_basis=radial_basis,
                n_filters=n_filters,
                use_stereos=use_stereos,
                activation=activation,
                use_ran=self.use_ran,
                cutoff_fn=cutoff_fn,
                n_out=n_embed_out,
                ran=self.ran,
                max_z=max_z,
            ),
            n_lembed,
            same_embed,
        )

        self.metal_embedding = MetalEmbedding(
            max_z, n_metal_basis, n_embed_out
        )
        self.temperature = temperature
        self.relativeselfattn = MultiHeadAttentionLayer(n_embed_out, 8)
        self.relativeattn = MultiHeadAttentionLayer(n_embed_out, 8)
        self.ffd = Dense(n_embed_out, n_embed_out, activation=nn.SiLU())

    def forward(self, inputs: Dict[str, torch.Tensor]):
        metals = inputs[structure.M].unsqueeze(0)
        MGp = inputs[structure.MGp].unsqueeze(0)
        Mpd = inputs[structure.MPd].unsqueeze(0)
        ligands = inputs[structure.L]
        n_L = inputs[structure.n_L]
        vl_cat = []
        B = metals.shape[1]

        x = self.metal_embedding(metals, MGp, Mpd)

        for i, ligand_embed in enumerate(self.ligand_embeddings):
            if ligands[i][structure.idx_m].size(0) == 0:
                continue

            vl = ligand_embed(ligands[i], B)
            vl_cat.append(vl)

        vl_cat = torch.cat(vl_cat, dim=0)

        # repad vl_cat
        vl_cat_T = vl_cat.permute(1, 0, 2)
        tmp = torch.zeros_like(vl_cat_T)
        max_size = max(torch.sum((vl_cat[..., 0] != 0), dim=1))
        te = torch.repeat_interleave(torch.arange(0, vl_cat.size(0), device=vl_cat.device)[:, None], max_size, dim=1).transpose(0, 1)
        te[te >= n_L[:, None]] = -1
        nonzeros_mask = (te != -1)
        zeros_mask = (vl_cat_T[..., 0] == 0)
        tmp[nonzeros_mask] = vl_cat_T[~zeros_mask]
        vl_cat = tmp.permute(1, 0, 2)[:max(n_L)]

        vl_cat_attn = self.relativeselfattn(vl_cat / self.temperature, vl_cat, vl_cat)
        vl_cat = vl_cat + vl_cat_attn
        vl_cat = self.relativeattn(x / self.temperature, vl_cat, vl_cat)
        vl_cat = self.ffd(vl_cat)
        x = x + vl_cat
        x = x.view(x.size(0) * x.size(1), x.size(2))

        inputs["scalar_representation"] = x
        return inputs




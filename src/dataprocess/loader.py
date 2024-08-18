import torch
from src import properties as structure
from torch.utils.data.dataloader import T_co
from torch.utils.data import Dataset
from typing import Sequence
from torch.utils.data import Sampler
from torch.utils.data.dataloader import _collate_fn_t
from typing import Optional
from torch.utils.data import DataLoader

__all__ = ['AtomsLoader', '_ligands_collate_fn', '_atoms_collate_fn', ]


def _ligands_collate_fn(batch):
    """
    Build batch from systems and properties & apply padding

    Args:
        examples (list):

    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    max_nligands = 14
    batch = sorted(batch, key=lambda x: x[structure.L][max_nligands], reverse=True)
    idx_keys = {structure.idx_i, structure.idx_j, structure.idx_o}
    elem = batch[0]
    if "label_regression" in elem.keys():
        label_regression_max_size = max({d["label_regression"].shape[0] for d in batch})
    if "digit" in elem.keys():
        digit_max_size = max({d["digit"].shape[0] for d in batch})

    lkeys = {key for key in elem[structure.L][0]}

    coll_batch = {}
    coll_batch[structure.L] = [{} for i in range(max_nligands)]
    for key in elem:
        if (key not in idx_keys) and (key != structure.L) and (key != "label_regression") \
         and (key != "digit"):
            coll_batch[key] = torch.cat([d[key] for d in batch], 0)
        elif key == "label_regression":
            coll_batch[key] = torch.stack(
                [torch.cat(
                    [d[key], torch.zeros(label_regression_max_size - d[key].shape[0], dtype=torch.long)]
                ) for d in batch],
                1
            )
        elif key == "digit":
            coll_batch[key] = torch.stack(
                [torch.cat(
                    [d[key], torch.zeros(digit_max_size - d[key].shape[0], dtype=torch.long)]
                ) for d in batch],
                1
            )

    coll_batch[structure.idx_m] = torch.arange(len(batch))
    coll_batch[structure.n_L] = torch.LongTensor([d[structure.n_L] for d in batch])

    for i in range(max_nligands):
        max_size = max({d[structure.L][i][structure.R].shape[-1] for d in batch})
        max_n_atoms = max({d[structure.L][i][structure.n_atoms] for d in batch})
        for lkey in lkeys:
            if (lkey not in idx_keys) and (lkey != structure.R):
                coll_batch[structure.L][i][lkey] = torch.cat([d[structure.L][i][lkey] for d in batch], 0)
            elif lkey == structure.R:
                coll_batch[structure.L][i][lkey] = torch.cat(
                    [
                        torch.cat([
                            d[structure.L][i][lkey], torch.zeros(
                                d[structure.L][i][lkey].shape[0],
                                max_size - d[structure.L][i][lkey].shape[1]
                            ) if len(d[structure.L][i][lkey]) > 0 else torch.tensor([])
                        ], 1) for d in batch
                    ],
                    0,
                )

        seg_m = torch.cumsum(coll_batch[structure.L][i][structure.n_atoms], dim=0)
        seg_m = torch.cat([torch.zeros((1,), dtype=seg_m.dtype), seg_m], dim=0)
        seg_o = torch.cumsum(
            torch.repeat_interleave(
                torch.tensor([max_n_atoms]), repeats=len(coll_batch[structure.L][i][structure.n_atoms]), dim=0
            ),
            dim=0
        )
        seg_o = torch.cat([torch.zeros((1,), dtype=seg_o.dtype), seg_o], dim=0)
        idx_m = torch.repeat_interleave(
            torch.arange(len(batch)), repeats=coll_batch[structure.L][i][structure.n_atoms], dim=0
        )
        coll_batch[structure.L][i][structure.idx_m] = idx_m
        coll_batch[structure.L][i][structure.n_atoms] = coll_batch[structure.L][i][structure.n_atoms][
            coll_batch[structure.L][i][structure.n_atoms] != 0
        ]

        for key in idx_keys:
            if key in lkeys:
                if key != structure.idx_o:
                    coll_batch[structure.L][i][key] = torch.cat(
                        [d[structure.L][i][key] + off for d, off in zip(batch, seg_m)], 0
                    )
                else:
                    coll_batch[structure.L][i][key] = torch.cat(
                        [d[structure.L][i][key] + off for d, off in zip(batch, seg_o)], 0
                    )

    return coll_batch



def _atoms_collate_fn(batch):
    """
    Build batch from systems and properties & apply padding

    Args:
        examples (list):

    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    elem = batch[0]
    idx_keys = {structure.idx_i, structure.idx_j, structure.idx_i_triples}
    # Atom triple indices must be treated separately
    idx_triple_keys = {structure.idx_j_triples, structure.idx_k_triples}

    coll_batch = {}
    for key in elem:
        if (key not in idx_keys) and (key not in idx_triple_keys):
            coll_batch[key] = torch.cat([d[key] for d in batch], 0)
        elif key in idx_keys:
            coll_batch[key + "_local"] = torch.cat([d[key] for d in batch], 0)

    seg_m = torch.cumsum(coll_batch[structure.n_atoms], dim=0)
    seg_m = torch.cat([torch.zeros((1,), dtype=seg_m.dtype), seg_m], dim=0)
    idx_m = torch.repeat_interleave(
        torch.arange(len(batch)), repeats=coll_batch[structure.n_atoms], dim=0
    )
    coll_batch[structure.idx_m] = idx_m

    for key in idx_keys:
        if key in elem.keys():
            coll_batch[key] = torch.cat(
                [d[key] + off for d, off in zip(batch, seg_m)], 0
            )

    # Shift the indices for the atom triples
    for key in idx_triple_keys:
        if key in elem.keys():
            indices = []
            offset = 0
            for idx, d in enumerate(batch):
                indices.append(d[key] + offset)
                offset += d[structure.idx_j].shape[0]
            coll_batch[key] = torch.cat(indices, 0)

    return coll_batch




class AtomsLoader(DataLoader):
    """Data loader for subclasses of BaseAtomsData"""

    def __init__(
        self,
        dataset: Dataset[T_co],
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler[int]] = None,
        batch_sampler: Optional[Sampler[Sequence[int]]] = None,
        num_workers: int = 0,
        collate_fn: _collate_fn_t = _atoms_collate_fn,
        pin_memory: bool = False,
        **kwargs
    ):
        super(AtomsLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            **kwargs
        )


from .base import Transform
from src.utils import as_dtype
from src import properties
import torch
from typing import Dict

__all__ = ['CastMap', 'CastTo32', ]


class CastMap(Transform):
    """
    Cast all inputs according to type map.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = True

    def __init__(self, type_map: Dict[str, str]):
        """
        Args:
            type_map: dict with source_type: target_type (as strings)
        """
        super().__init__()
        self.type_map = type_map

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if properties.L in inputs.keys():
            l_limit = {i for i in range(14)}
        for k, v in inputs.items():
            if k == properties.L:
                for i in l_limit:
                    for lk, lv in v[i].items():
                        vdtype = str(lv.dtype).split(".")[-1]
                        if vdtype in self.type_map:
                            inputs[k][i][lk] = lv.to(dtype=as_dtype(self.type_map[vdtype]))
            else:
                vdtype = str(v.dtype).split(".")[-1]
                if vdtype in self.type_map:
                    inputs[k] = v.to(dtype=as_dtype(self.type_map[vdtype]))
        return inputs



class CastTo32(CastMap):
    """Cast all float64 tensors to float32"""

    def __init__(self):
        super().__init__(type_map={"float64": "float32"})





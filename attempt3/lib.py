from typing import Type
import cutlass
from cutlass import cute
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils

# this is the library we'd import, generated code can use any functions from this lib

def get_smem_layout(
        major_mode: LayoutEnum, dtype: Type[cutlass.Numeric],
        major_mode_size: int, modes: tuple, mode_order: tuple):
    atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(major_mode, dtype, major_mode_size),
        dtype
    )
    layout = cute.tile_to_shape(
        atom, modes, mode_order
    )
    return layout

def get_tiled_mma(a_dtype, b_dtype, a_major_mode, b_major_mode, acc_dtype, tile_shape):
    # if tile shape is none then you just do the entire tile size
    pass
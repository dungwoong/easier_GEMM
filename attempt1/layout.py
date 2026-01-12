from typing import List, Tuple, Type
from enum import Enum

import cutlass
from cutlass import cute
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils


class GemmPart(Enum):
    A = 'A'
    B = 'B'


class SwizzledSMEM:
    def __init__(self, name, gemm_type: GemmPart, major_mode: LayoutEnum, gemm_mnk: tuple, dtype: Type[cutlass.Numeric], stages: int):
        assert dtype.width == 16, 'Only 16-bit types are supported right now'
        self.layout = self._get_layout(gemm_type, major_mode, gemm_mnk, dtype, stages)
        self.gemm_mnk = gemm_mnk
        self.buffer_align_bytes = 1024
        self.dtype = dtype
        self.stages = stages
        self.tensor = None
        self.gemm_type = gemm_type
        self.tile_shape = (gemm_mnk[0], gemm_mnk[2]) if gemm_type == GemmPart.A else (gemm_mnk[1], gemm_mnk[2])

        self.name = name
        self.storage_type = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(self.layout)], self.buffer_align_bytes]
    
    def populate_ptr(self, shared_storage):
        self.tensor = getattr(shared_storage, self.name).get_tensor(self.layout.outer, swizzle=self.layout.inner)

    def _get_layout(self, gemm_type, major_mode, tile_shape_mnk, dtype, stages):
        assert gemm_type in (GemmPart.A, GemmPart.B)
        if gemm_type == GemmPart.A:
            return sm90_utils.make_smem_layout_a(major_mode, tile_shape_mnk, dtype, stages)
        else:
            return sm90_utils.make_smem_layout_b(major_mode, tile_shape_mnk, dtype, stages)
    
    @property
    def layout_one_stage(self):
        return cute.slice_(self.layout, (None, None, 0))

    def get_one_stage_bytes(self):
        return cute.size_in_bytes(self.dtype, self.layout_one_stage)
    
    def __str__(self):
        return 'SMEMT(' + str(self.tensor) + ')'


class GMEMTensor:
    def __init__(self, tensor: cute.Tensor):
        self.tensor = tensor
        assert cute.rank(tensor) == 2, 'Only support 2D tensors for now'
    
    @property
    def rows(self):
        return cute.size(self.tensor, mode=[0])

    @property
    def cols(self):
        return cute.size(self.tensor, mode=[1])
    
    @property
    def major_mode(self):
        return LayoutEnum.from_tensor(self.tensor)
    
    def __str__(self):
        return 'GMEMT(' + str(self.tensor) + ')'


class TMACopyG2S:
    # No multicasting for now, handle that later
    def __init__(self, src_gmem: GMEMTensor, dst_smem: SwizzledSMEM, cta_coord_mn: Tuple[int, int]):
        self.src_gmem, self.dst_smem = src_gmem, dst_smem
        self.op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        self.tma_load_bytes = dst_smem.get_one_stage_bytes()
        self.tma_atom, self.tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(self.op, src_gmem.tensor, dst_smem.layout_one_stage, dst_smem.tile_shape, num_multicast=1)
        
        
        mn_dim_idx = 0 if dst_smem.gemm_type == GemmPart.A else 1
        self.partitioned_gmem = cute.local_tile(self.tma_tensor, cute.select(dst_smem.gemm_mnk, [mn_dim_idx, 2]), (cta_coord_mn[mn_dim_idx], None))
        self.k_iters = cute.size(self.partitioned_gmem, mode=[2]) # MN, K, restK
        self.copy_fn = self.get_copy_fn()
    
    def get_copy_fn(self, **kwargs):
        s, g = cute.nvgpu.cpasync.tma_partition(self.tma_atom, 0, cute.make_layout(1), 
                                                cute.group_modes(self.dst_smem.tensor, 0, cute.rank(self.dst_smem.tensor) - 1),
                                                cute.group_modes(self.partitioned_gmem, 0, cute.rank(self.partitioned_gmem) - 1)
                                                )
        print(s)
        print(g)
        def copy_tma(src_idx, dst_idx, **kwargs2):
            cute.copy(self.tma_atom, g[None, src_idx], s[None, dst_idx], **kwargs2, **kwargs)
        return copy_tma
    
    @cute.jit
    def prefetch_descriptor(self):
        cute.nvgpu.cpasync.prefetch_descriptor(self.tma_atom)

from cutlass import cute

from typing import Tuple

from layout import *


class Test:
    def __init__(self):
        self.ab_stage = 2
        self.tile_shape_mnk = (128, 256, 64)
        self.dtype = cutlass.BFloat16
        self.shared_storage = None
    
    @cute.kernel
    def kernel(self, a: cute.Tensor, b: cute.Tensor):
        gmem_A = GMEMTensor(a)
        gmem_B = GMEMTensor(b)

        smem_A = SwizzledSMEM('sA', GemmPart.A, gmem_A.major_mode, self.tile_shape_mnk, self.dtype, 2)
        smem_B = SwizzledSMEM('sB', GemmPart.B, gmem_B.major_mode, self.tile_shape_mnk, self.dtype, 2)

        self._populate_smem([smem_A, smem_B])

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        smem_A.populate_ptr(storage)
        smem_B.populate_ptr(storage)


    def _populate_smem(self, shared_storage_objects):
        fields = {obj.name: obj.storage_type for obj in shared_storage_objects}
        cls = type("SharedStorage", (), dict())
        cls.__annotations__ = fields # Store stuff in annotations
        self.shared_storage = cute.struct(cls)
import cuda.bindings.driver as cuda

from cutlass import cute

from typing import Tuple

from layout import *
from cutlass.cute.runtime import from_dlpack
import torch


class Test:
    def __init__(self):
        self.ab_stage = 2
        self.tile_shape_mnk = (128, 256, 64)
        self.dtype = cutlass.BFloat16
        self.shared_storage = None
    
    def __call__(self, a: cute.Tensor, b: cute.Tensor, stream: cuda.CUstream):
        # just launch 1 warpgroup that does these TMA loads and yeah
        self.kernel(a, b).launch(grid=1, block=128, stream=stream)
    
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

def convert_from_dlpack(tensor, mode=0, stride_order=(0, 1)):
    return from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
        mode=mode, stride_order=stride_order
    )


if __name__ == '__main__':
    lst = [i for i in range(128*256)]
    a = torch.tensor(lst, dtype=torch.bfloat16).reshape((128, 256)).to('cuda')
    b = torch.tensor(lst, dtype=torch.bfloat16).reshape((128, 256)).to('cuda')
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    a_cute = convert_from_dlpack(a)
    b_cute = convert_from_dlpack(b)
    tk = Test()
    tk(a, b, current_stream)
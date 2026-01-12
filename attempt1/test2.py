import cuda.bindings.driver as cuda

from cutlass import cute

from typing import Tuple

from layout import *
from cutlass.cute.runtime import from_dlpack
import torch


class Test:
    def __init__(self):
        self.ab_stage = 2
        self.tile_shape_mnk = (64, 128, 64)
        self.dtype = cutlass.BFloat16
        self.shared_storage = None
    
    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, stream: cuda.CUstream):
        # just launch 1 warpgroup that does these TMA loads and yeah
        self.kernel(a, b).launch(grid=1, block=128, stream=stream)
    
    def create_initial_state(self, a: cute.Tensor, b: cute.Tensor):
        state = {
            'gmem_A': GMEMTensor(a),
            'gmem_B': GMEMTensor(b),
            'smem_A': SwizzledSMEM('sA', GemmPart.A, LayoutEnum.from_tensor(a), self.tile_shape_mnk, self.dtype, 2),
            'smem_B': SwizzledSMEM('sB', GemmPart.B, LayoutEnum.from_tensor(b), self.tile_shape_mnk, self.dtype, 2)
        }
        return state
    
    @cute.kernel
    def kernel(self, a: cute.Tensor, b: cute.Tensor):
        state = self.create_initial_state(a, b)

        self._populate_smem([state['smem_A'], state['smem_B']])

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        state['smem_A'].populate_ptr(storage)
        state['smem_B'].populate_ptr(storage)

        print(state['smem_A'].tensor)
        print(state['smem_B'].tensor)

        cta_coord_mn = (0, 0)
        a_cpy = TMACopyG2S(state['gmem_A'], state['smem_A'], cta_coord_mn)
        b_cpy = TMACopyG2S(state['gmem_B'], state['smem_B'], cta_coord_mn)
        print(a_cpy.tma_tensor)
        print(b_cpy.tma_tensor)


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
    tk(a_cute, b_cute, current_stream)
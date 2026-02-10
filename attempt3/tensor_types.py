from dataclasses import dataclass
from enum import Enum

class MajorModes(Enum):
    ROW=0
    COL=1


# TODO may have to add dtype
@dataclass
class GlobalMatrix:
    dims: tuple
    dim_order: tuple

    def __post_init__(self):
        assert len(self.dims) == len(self.dim_order), 'dims must match dim order'
        assert 0 in self.dim_order and 1 in self.dim_order, 'dim_order must have 0 and 1'
        self.dim0_before_1 = self.dim_order.index(0) < self.dim_order.index(1)

    # TODO check these
    def major_mode_a(self):
        return MajorModes.COL if self.dim0_before_1 else MajorModes.ROW
    
    def major_mode_b(self):
        return MajorModes.ROW if self.dim0_before_1 else MajorModes.COL
    
    def major_mode_c(self):
        return MajorModes.COL if self.dim0_before_1 else MajorModes.ROW
    
    def __repr__(self):
        return f'GlobalMatrix({self.dims}, {self.dim_order})'


@dataclass
class SharedMatrix:
    nrows: int
    ncols: int
    stages: int

    def __repr__(self):
        return f'SharedMatrix({self.nrows}, {self.ncols}, {self.stages})'


class RegisterWGMMA:
    ...

class RegisterRowSum:
    ...
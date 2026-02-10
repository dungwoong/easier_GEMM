from dataclasses import dataclass, field
from enum import Enum
from typing import List
import textwrap

class VarState(Enum):
    CLASSARG=0 # self.x = x
    CLASSCONST=1 # self.x = const
    CLASSVAR=2 # self.x = None (set in __call__)
    CALLARG=3 # passed into call
    NONE=4

@dataclass
class Decl:
    label: str=None
    varstate: VarState=VarState.NONE
    stmt: str=None
    reads: set = field(default_factory=list)
    const_val: str = None

    def __post_init__(self):
        self.reads = set(r.name if isinstance(r, Decl) else r for r in self.reads)
        assert not (self.varstate==VarState.CALLARG and self.reads), 'Call args must have no dependencies'
        self.stmt = self.stmt.replace('$NAME', self.name) if self.stmt is not None else None

    @property
    def name(self):
        return (
            f'self.{self.label}' if self.varstate in 
            (VarState.CLASSARG, VarState.CLASSCONST, VarState.CLASSVAR) 
            else self.label)
    
    @property
    def writes(self):
        return (self.name,)

    @property
    def init_arg(self):
        return self.label if self.varstate == VarState.CLASSARG else None

@dataclass
class TupleDecl(Decl):
    _writes: List[str] = field(default_factory=list)
    label: str=None
    varstate: VarState=VarState.NONE
    stmt: str=None
    reads: set = field(default_factory=list)
    const_val: str = None

    @property
    def name(self):
        return ', '.join(
            f'self.{l}' if self.varstate in 
            (VarState.CLASSARG, VarState.CLASSCONST, VarState.CLASSVAR) 
            else l for l in self._writes)
    
    @property
    def writes(self):
        return (
            f'self.{l}' if self.varstate in 
            (VarState.CLASSARG, VarState.CLASSCONST, VarState.CLASSVAR) 
            else l for l in self._writes)
    
    def getname(self, idx):
        return self.writes[idx]
    
    @property
    def init_arg(self):
        return ', '.join(self.writes) if self.varstate == VarState.CLASSARG else None

# we have a list of declarations
# to generate the init declaration, set args to classargs, add classconst and classvars
# If you have anything kernel scope, anything that reads that must be kernel scope too
# and then anything out of kernel scope that's call scope but read in kernel should be passed as args

class Declarations:
    def __init__(self, decls: List[Decl]):
        self.decls = decls

        # symbol table stores variables that are initialized
        # you can only read these variables I guess(...hopefully)
        self.symbols = []
    
    # these return string because there's only one way to generate, no shuffling required
    def init_args_str(self):
        return ', '.join([d.init_arg for d in self.decls if d.init_arg is not None])

    def call_args_str(self):
        return ', '.join([d.name for d in self.decls if d.varstate == VarState.CALLARG])

    def init_assignments_str(self):
        output = []
        for d in self.decls:
            if d.varstate not in (VarState.CLASSARG, VarState.CLASSVAR, VarState.CLASSCONST):
                continue
            if d.varstate == VarState.CLASSARG:
                output.append(f'{d.name} = {d.init_arg}')
            elif d.varstate == VarState.CLASSVAR:
                output.append(f'{d.name} = None')
            else: # classconst
                output.append(f'{d.name} = {d.const_val}')
        return '\n'.join(output)
    
    def _push_class_symbols(self):
        # all symbols that are ready by the end of the __init__
        output = set()
        for d in self.decls:
            if d.varstate in (VarState.CLASSARG, VarState.CLASSCONST):
                output.update(d.writes)
        self.symbols.append(output)
    
    def _push_call_args(self):
        assert len(self.symbols) == 1, "Expect only __init__ symbols at this point"
        output = set()
        for d in self.decls:
            if d.varstate == VarState.CALLARG:
                output.update(d.writes)
        self.symbols.append(output)
    
    def _var_exists(self, vname: str):
        for i in range(len(self.symbols)):
            sym_tbl = self.symbols[-(i+1)]
            if vname in sym_tbl:
                return True
        return False
    
    def check_call_assignments(self):
        for d in self.decls:
            if d.varstate in (VarState.CLASSARG, VarState.CALLARG):
                continue
            for v in d.reads:
                assert self._var_exists(v), f'Var {v} undefined when trying to write {d.name}'
            self.symbols[-1].update(d.writes)

    def call_string(self):
        output = []
        for d in self.decls:
            if d.varstate in (VarState.CLASSARG, VarState.CALLARG, VarState.CLASSCONST):
                continue
            assert d.stmt is not None, f"Error: {d.name} setup has no statement"
            output.append(d.stmt)
        print(output)
        return '\n'.join(output)
    
    # TODO kernel should pass in stuff that's required ONLY, otherwise you might pass in invalid args
    def mark_kernel_variables(self):
        # All variables reading from a kernel level variable belong in the kernel code
        # Call variables read from the kernel should be passed into the kernel call
        # If a var is marked otherwise, raise an error
        pass

def get_global_tensor(name):
    return Decl(name, VarState.CALLARG)

def get_layout(global_tensor: Decl):
    # this only works on 2d tensors
    name = f'{global_tensor.name}_layout'
    return Decl(name, VarState.CLASSVAR, 
                f'$NAME = utils.LayoutEnum.from_tensor({global_tensor.name})', 
                reads=[global_tensor])

# we can add like a %NAME or smth idc
def get_dtype(global_tensor: Decl, name='dtype'):
    decl = Decl(name, VarState.CLASSVAR,
                reads=[global_tensor])
    decl.stmt = f'{decl.name} = {global_tensor.name}.element_type'
    return decl


# decls need setup capabilities elsewhere to generate functions
def get_mma_atom(name, dtype: Decl, layout_a: Decl, layout_b: Decl, acc_dtype: Decl, tile_mn: Decl):
    decl = Decl(name, VarState.CLASSVAR, reads=[dtype, layout_a, layout_b, acc_dtype, tile_mn])
    decl.stmt = f'{decl.name} = get_mma_atom({dtype.name}, {layout_a.name}, {layout_b.name}, {acc_dtype.name}, {tile_mn.name})'
    return decl

def get_smem_layout(name, dtype: Decl, layout: Decl, stages: Decl):
    return Decl(name, VarState.CLASSVAR, reads=[dtype, layout, stages], stmt=f'$NAME = get_smem_layout({dtype.name}, {layout.name}, {stages.name})')

def get_tma_tensor_atom(tag, tensor, layout, tile, tx):
    return TupleDecl(_writes=(f'tma_atom_{tag}', f'tma_tensor_{tag}'), 
                     reads=(tensor, layout, tile, tx),
                     stmt=f'$NAME = self._tma_tns_incr_tx({tensor.name}, {layout.name}, {tile.name}) # implicit read {tx.name}')

ab_stage = Decl('ab_stage', VarState.CLASSARG)
tile_mn = Decl('tile_mnk', VarState.CLASSARG)
acc_dtype = Decl('acc_dtype', VarState.CLASSCONST, const_val='cutlass.Float32')
a = get_global_tensor('a')
b = get_global_tensor('b')
dtype = get_dtype(a)
a_layout = get_layout(a)
b_layout = get_layout(b)
mma_atom = get_mma_atom('tiled_mma', dtype, a_layout, b_layout, acc_dtype, tile_mn)

# AB share the same stage, but hard to figure out this relationship when doing codegen. We could just hardcode it anyways
# TMA load bytes might also have to be shared if we combine barriers...
# we can have some algorithm to combine these though
smem_a = get_smem_layout('smem_layout_a', dtype, a_layout, ab_stage)
smem_b = get_smem_layout('smem_layout_b', dtype, a_layout, ab_stage)

# If we combine loads + pipelines, that setup can be done at a higher level
tma_bytes = Decl('tma_bytes', VarState.CLASSVAR, '$NAME = 0')
tma_a = get_tma_tensor_atom('a', a, a_layout, tile_mn, tma_bytes)
tma_b = get_tma_tensor_atom('b', b, b_layout, tile_mn, tma_bytes)

decls = [ab_stage, tile_mn, acc_dtype, dtype, a, b, a_layout, b_layout, mma_atom, smem_a, smem_b, tma_bytes, tma_a, tma_b]

d = Declarations(decls)
init_args = d.init_args_str()
init_assignments = d.init_assignments_str()
call_args = d.call_args_str()
d._push_class_symbols()
d._push_call_args()
d.check_call_assignments()
call_setup = d.call_string()
output = f"""
class Kernel:
    def __init__(self, {init_args}):
{textwrap.indent(init_assignments, '        ')}

    def __call__(self, {call_args}):
{textwrap.indent(call_setup, '        ')}
        self.populate_shared_storage() # do this at the end

"""
print(output)
print(d.symbols)
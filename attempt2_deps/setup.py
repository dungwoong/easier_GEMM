from dataclasses import dataclass, field
from enum import Enum
from typing import List

class Scopes(Enum):
    CALL=0
    KERNEL=1
    NONE=None

@dataclass
class VarDecl:
    name: str # this is the only thing that should be written
    reads: set = field(default_factory=set)
    is_classvar: bool=False
    is_classarg: bool=False # if classarg, must also be classvar(e.g. self.x = x)
    is_callarg: bool=False
    const_val: any=None
    scope_hint: Scopes=Scopes.NONE
    stmt: str=None

    def __post_init__(self):
        for item in self.reads:
            if isinstance(item, VarDecl):
                self.reads.remove(item)
                self.reads.add(item.name)
        self.name = f'self.{self.name}' if self.is_classvar else self.name
        self.stmt = self.stmt if self.stmt is not None else f"# WARNING: NO STMT({self.name})"
    
    def __hash__(self):
        return hash(self.name) # TODO
    
    def __repr__(self):
        return f'{self.name}, {self.stmt}'

def get_global_tensor(name): # passing in mA, mB etc.
    return VarDecl(name, scope_hint=Scopes.CALL, is_callarg=True)

def get_dtype(global_tensor: VarDecl):
    return VarDecl(f'dtype', set([global_tensor]), is_classvar=True, stmt=f'self.dtype = {global_tensor.name}.element_type')

def get_layout(global_tensor: VarDecl):
    name = f'{global_tensor.name}_layout'
    return VarDecl(name, set([global_tensor]), is_classvar=True, stmt=f'self.{name} = utils.LayoutEnum.from_tensor({global_tensor.name})')

def get_mma_atom(name, dtype: VarDecl, a_layout: VarDecl, b_layout: VarDecl, acc_dtype: VarDecl, mn: VarDecl):
    return VarDecl(name, set([dtype, a_layout, b_layout, acc_dtype, mn]), 
                   stmt=f"""
{name} = sm90_utils.make_trivial_tiled_mma(
    {dtype.name},
    {dtype.name},
    {a_layout.name}.sm90_mma_major_mode(),
    {b_layout.name}.sm90_mma_major_mode(),
    {acc_dtype.name},
    atom_layout_mnk = ({mn.name}[0] // 64, 1, 1),
    tiler_mn = (64, {mn.name}[1])
)
""".strip())


# Processing
def get_classargs(decls: List[VarDecl]):
    output = []
    for d in decls:
        if d.is_classarg:
            assert d.is_classvar
            output.append(d)
    return output

def get_callargs(decls: List[VarDecl]):
    return [d for d in decls if d.is_callarg]

def get_classvars(decls: List[VarDecl]):
    return [d for d in decls if d.is_classvar]

# Generate argslist
def genargs(decls: List[VarDecl]):
    return ', '.join(c.name for c in decls)

def gen_classvars_assn(classvars: List[VarDecl]):
    output = ''
    for c in classvars:
        if c.is_classarg:
            assert c.const_val is None, 'cant have classarg with const val'
            output += f'{c.name} = {c.name.replace("self.", "")}\n'
        elif c.const_val is not None:
            output += f'{c.name} = {c.const_val}\n'
        else:
            output += f'{c.name} = None\n'
    return output.strip()


ab_stage = VarDecl('ab_stage', is_classvar=True, is_classarg=True)
tile_mn = VarDecl('tile_mn', is_classvar=True, is_classarg=True)
acc_dtype = VarDecl('acc_dtype', is_classvar=True, const_val='cutlass.Float32')
ga = get_global_tensor('a')
gb = get_global_tensor('b')
gc = get_global_tensor('c')
dtype = get_dtype(ga)
layout_a = get_layout(ga)
layout_b = get_layout(gb)
layout_c = get_layout(gc)
mma_atom = get_mma_atom('mma_atom', dtype, layout_a, layout_b, acc_dtype, tile_mn)

decls = [ab_stage, tile_mn, dtype, acc_dtype, ga, gb, gc, layout_a, layout_b, layout_c, mma_atom]

classargs = get_classargs(decls)
classvars = get_classvars(decls)
callargs = get_callargs(decls)

print('classargs:')
print(genargs(classargs))

print('classvars:')
print(gen_classvars_assn(classvars))

print('callargs:')
print(genargs(callargs))

def is_var_exists(var, scope_stack):
    for i in range(len(scope_stack)):
        sym_tbl = scope_stack[-(i+1)]
        if var in sym_tbl:
            return True
    return False

def process_call_decl(d: VarDecl, scope_stack):
    for v in d.reads:
        assert is_var_exists(v, scope_stack), v
    
    # print(f'adding {d.name}')
    scope_stack[-1].add(d.name)
    

class_symbols = set(a.name for a in decls if (a.is_classarg or a.const_val is not None))
call_symbols = set(a.name for a in callargs)
scope_stack = [class_symbols, call_symbols]
remaining = [d for d in decls if (d not in classargs and d not in callargs and d.const_val is None)] # classvars that aren't args get assigned
call_decls = ''
for decl in remaining:
    # print(f'processing {decl.stmt}')
    process_call_decl(decl, scope_stack)
    call_decls += decl.stmt + '\n'

print('calldecls:')
print(call_decls)

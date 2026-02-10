from enum import Enum
from typing import Optional
from tensor_types import GlobalMatrix, SharedMatrix
from dataclasses import dataclass, field
from enum import Enum
import textwrap

class NodeNamespace(Enum):
    ANY=0
    LOW=1
    HIGH=2


# Type system: specify a name, metadata
@dataclass
class Type:
    name: str
    meta: dict = field(default_factory=dict)

@dataclass
class Var:
    ir_name: str
    # add e.g. is it a classvar, etc.
    classvar: bool=False
    
    def __repr__(self):
        return ('self.' if self.classvar else '') + self.ir_name
    
    def __hash__(self):
        return hash((self.ir_name, self.classvar))


# global vars contains k etc. it should be like a symbol table
# assume we have these nodes sorted in one k loop. We wanna generate all this in order
# and then capture dependencies after
# symbol table can contain AST Identifier Nodes so I can use them while generating tbh

# we can annotate scopes for each node so we can make sure lookups are fine
class NodeBase:
    namespace = NodeNamespace.HIGH # this won't work because e.g. scopes/loops are applicable anywhere
    # we can model warp specialization as scopes
    def __init__(self):
        self.reads = set()
        self.writes = set()

        self.children = []
    
    def add_child(self, c):
        assert isinstance(c, NodeBase)
        self.children.append(c)
    
    def add_children(self, *children):
        for c in children:
            assert isinstance(c, NodeBase)
        self.children.extend(children)
    
    # setup and generate are codegen functions, to go to even lower IR
    def setup(self):
        return []
    
    def generate(self):
        return []
    
    def accept(self, visitor):
        visit_fn = getattr(visitor, f'visit_{self.__class__.__name__}', None)
        if callable(visit_fn):
            visit_fn(self)


# References: just check if things are in scope
# Declarations: you can actually initialize stuff
# ----- these will be outside of the loop hopefully so 
# Ops/functions: they specify what they read/write

# we need to decide on variable names ahead of time

class ClassArgDecl(NodeBase):
    def __init__(self, var: Var):
        super().__init__()
        var.classvar = True
        self.var = var
    
    @property
    def arg(self):
        return repr(self.var).replace("self.", "")
    
    def __repr__(self):
        return f'{self.var} = {self.arg}'

class Reference(NodeBase): ...

# you have references and decls
# references check var exists, decl checks var doesn't exist
class VarReference(Reference):
    def __init__(self, var: Var):
        super().__init__()
        self.var = var
    
    def __repr__(self):
        return repr(self.var)


class ConstantReference(Reference):
    def __init__(self, val):
        super().__init__()
        self.val = val
    
    def __repr__(self):
        return str(self.val)

class KernelClass(NodeBase):
    def __init__(self):
        super().__init__()

    @property
    def call(self):
        return self.children[-1]
    
    @property
    def args(self):
        return [a for a in self.children if isinstance(a, ClassArgDecl)]

    def __repr__(self):
        args = ', '.join(a.arg for a in self.args)
        children = '\n'.join(repr(c) for c in self.children[:-1])
        return f"""
class Kernel:
  def __init__(self, {args})
{textwrap.indent(children, "    ")}

{textwrap.indent(repr(self.call), "  ")}
""".strip()
    
class Scope(NodeBase):
    def __init__(self, nthreads):
        super().__init__()
        self.nthreads = nthreads
    
    def __repr__(self):
        children = '\n'.join(repr(c) for c in self.children)
        return f"""if (create_group({self.nthreads})):
{textwrap.indent(children, "  ")}
""".strip()

class CallFunction(NodeBase):

    @property
    def kernel_defn(self):
        return self.children[-1]
    
    @property
    def args(self):
        return [a for a in self.children if isinstance(a, GlobalArgDecl)]
    
    def __repr__(self):
        args = ', '.join(repr(a.var) for a in self.args) + (', ' if self.args else '')
        children = '\n'.join(repr(c) for c in self.children[:-1])
        children += f"\nself.kernel('TODO').launch(grid=TODO, block=[{self.kernel_defn.nthreads}], cluster=TODO, stream=stream)"
        return f"""@cute.jit
def call({args}stream: cuda.CUstream):
{textwrap.indent(children, "  ")}

{repr(self.kernel_defn)}
""".strip()

class KernelFunction(NodeBase):
    def __init__(self):
        super().__init__()

    @property
    def nthreads(self):
        return self.children[0]
    
    # last item must be a loop
    @property
    def mainloop(self):
        assert len(self.children) >= 1
        return self.children[-1]
    
    def __repr__(self):
        children = '\n'.join(repr(c) for c in self.children[1:])
        return f"""@cute.kernel
def kernel(TODO): # {self.nthreads} threads
{textwrap.indent(children, "  ")}
""".strip()

class Declaration(NodeBase):
    def __init__(self):
        super().__init__()
        self.type = None

class VarDecl(Declaration):
    def __init__(self, var: Var, t: Type=None):
        super().__init__()
        self.var = var
        self.type = t
    
    def __repr__(self):
        return repr(self.var)


class ConstantDecl(Declaration):
    def __init__(self, var: Var, value):
        super().__init__()
        self.value = value
        self.type = type(value)
        self.var = var
    
    def __repr__(self):
        return f'{self.var} = {self.value}'

# loops can have statements in the loop conditions and then dataflow ops
# hopefully that's okay
# how will we model producer_tail...? maybe just as a pass?
class Loop(NodeBase):
    def __init__(self, loopvar_name: str):
        super().__init__()
        loopvar = VarDecl(Var(loopvar_name))
        self.add_child(loopvar)
        # after are statements(?)
        # we could track #init and #advance stmts and treat the loop that way if needed
    
    @property
    def var(self):
        return self.children[0]
    
    def __repr__(self):
        # we need to put loop incr stuff at the end of the loop
        children = '\n'.join(repr(c) for c in self.children[1:])
        return f"""for {repr(self.var)} in cutlass.range('TODO', unroll=1):
{textwrap.indent(children, "  ")}
""".strip()

class GlobalArgDecl(Declaration): # collect global arg param
    def __init__(self, t, var: VarDecl): # idk what to do about this for now...
        super().__init__()
        self.dtype = t # should this be a child...
        self.type = Type('GlobalTensor') # so we assume that only global tensors are allowed as global args...
        self.add_child(var)
    
    @property
    def var(self):
        return self.children[0]
    
    def __repr__(self):
        # return f'{self.var} = args.collect({self.dtype})'
        return f'# {self.var} is {self.dtype}' # we could add a statement to permute these things or smth

# could this setup the shared layout?
# this makes stuff and gets us to the point of defining sA or whatever
# TODO make sure this has sufficient info to give us the shared matrix
# setup will be get the layout, gen will be take ptr from smem
class SharedDecl(NodeBase):
    def __init__(self, r, c, stages, var: VarDecl):
        super().__init__()
        var.type = Type('SHMATRIX', meta={'r': r, 'c': c, 'stages': stages})
        self.add_child(var)
        self.add_child(r)
        self.add_child(c)
        self.add_child(stages)
    
    @property
    def var(self):
        return self.children[0]
    
    def __repr__(self):
        return f'{self.var} = SharedMatrix{self.children[1], self.children[2], self.children[3]}'

class GemmAccDecl(NodeBase):
    def __init__(self, gemm, r, c, var: VarDecl):
        super().__init__()
        var.type = Type('WGMMA_ACC', meta={'r': r, 'c': c})
        self.add_children(var, gemm, r, c)

    def __repr__(self):
        return f'{self.children[0]} = GemmAcc{tuple(self.children[1:])}'

class PipelineDecl(NodeBase):
    def __init__(self, decl_var, stages: Reference):
        super().__init__()
        decl_var.type = Type('PIPELINE', meta={'stages': stages})
        self.add_children(decl_var, stages)
    
    @property
    def var(self):
        return self.children[0]
    
    @property
    def stages(self):
        return self.children[1]

    def __repr__(self):
        return f'{self.var} = Pipeline({self.stages})'

# need
# - is row iter or col iter
# - tile size
# - row/col index
# tma_load(g, s, tilesize, coord_row, coord_col, shared_idx)
# TODO can we get away with no Exprs in this grammar?
class TMALoadG2S(NodeBase):
    def __init__(self, global_tensor: Reference, shared_tensor: Reference, 
                 tile_size_m: Reference, tile_size_n: Reference, 
                 coord_row: Optional[Reference], coord_col: Optional[Reference], loop_var: Optional[Reference],
                 shared_idx: Reference):
        super().__init__()
        self.add_child(global_tensor)
        self.add_child(shared_tensor)
        self.add_child(tile_size_m)
        self.add_child(tile_size_n)
        self.add_child(loop_var if coord_row is None else coord_row)
        self.add_child(loop_var if coord_col is None else coord_col)
        self.add_child(shared_idx)
    
    @property
    def global_tensor(self):
        return self.children[0]
    
    @property
    def shared_tensor(self):
        return self.children[1]
    
    @property
    def tile_size(self):
        return (self.children[2], self.children[3])
    
    @property
    def coords(self):
        return (self.children[4], self.children[5])

    @property
    def shared_idx(self):
        return self.children[6]
    
    # def setup(self):
    #     # need to partition etc.
    #     # cute.make smem layout atom
    #     # cute.tile to shape
    #     # makes sure all the reads/writes are initialized
    #     # maybe we could have an initializes attr for setup stuff
    #     pass

    # def generate(self):
    #     arr = ProdArrive()
    #     ld = TMALoad()
    #     wait = ConsWait()
    #     return [arr, ld, wait]
    
    def __repr__(self):
        return f"{self.shared_tensor}[{self.shared_idx}] = TMAload({self.global_tensor}, {self.tile_size}, {self.coords})"

class Gemm(NodeBase):
    def __init__(self, gemm_tiled,
                 a_tile, b_tile, c_tile, a_idx, b_idx):
        super().__init__()
        # TODO we need to allow array accesses
        self.add_children(gemm_tiled, a_tile, b_tile, c_tile, a_idx, b_idx)

    def __repr__(self):
        return f"Gemm({self.children[0]}).run{tuple(self.children[1:])}"

class GemmDecl(NodeBase):
    def __init__(self, gemm_tiled: Declaration, dtype, a_major_mode, b_major_mode, acc_dtype, tile_m, tile_n, tile_k):
        super().__init__()
        gemm_tiled.type = Type('TILED_GEMM', meta=dict()) # TODO we could have gemm decide number of threads
        self.add_children(gemm_tiled, dtype, a_major_mode, b_major_mode, acc_dtype, tile_m, tile_n, tile_k)
    
    def __repr__(self):
        return f"{self.children[0]} = TiledGemm{tuple(self.children[1:])}"

# we should also write like an nwarps calculation somehow so we can use it when warp specializing(?)
class CalculateTotalThreadsDecl(NodeBase): # calculate number of threads, we need to somehow do this based on mma sizes etc
    # we can probably throw this on before doing analysis you just add a pass to move MMA type decls into __call__ and add CalculateThreadsDecl
    def __init__(self, vardecl: Declaration, warp_specialize: Reference, *args):
        super().__init__()
        self.add_child(vardecl)
        self.add_child(warp_specialize)
        self.add_children(*args)
    
    def __repr__(self):
        args = ', '.join(repr(c) for c in self.children[2:]) + ', ' if len(self.children) > 2 else ''
        return f"{self.children[0]} = NThreads({args}warp_specialize={self.children[1]})"


def postorder_visit(visitor, root: KernelClass):
    stack = [(root, 0)] # node, next_child_idx

    while stack:
        node, i = stack.pop()
        if i == 0: # first time seeing this node
            visitor.enter(node)
        if i < len(node.children):
            stack.append((node, i+1))
            stack.append((node.children[i], 0))
        else:
            visitor.visit(node)
            visitor.exit(node)
    print(f'{visitor.label} completed')


class DeclCheckVisitor:
    label = 'Decl Check'
    def __init__(self):
        # these might have to be dicts later
        self.symbols = [] # class, call, kernel, loop
        self.current_stack_idx = -1
        self.kernel_fn = None
        self.kernel_cls = None
    
    def visit(self, node):
        node.accept(self)
    
    def get_var(self, var: Var):
        result = None
        for i in range(self.current_stack_idx+1):
            bwd_idx = self.current_stack_idx - i
            tbl = self.symbols[bwd_idx]
            if var in tbl and not (not var.classvar and bwd_idx == 0):
                if result is not None:
                    print(f'Warning: {var} exists in multiple scopes')
                result = bwd_idx
        return result
    
    def enter(self, node):
        if isinstance(node, (KernelClass, CallFunction, KernelFunction, Scope, Loop)):
            self.current_stack_idx += 1
            new_symbols = set()
            self.symbols.append(new_symbols)
            node._symbols = new_symbols
        if isinstance(node, KernelClass):
            self.kernel_cls = node
    
    def exit(self, node):
        if isinstance(node, (KernelClass, CallFunction, KernelFunction, Scope, Loop)):
            self.current_stack_idx -= 1
            self.symbols.pop()

    def _visit_Decl(self, decl):
        if self.get_var(decl.var) is not None: 
            print(f'[ERROR] {decl.var} was declared twice')
        self.symbols[self.current_stack_idx].add(decl.var)
    
    def visit_VarReference(self, ref: VarReference):
        if self.get_var(ref.var) is None:
            print(f'[ERROR] {ref.var} was referenced but not found')
    
    def visit_VarDecl(self, decl):
        self._visit_Decl(decl)
    
    def visit_GlobalArgDecl(self, decl: GlobalArgDecl):
        self._visit_Decl(decl)
    
    def visit_ClassArgDecl(self, decl: ClassArgDecl):
        if self.current_stack_idx != 0:
            print(f'[ERROR] class __init__ arg {decl.var} not in classdef')
        self._visit_Decl(decl)
    
    def visit_ConstantDecl(self, decl: ConstantDecl):
        self._visit_Decl(decl)
    

def var_ref(name):
    return VarReference(Var(name))

def var_ref2(var):
    return VarReference(var)


tile_m = Var('tile_m', classvar=True)
tile_n = Var('tile_n', classvar=True)
tile_k = Var('tile_k', classvar=True)
stages = Var('stage', classvar=True)
dtype = Var('dtype', True)
acc_dtype = Var('acc_dtype', True)
a_g = Var('a')
b_g = Var('b')
pipe = Var('pipe_ab')
acc = Var('acc')
nthreads = Var('nthreads')
kernel = KernelFunction()
kernel.add_child(VarReference(nthreads))
kernel.add_child(SharedDecl(var_ref2(tile_m), var_ref2(tile_k), var_ref2(stages), VarDecl(Var('As')))) # but for SMEM we should store the dims ahead of time
kernel.add_child(SharedDecl(var_ref2(tile_n), var_ref2(tile_k), var_ref2(stages), VarDecl(Var('Bs'))))
kernel.add_child(GemmAccDecl(var_ref('tiled_mma'), VarReference(tile_m), VarReference(tile_n), VarDecl(acc)))
loop = Loop('k')
loop.add_child(TMALoadG2S(var_ref('a'), var_ref('As'), var_ref2(tile_m), var_ref2(tile_k), ConstantReference('bidx'), None, var_ref('k'), var_ref2(pipe))) # need to add pipelinestate
loop.add_child(TMALoadG2S(var_ref('b'), var_ref('Bs'), var_ref2(tile_n), var_ref2(tile_k), ConstantReference('bidy'), None, var_ref('k'), var_ref2(pipe)))
loop.add_child(Gemm(var_ref('tiled_mma'), var_ref('As'), var_ref('Bs'), var_ref('acc'), var_ref2(pipe), var_ref2(pipe)))

scope = Scope(256)
scope.add_child(PipelineDecl(VarDecl(pipe), VarReference(stages)))
scope.add_child(loop)
kernel.add_child(scope)
# print(kernel)

# TODO need to enforce runtime vars can't be declared in __call__ e.g. pipeline, etc.
callfunction = CallFunction()
callfunction.add_child(GemmDecl(VarDecl(Var('tiled_mma')), VarReference(dtype), 
                   ConstantReference('ROW'), ConstantReference('COL'), # higher-level nodes decide this
                   VarReference(acc_dtype), VarReference(tile_m), VarReference(tile_n), VarReference(tile_k)))
callfunction.add_child(GlobalArgDecl(GlobalMatrix((64, 128), (0, 1)), VarDecl(a_g))) # we can store this data here since we'd only know this at runtime
callfunction.add_child(GlobalArgDecl(GlobalMatrix((128, 128), (0, 1)), VarDecl(b_g)))
callfunction.add_child(CalculateTotalThreadsDecl(VarDecl(nthreads, int), ConstantReference(False), var_ref('tiled_mma')))
callfunction.add_child(kernel)

kc = KernelClass()
kc.add_child(ClassArgDecl(tile_m))
kc.add_child(ClassArgDecl(tile_n))
kc.add_child(ConstantDecl(tile_k, 64))
kc.add_child(ClassArgDecl(stages))
kc.add_child(ConstantDecl(dtype, 'cutlass.Bfloat16'))
kc.add_child(ConstantDecl(acc_dtype, 'cutlass.Float32'))
kc.add_child(callfunction)
# print(kc)

dv = DeclCheckVisitor()
postorder_visit(dv, kc)
print(kc)

# CHECKS
# non-class args declared in class should be removed when entering scope btw
# decls: make sure var doesn't exist and add to symbol table
# refs: make sure var exists
# pipeline decls: should not be in the mainloop
# we can check shapes and types but since this will come from higher level representation we could leave it for now

# how will we figure out producer consumer-scoped loops? What if we want to support more loops?
# maybe each loops specifies nthreads and we can make sure thing will work there
# otherwise, you have to check deps when you're DOING the transformation I would guess, or like after you've done all the transforms?

# We can make sure MMAs use the same number of threads and stuff
# - have a division strategy e.g. (64, None), pass as metadata to kernel
# - for each mma, make sure the non-null dimensions are the same(e.g. tile_m in this case)
# - for total threads, we can just assume all mmas will be the same scope, and loads will be one WG

# after...
# - split things up
# - move things to better levels
# - break things into PC
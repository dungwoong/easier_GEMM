from enum import Enum
from typing import Optional, Union
from tensor_types import GlobalMatrix, SharedMatrix
from dataclasses import dataclass, field
from enum import Enum
import textwrap
import my_types as T
from util import flatten_once
from copy import deepcopy

class NodeNamespace(Enum):
    ANY=0
    LOW=1
    HIGH=2


# Type system: specify a name, metadata
# with this type system, we can't check for inherited types, maybe we could just make a bunch of placeholder classes...
@dataclass
class Type:
    type: type
    meta: tuple = field(default_factory=tuple) # just remember what each key is or we can put staticmethods in the type

    def __post_init__(self):
        assert isinstance(self.type, type), f'{self.type} is not type'

@dataclass
class Var:
    ir_name: str
    # add e.g. is it a classvar, etc.
    classvar: bool=False
    
    def __repr__(self):
        return ('self.' if self.classvar else '') + self.ir_name
    
    def __hash__(self):
        return hash((self.ir_name, self.classvar))
    
    def __eq__(self, value):
        if not isinstance(value, Var):
            return NotImplemented
        return self.ir_name == value.ir_name and self.classvar == value.classvar


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
        self.parent = None
    
    def set_parent(self, p):
        assert self.parent is None, f'{self} {type(self)} {self.parent}, {p}'
        self.parent = p
    
    def add_child(self, c):
        assert isinstance(c, NodeBase)
        self.children.append(c)
        c.set_parent(self)
    
    def add_children(self, *children):
        for c in children:
            assert isinstance(c, NodeBase)
            c.set_parent(self)
        self.children.extend(children)
    
    def replace_child(self, old_child, new_child):
        assert old_child in self.children
        idx = self.children.index(old_child)
        self.children[idx] = new_child
        old_child.parent = None
    
    # setup and generate are codegen functions, to go to even lower IR
    def setup(self):
        return []
    
    def generate(self):
        return []
    
    def accept(self, visitor):
        visit_fn = getattr(visitor, f'visit_{self.__class__.__name__}', None)
        if callable(visit_fn):
            return visit_fn(self)


# References: just check if things are in scope
# Declarations: you can actually initialize stuff
# ----- these will be outside of the loop hopefully so 
# Ops/functions: they specify what they read/write

# we need to decide on variable names ahead of time
# Declarations are just declaring the variable, other stuff should actually be renamed as "initialization" tbh
class Declaration(NodeBase):
    def __init__(self):
        super().__init__()
        self.var = None
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
        self.type = Type(type(value))
        self.var = var
    
    def __repr__(self):
        return f'{self.var} = {self.value}'

class ClassArgDecl(Declaration):
    def __init__(self, var: Var, type: Type):
        super().__init__()
        var.classvar = True
        self.var = var
        self.type = type
    
    @property
    def arg(self):
        return repr(self.var).replace("self.", "")
    
    def __repr__(self):
        return f'{self.var} = {self.arg}'

# you have references and decls
# references check var exists, decl checks var doesn't exist

class VarReference(NodeBase):
    def __init__(self, var: Var):
        super().__init__()
        self.var = var
    
    def __repr__(self):
        return repr(self.var)
    
    def __hash__(self):
        return hash(self.var)
    
    def __eq__(self, other):
        if not isinstance(other, VarReference):
            return NotImplemented
        return self.var == other.var


class ConstantReference(NodeBase):
    def __init__(self, val):
        super().__init__()
        self.val = val
    
    @property
    def var(self): # compatible with VarReference when needed :(
        return self.val
    
    def __repr__(self):
        return str(self.val)

Argument = Union[VarReference, ConstantReference]

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

class ThreadScope(NodeBase):
    def __init__(self):
        super().__init__()
    
    @property
    def nthreads(self):
        return self.children[0]
    
class Scope(ThreadScope):
    def __init__(self, nthreads: int):
        super().__init__()
        self.children.append(ConstantReference(nthreads))
    
    def __repr__(self):
        children = '\n'.join(repr(c) for c in self.children[1:])
        return f"""if (create_group({self.nthreads})):
{textwrap.indent(children, "  ")}
""".strip()

class CallFunction(NodeBase):
    def __init__(self):
        super().__init__()
        self.kernel_args = None # args to put into kernel

    @property
    def kernel_defn(self):
        return self.children[-1]
    
    @property
    def args(self):
        return [a for a in self.children if isinstance(a, GlobalTensorInit)]
    
    def __repr__(self):
        args = ', '.join(repr(a.var) for a in self.args) + (', ' if self.args else '')
        kernel_args = ', '.join(repr(a) for a in self.kernel_args)
        children = '\n'.join(repr(c) for c in self.children[:-1])
        children += f"\nself.kernel({kernel_args}).launch(grid=TODO, block=[{self.kernel_defn.nthreads}], stream=stream) # no cluster for now"
        return f"""@cute.jit
def call({args}stream: cuda.CUstream):
{textwrap.indent(children, "  ")}

{repr(self.kernel_defn)}
""".strip()

class KernelFunction(ThreadScope):
    def __init__(self):
        super().__init__()
        self.kernel_args = None

    @property
    def nthreads(self):
        return self.children[0]
    
    def children_to_ignore(self): # when generating kernel args
        return self.children[:1]
    
    def __repr__(self):
        children = '\n'.join(repr(c) for c in self.children[1:])
        args = ', '.join(repr(a) for a in self.kernel_args)
        return f"""@cute.kernel
def kernel({args}): # {self.nthreads} threads
{textwrap.indent(children, "  ")}
""".strip()

# loops can have statements in the loop conditions and then dataflow ops
# hopefully that's okay
# how will we model producer_tail...? maybe just as a pass?
class Loop(NodeBase):
    def __init__(self, loopvar_name: str, extent: ConstantReference):
        super().__init__()
        loopvar = VarDecl(Var(loopvar_name), Type(int))
        self.add_child(loopvar)
        self.add_child(extent)
        # after are statements(?)
        # we could track #init and #advance stmts and treat the loop that way if needed
    
    @property
    def var(self):
        return self.children[0]
    
    def __repr__(self):
        # we need to put loop incr stuff at the end of the loop
        children = '\n'.join(repr(c) for c in self.children[2:])
        return f"""for {repr(self.var)} in cutlass.range({self.children[1]}, unroll=1):
{textwrap.indent(children, "  ")}
""".strip()

# TODO these should actually do some work e.g. convert the global matrix into TMA atom/tensor
class GlobalTensorInit(NodeBase): # collect global arg param
    def __init__(self, t, var: VarDecl): # idk what to do about this for now...
        # TODO deal with t later, what is going on
        super().__init__()
        self.dtype = t # should this be a child...
        var.type = Type(T.GlobalTensor) # so we assume that only global tensors are allowed as global args...
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
class SharedTensorInit(NodeBase):
    def __init__(self, r, c, stages, var: VarDecl):
        super().__init__()
        var.type = Type(T.SharedTensor)
        self.add_child(var)
        self.add_child(r)
        self.add_child(c)
        self.add_child(stages)
    
    @property
    def var(self):
        return self.children[0]
    
    def __repr__(self):
        return f'{self.var} = SharedMatrix{self.children[1], self.children[2], self.children[3]}'

class GemmAccInit(NodeBase):
    def __init__(self, gemm, r, c, var: VarDecl):
        super().__init__()
        var.type = Type(T.WgmmaAcc)
        self.add_children(var, gemm, r, c)

    def __repr__(self):
        return f'{self.children[0]} = GemmAcc{tuple(self.children[1:])}'

# TODO add shared pointer somehow
class PipelineInit(NodeBase):
    def __init__(self, decl_var, stages: Argument):
        super().__init__()
        decl_var.type = Type(T.Pipeline)
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
    def __init__(self, global_tensor: Argument, shared_tensor: Argument, 
                 tile_size_m: Argument, tile_size_n: Argument, 
                 coord_row: Optional[Argument], coord_col: Optional[Argument], loop_var: Optional[Argument],
                 pipeline: Argument, pipe_state: Argument):
        super().__init__()
        self.add_child(global_tensor)
        self.add_child(shared_tensor)
        self.add_child(tile_size_m)
        self.add_child(tile_size_n)
        self.add_child(loop_var if coord_row is None else coord_row)
        self.add_child(loop_var if coord_col is None else coord_col)
        self.add_child(pipeline)
        self.add_child(pipe_state)
    
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
    def pipe(self):
        return self.children[6]
    
    @property
    def state(self):
        return self.children[7]
    
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
        return f"{self.shared_tensor}[{self.state}] = TMAload({self.global_tensor}, {self.tile_size}, {self.coords}, {self.pipe})"


# there could be some interesting checks here, we need to think about it but we can just add it to the visitor later
# I actually don't care, we can just make a class that handles this entire thing, once we point memory at it
# TODO maybe set whether to reuse memory here
class EpilogueStore(NodeBase):
    def __init__(self, accumulators: Argument, global_tensor: Argument): # TODO add output coords
        super().__init__()
        self.add_children(accumulators, global_tensor)
    
    def __repr__(self):
        return f'{self.children[1]} = epilogue_store({self.children[0]})'


class Gemm(NodeBase):
    def __init__(self, gemm_tiled,
                 a_tile, b_tile, c_tile, a_idx, b_idx):
        super().__init__()
        # TODO we need to allow array accesses
        self.add_children(gemm_tiled, a_tile, b_tile, c_tile, a_idx, b_idx)

    def __repr__(self):
        return f"Gemm({self.children[0]}).run{tuple(self.children[1:])}"

class TiledGemmInit(NodeBase):
    def __init__(self, gemm_tiled: Declaration, dtype, a_major_mode, b_major_mode, acc_dtype, tile_m, tile_n, tile_k):
        super().__init__()
        gemm_tiled.type = Type(T.TiledGemm) # TODO we could have gemm decide number of threads
        self.add_children(gemm_tiled, dtype, a_major_mode, b_major_mode, acc_dtype, tile_m, tile_n, tile_k)
    
    def __repr__(self):
        return f"{self.children[0]} = TiledGemm{tuple(self.children[1:])}"

class PipelineStateInit(NodeBase):
    def __init__(self, state: Declaration, stages: Argument):
        super().__init__()
        assert isinstance(stages, Argument)
        state.type = Type(T.PipelineState, (stages.var,)) # TODO not sure if this is how we should do it
        self.add_children(state, stages)
    
    def __repr__(self):
        return f"{self.children[0]} = PipelineState({self.children[1]})"


# we should also write like an nwarps calculation somehow so we can use it when warp specializing(?)
class CalculateTotalThreads(NodeBase): # calculate number of threads, we need to somehow do this based on mma sizes etc
    # we can probably throw this on before doing analysis you just add a pass to move MMA type decls into __call__ and add CalculateThreadsDecl
    def __init__(self, vardecl: Declaration, warp_specialize: Argument, *args):
        super().__init__()
        self.add_child(vardecl)
        self.add_child(warp_specialize)
        self.add_children(*args)
    
    def __repr__(self):
        args = ', '.join(repr(c) for c in self.children[2:]) + ', ' if len(self.children) > 2 else ''
        return f"{self.children[0]} = NThreads({args}warp_specialize={self.children[1]})"


def copy_arg(c: Argument):
    assert isinstance(c, Argument), "Only Arguments(References to variables or constants) can be copied"
    cpy = deepcopy(c)
    cpy.parent = None
    return cpy

# def CustomType(name, repr_fn, superclass=NodeBase):
#     assert issubclass(superclass, NodeBase)
#     assert callable(repr_fn)
#     return type(name, (superclass,), {'__repr__': repr_fn})

class CustomFuncNode(NodeBase):
    def __init__(self, stmt, children):
        super().__init__()
        self._stmt = stmt # lambda function that takes self as argument
        children = [self._copy_remove_parent(c) for c in children]
        self.add_children(*children)
    
    # if you wanna reuse a decl you have to go thru this(TODO we can refactor this to be better in a sec)
    def _copy_remove_parent(self, c: Argument):
        assert isinstance(c, Argument), "Only VarReferences and ConstantReferences can be copied"
        cpy = deepcopy(c)
        cpy.parent = None
        return cpy
    
    def stmt(self):
        return self._stmt(self)
    
    def __repr__(self):
        return self.stmt()


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


# decls happen once
# references are valid
# functions get called with the correct type
class DeclCheckVisitor:
    label = 'Decl Check'
    def __init__(self):
        # these might have to be dicts later
        self.symbols = [] # class, call, kernel, loop
        self.nthreads = []
        self.current_stack_idx = -1
        self.kernel_fn = None
        self.call_fn = None
        self.kernel_cls = None
        self.kernel_args = set() # any args referenced in kernel that were declared in __call__
    
    def visit(self, node):
        node.accept(self)
    
    def _current_scope_threads(self):
        return self.nthreads[self.current_stack_idx]
    
    def _get_var(self, var: Var):
        result = None
        t = None
        for i in range(self.current_stack_idx+1):
            bwd_idx = self.current_stack_idx - i
            tbl = self.symbols[bwd_idx]
            if var in tbl and not (not var.classvar and bwd_idx == 0 and self.current_stack_idx > 0):
                if result is not None:
                    print(f'Warning: {var} exists in multiple scopes')
                else:
                    result = bwd_idx
                    t = tbl[var]
        return result, t
    
    def get_var_scope(self, var: Var):
        return self._get_var(var)[0]
    
    def get_var_type(self, var: Var):
        return self._get_var(var)[1]
    
    def _check_var_type(self, var: Var, type: Type):
        return self.get_var_type(var) == type
    
    def _assert_or_print_error(self, val, output):
        if not val:
            print(f'[ERROR] {output}')
        return val

    def _compare_types(self, t1, t2):
        if isinstance(t1, Type) and isinstance(t2, Type):
            return issubclass(t1.type, t2.type) # we can later check meta too
        elif isinstance(t1, Type):
            return issubclass(t1.type, t2)
        elif isinstance(t2, Type):
            return issubclass(t1, t2.type)
        else:
            return issubclass(t1, t2)
    
    def _argument_is_type(self, arg: Argument, t: Type | type):
        # precondition: var exists in the symbol table with a type
        if isinstance(arg, ConstantReference):
            return self._compare_types(t, type(arg.val))
        elif isinstance(arg, VarReference):
            return self._compare_types(t, self.get_var_type(arg.var))
        else:
            raise NotImplementedError()
    
    def _decl_is_type(self, decl: VarDecl, t: Type | type):
        return self._compare_types(decl.type, t)
    
    def enter(self, node):
        if isinstance(node, (KernelClass, CallFunction, KernelFunction, Scope, Loop)):
            self.current_stack_idx += 1
            new_symbols = dict()
            self.symbols.append(new_symbols)
            node._symbols = new_symbols
            self.nthreads.append(node.nthreads.val if isinstance(node, ThreadScope) and isinstance(node.nthreads, ConstantReference) and isinstance(node.nthreads.val, int) else None)
        if isinstance(node, CallFunction):
            self.call_fn = node
        elif isinstance(node, KernelClass):
            self.kernel_cls = node
        elif isinstance(node, KernelFunction):
            self.kernel_fn = node
    
    def exit(self, node):
        if isinstance(node, (KernelClass, CallFunction, KernelFunction, Scope, Loop)):
            self.current_stack_idx -= 1
            self.symbols.pop()
        if isinstance(node, KernelFunction): # when exiting kernel fn, you should know all kernel args
            args = [a for a in list(self.kernel_args) if a not in self.kernel_fn.children_to_ignore()]
            self.kernel_fn.kernel_args = args
            self.call_fn.kernel_args = args

    def _visit_Decl(self, decl: Declaration):
        if self.get_var_scope(decl.var) is not None: 
            print(f'[ERROR] {decl.var} was declared twice')
        self.symbols[self.current_stack_idx][decl.var] = decl.type
    
    def visit_VarReference(self, ref: VarReference):
        var_lvl = self.get_var_scope(ref.var)
        if var_lvl is None:
            print(f'[ERROR] {ref.var} was referenced but not found')
        if self.current_stack_idx >= 2 and var_lvl == 1: # var is declared in __call__ but accessed in kernel
            self.kernel_args.add(ref)
    
    def visit_VarDecl(self, decl):
        self._visit_Decl(decl)
    
    def visit_GlobalArgDecl(self, decl: GlobalTensorInit):
        self._visit_Decl(decl)
    
    def visit_ClassArgDecl(self, decl: ClassArgDecl):
        if self.current_stack_idx != 0:
            print(f'[ERROR] class __init__ arg {decl.var} not in classdef')
        self._visit_Decl(decl)
    
    def visit_ConstantDecl(self, decl: ConstantDecl):
        self._visit_Decl(decl)
    
    def visit_SharedTensorInit(self, node: SharedTensorInit):
        self._assert_or_print_error(self.current_stack_idx == 2, 'SharedTensorInit can only happen in kernel')
        if not self._assert_or_print_error(len(node.children) == 4, 'SharedTensorInit must have 4 children'):
            return
        self._assert_or_print_error(all(isinstance(c, Argument) for c in node.children[1:]), f"Children of SharedTensorInit must be arguments")
        error = "SharedTensorInit r: int, c: int, stage: int"
        self._assert_or_print_error(self._argument_is_type(node.children[1], int), error)
        self._assert_or_print_error(self._argument_is_type(node.children[2], int), error)
        self._assert_or_print_error(self._argument_is_type(node.children[3], int), error)
    
    def visit_GemmAccInit(self, node: GemmAccInit):
        error = "GemmAccInit gemm: TiledGemm, r: int, c: int"
        self._assert_or_print_error(self.current_stack_idx > 1, "GemmAcc can only be declared in kernel")
        if not self._assert_or_print_error(len(node.children) == 4, error):
            return
        self._assert_or_print_error(all(isinstance(c, Argument) for c in node.children[1:]), f"Children of GemmAccInit must be arguments")
        self._assert_or_print_error(isinstance(node.children[0], VarDecl) and self._decl_is_type(node.children[0], T.WgmmaAcc), f'GemmAccInit must declare WgmmaAcc')
        self._assert_or_print_error(self._argument_is_type(node.children[1], T.TiledGemm), error)
        self._assert_or_print_error(self._argument_is_type(node.children[2], int), error) # do we even need these, shouldn't this be in the wgmmaAcc metadata??
        self._assert_or_print_error(self._argument_is_type(node.children[3], int), error)
    
    def visit_PipelineInit(self, node: PipelineInit):
        self._assert_or_print_error(self.current_stack_idx == 2, "Pipeline can only be declared in kernel")
        error = "Pipeline decl: VarDecl stages: int"
        if not self._assert_or_print_error(len(node.children) == 2, error): # TODO add shared pointer somehow
            return
        self._assert_or_print_error(isinstance(node.children[0], VarDecl) and self._decl_is_type(node.children[0], T.Pipeline), error)
        self._assert_or_print_error(isinstance(node.children[1], Argument) and self._argument_is_type(node.children[1], int), error)
    
    def visit_TMALoadG2S(self, node: TMALoadG2S):
        self._assert_or_print_error(self.current_stack_idx > 1, "TMALoadG2S can only be done in kernel")
        error = "TMALoadG2S: global_tensor: GlobalTensor, shared_tensor: GlobalTensor, tile_size_m: int, tile_size_n: int, coord_row: int, coord_col: int, loop_var: int, shared_idx: int"
        if not self._assert_or_print_error(len(node.children) == 8, error):
            return
        for i, t in enumerate((T.GlobalTensor, T.SharedTensor, int, int, int, int, T.Pipeline, T.PipelineState)):
            self._assert_or_print_error(isinstance(node.children[i], Argument) and self._argument_is_type(node.children[i], t), f'TMALoadG2S: Problem at {node.children[i]}. Expected type {str(t)}')


class LowerVisitor:
    label="Lower"
    def __init__(self):
        pass

    def enter(self, node): ...

    def exit(self, node: NodeBase):
        # flatten children
        node.children = flatten_once(node.children)
    
    def visit(self, node: NodeBase):
        output = node.accept(self)
        if isinstance(output, list):
            p = node.parent
            p.replace_child(node, output)

    def visit_TMALoadG2S(self, node: TMALoadG2S):
        # TODO I can maybe use the type() type declaration method to actually create quick custom classes
        pipe_wait = CustomFuncNode(stmt=lambda self: f'{node.state}.producer_acquire()', children=[node.state, node.pipe])
        load_node = CustomFuncNode(stmt=lambda self: f'TMAload({node.global_tensor}, {node.shared_tensor}, {node.coords}, {node.state}, {node.pipe})', children=[node.global_tensor, node.shared_tensor, *node.coords, node.state, node.pipe])
        pipe_commit = CustomFuncNode(stmt=lambda self: f'{node.state}.producer_commit()', children=[node.state])
        return [pipe_wait, load_node, pipe_commit]

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
c_g = Var('c')
pipe_a = Var('pipe_a')
pipe_b = Var('pipe_b')
acc = Var('acc')
nthreads = Var('nthreads')
a_state = Var('a_state')
b_state = Var('b_state')
kernel = KernelFunction()
kernel.add_child(VarReference(nthreads))
kernel.add_child(PipelineInit(VarDecl(pipe_a), VarReference(stages)))
kernel.add_child(PipelineInit(VarDecl(pipe_b), VarReference(stages)))
kernel.add_child(SharedTensorInit(var_ref2(tile_m), var_ref2(tile_k), var_ref2(stages), VarDecl(Var('As')))) # but for SMEM we should store the dims ahead of time
kernel.add_child(SharedTensorInit(var_ref2(tile_n), var_ref2(tile_k), var_ref2(stages), VarDecl(Var('Bs'))))
kernel.add_child(GemmAccInit(var_ref('tiled_mma'), VarReference(tile_m), VarReference(tile_n), VarDecl(acc)))
loop = Loop('k', ConstantReference(4))

# TODO fix these constant references to bidx, bidy
loop.add_child(TMALoadG2S(var_ref('a'), var_ref('As'), var_ref2(tile_m), var_ref2(tile_k), ConstantReference('bidx'), None, var_ref('k'), var_ref2(pipe_a), var_ref2(a_state))) # need to add pipelinestate
loop.add_child(TMALoadG2S(var_ref('b'), var_ref('Bs'), var_ref2(tile_n), var_ref2(tile_k), ConstantReference('bidy'), None, var_ref('k'), var_ref2(pipe_b), var_ref2(b_state)))
loop.add_child(Gemm(var_ref('tiled_mma'), var_ref('As'), var_ref('Bs'), var_ref('acc'), var_ref2(a_state), var_ref2(b_state)))

scope = Scope(256)
scope.add_child(PipelineStateInit(VarDecl(a_state), VarReference(stages))) # we can also check that these aren't classvars
scope.add_child(PipelineStateInit(VarDecl(b_state), VarReference(stages))) # TODO use this
scope.add_child(loop)
scope.add_child(EpilogueStore(VarReference(acc), VarReference(c_g)))
kernel.add_child(scope)
# print(kernel)

# TODO need to enforce runtime vars can't be declared in __call__ e.g. pipeline, etc.
callfunction = CallFunction()
callfunction.add_child(TiledGemmInit(VarDecl(Var('tiled_mma')), VarReference(dtype), 
                   ConstantReference('ROW'), ConstantReference('COL'), # higher-level nodes decide this
                   VarReference(acc_dtype), VarReference(tile_m), VarReference(tile_n), VarReference(tile_k)))
callfunction.add_child(GlobalTensorInit(GlobalMatrix((64, 128), (0, 1)), VarDecl(a_g))) # we can store this data here since we'd only know this at runtime
callfunction.add_child(GlobalTensorInit(GlobalMatrix((128, 128), (0, 1)), VarDecl(b_g)))
callfunction.add_child(GlobalTensorInit(GlobalMatrix((64, 128), (0, 1)), VarDecl(c_g)))
callfunction.add_child(CalculateTotalThreads(VarDecl(nthreads, int), ConstantReference(False), var_ref('tiled_mma')))
callfunction.add_child(kernel)

kc = KernelClass()
kc.add_child(ClassArgDecl(tile_m, int))
kc.add_child(ClassArgDecl(tile_n, int))
kc.add_child(ConstantDecl(tile_k, 64))
kc.add_child(ClassArgDecl(stages, int))
kc.add_child(ConstantDecl(dtype, 'cutlass.Bfloat16'))
kc.add_child(ConstantDecl(acc_dtype, 'cutlass.Float32'))
kc.add_child(callfunction)
# print(kc)

dv = DeclCheckVisitor()
postorder_visit(dv, kc)
# print(kc)

pv = LowerVisitor()
postorder_visit(pv, kc)
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

# TODOs for figuring out kernel:
# - permuting global matrices, if needed(that can come from higher-level)
# - deciding attrs like row/col major when initializing gemm(can come from higher-level)
# - get gridsize
# - get cluster size: start with 1 for now
# - sort out the block stuff, we should simplify to warpids if possible. ALSO have to deal with producer loop which has two scopes technically
# - k iter undecided
# - if loops have increment instructions we need to apply it
# - get kernel args

# after...
# - split things up
# - move things to better levels
# - emit lower-level stuff to lower to cutedsl
# - break things into PC
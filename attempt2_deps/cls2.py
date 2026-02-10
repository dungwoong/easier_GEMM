from dataclasses import dataclass, field
from typing import List
import textwrap
from functools import partial

@dataclass
class ForLoop:
    # num_warps: int
    symbols: dict # arr, extent and we assume things loop in a modulo
    start: int = 0


# TODO we probably need to have some sort of iteration marker
@dataclass
class Op:
    reads: set = field(default_factory=set)
    writes: set = field(default_factory=set)
    tag: str = "S"

    def getidx(self, x: int, extent: int, var: str):
        """the index of <var> that we access on loop <x>, 
        where var has <extent> stages"""
        return 0

    def __hash__(self):
        return id(self)


# we can hack this to be like "A_inflight" and then wait writes from A_inflight to As
class LoadNext(Op): # this assumes everything loops in a modulo fashion
    def __init__(self, f, t):
        super().__init__()
        self.f, self.t = f, t
        self.reads.add(f)
        self.writes.add(t)
    
    def getidx(self, x, extent, var):
        return x % extent # HACK: emulate load next
    
    def __repr__(self):
        return f'{self.tag}: {self.t} = load({self.f})'

class Wait(Op):
    # model wait as a fetch from an inflight copy to an inflight store
    # can't we somehow keep track of equivalences to know what data will be fed into what idk
    def __init__(self, fs, ts, k):
        super().__init__()
        self.k = k
        self.reads = self.reads.union(fs)
        self.writes = self.writes.union(ts)

    def getidx(self, x, extent, var):
        if var in self.writes:
            return (x % extent)
        else:
            return [(x+extent - i) % extent for i in range(self.k + 1)]
    
    def idx_to_empty(self, x, extent):
        return (x + extent - self.k) % extent
    
    def __repr__(self):
        return f'{self.tag}: wait({self.k})'

class MMASync(Op):
    def __init__(self, a, b, c):
        super().__init__()
        self.a, self.b, self.c = a, b, c
        self.reads.add(a)
        self.reads.add(b)
        self.reads.add(c)
        self.writes.add(c)
    
    def getidx(self, x, extent, var):
        return x % extent
    
    def __repr__(self):
        return f'{self.tag}: MMA({self.a}, {self.b}, {self.c})'


# S1 -[k]-> S2 means S1[i] must happen before S2[i+k]
class Dependency:
    def __init__(self, s1: Op, s2: Op, t, vec):
        self.s1, self.s2, self.t, self.vec = s1, s2, t, vec
    
    def __repr__(self):
        return f'{self.s1.tag} -[{self.vec}]-> {self.s2.tag} ({self.t})'

def enumerate_statements(stmts):
    for i, stmt in enumerate(stmts):
        stmt.tag = f'S{i}'

def _make_symbol_dict(loop: ForLoop):
    ret = dict()
    for s in loop.symbols:
        for k in range(loop.symbols[s]):
            ret[(s, k)] = None
    return ret

def add_dependency(new_dep, deps, track):
    # if S1 -[0]-> S2 then you can't have S1 -[1]-> S2 it's redundant
    s1, s2, vec = new_dep.s1, new_dep.s2, new_dep.vec
    if (s1, s2) in track and track[(s1, s2)] <= vec:
        return
    deps.append(new_dep)
    track[(s1, s2)] = vec

# if you're running the kth iter of the loop, and we have dep on first iter, A[i] --> A[i+k] so dependency is [k]
def add_dependencies(pre_stmts: List[Op], loop_stmts: List[Op], loop: ForLoop, unroll=0):
    last_write = _make_symbol_dict(loop) # RAW, WAW
    last_read = _make_symbol_dict(loop) # WAR
    deps = []
    tracker = dict()
    add_dep = partial(add_dependency, deps=deps, track=tracker)
    dep_vector = 0

    def check_wait(s, k):
        if not isinstance(s, Wait):
            return
        for r in s.reads:
            for i in range(s.k + 1):
                tmp = (r, i)
                assert last_write[tmp] is not None, f'Wait statement where buffer {i} was empty {str(s)}'

    def process(s, k):
        for r in s.reads:
            # the correct way to do waits is if wait(k) then you need k+1 buflen and you have to check the ENTIRE buffer is full but only write the last item
            # we can even check that you don't have a WAW to an inflight buffer, meaning you have two writes without a read
            if isinstance(s, Wait):
                for idx in range(s.k + 1):
                    assert last_write[(r, idx)] is not None, f'Wait statement where buffer {idx} was empty {str(s)}'
                    add_dep(Dependency(last_write[(r, idx)], s, 'RAW', dep_vector))
                tmp = (r, s.idx_to_empty(k, loop.symbols[r]))
                last_write[tmp] = None # empty last buffer
            else:
                tmp = (r, s.getidx(k, loop.symbols[r], r))
                if last_write[tmp] is not None: # RAW
                    add_dep(Dependency(last_write[tmp], s, 'RAW', dep_vector))

                    if isinstance(s, Wait):
                        last_write[tmp] = None
            last_read[tmp] = s
        for r in s.writes:
            tmp = (r, s.getidx(k, loop.symbols[r], r))
            if last_read[tmp] is not None: # WAR
                add_dep(Dependency(last_read[tmp], s, 'WAR', dep_vector))
            if last_write[tmp] is not None: # WAW
                add_dep(Dependency(last_write[tmp], s, 'WAW', dep_vector))
            last_write[tmp] = s

    # if a statment reads/writes the same memory, you get a WAR
    for s in pre_stmts:
        process(s, 0)
    for k in range(loop.start, loop.start + unroll + 1):
        for s in loop_stmts:
            process(s, k)
        dep_vector += 1
            
    return deps

def print_loop(l, pre_stmts, stmts):
    s = ''
    for stmt in pre_stmts:
        s += str(stmt) + '\n'
    s += 'for ...\n'
    stm = ''
    for stmt in stmts:
        stm += str(stmt) + '\n'
    s = s + textwrap.indent(stm, '  ')
    print(s)

l = ForLoop({'A': 10, 'B': 10, 'As': 2, 'Bs': 2, 'As_inflight': 2, 'Bs_inflight': 2, 'Cr': 1}, start=1)
stmts = [LoadNext('A', 'As_inflight'), LoadNext('B', 'Bs_inflight'), Wait(['As_inflight', 'Bs_inflight'], ['As', 'Bs'], 1), MMASync('As', 'Bs', 'Cr')]
pre_stmts = [LoadNext('A', 'As_inflight'), LoadNext('B', 'Bs_inflight')]
enumerate_statements(pre_stmts + stmts)
deps = add_dependencies(pre_stmts, stmts, l, unroll=3)
print_loop(l, pre_stmts, stmts)
for d in deps:
    print(d)
# you'll get MMA --> MMA(WAR)
# if we had buffers, we should have a dependency vector of size [nstages]

# Let's try to add wait and arrive
# we can try to hack it so arrive is like reading from the buffers, wait is like writing
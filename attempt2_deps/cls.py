from dataclasses import dataclass, field
from typing import List
import textwrap

@dataclass
class ForLoop:
    num_warps: int
    symbols: dict


# TODO we probably need to have some sort of iteration marker
@dataclass
class Op:
    reads: set = field(default_factory=set)
    writes: set = field(default_factory=set)
    tag: str = "S"


class LoadSync(Op):
    def __init__(self, f, t):
        super().__init__()
        self.f, self.t = f, t
        self.reads.add(f)
        self.writes.add(t)
    
    def __repr__(self):
        return f'{self.tag}: {self.t} = load({self.f})'

class ProdArrive(Op):
    def __init__(self, loaded):
        super().__init__()
        self.reads = self.reads.union(loaded)
    
    def __repr__(self):
        return f'{self.tag}: Arrive({self.reads})'

class ConsWait(Op):
    def __init__(self, loaded):
        super().__init__()
        self.writes = self.writes.union(loaded)
    
    def __repr__(self):
        return f'{self.tag}: Wait({self.writes})'

class MMASync(Op):
    def __init__(self, a, b, c):
        super().__init__()
        self.a, self.b, self.c = a, b, c
        self.reads.add(a)
        self.reads.add(b)
        self.reads.add(c)
        self.writes.add(c)
    
    def __repr__(self):
        return f'{self.tag}: MMA({self.a}, {self.b}, {self.c})'

class OnlineSoftmax(Op):
    def __init__(self, x, scale):
        super().__init__()
        self.x, self.scale = x, scale
        self.reads.add(x)
        self.writes.add(scale)
    
    def __repr__(self):
        return f'{self.tag}: {self.scale} = OSoftmax({self.x})'

class RescaleO(Op):
    def __init__(self, x, scale):
        super().__init__()
        self.x, self.scale = x, scale
        self.reads.add(x)
        self.reads.add(scale)
        self.writes.add(x)
    
    def __repr__(self):
        return f'{self.tag}: {self.x} = {self.scale} * {self.x}'



class Dependency:
    def __init__(self, s1: Op, s2: Op, t):
        self.s1, self.s2, self.t = s1, s2, t
    
    def __repr__(self):
        return f'{self.s1.tag} --> {self.s2.tag} ({self.t})'

def enumerate_statements(stmts):
    for i, stmt in enumerate(stmts):
        stmt.tag = f'S{i}'

def add_dependencies(stmts: List[Op], loop: ForLoop):
    last_write = {s: None for s in loop.symbols} # RAW, WAW
    last_read = {s: None for s in loop.symbols} # WAR
    deps = []

    # if a statment reads/writes the same memory, you get a WAR
    for s in stmts:
        for r in s.reads:
            if last_write[r] is not None: # RAW
                deps.append(Dependency(last_write[r], s, 'RAW'))
            last_read[r] = s
        for r in s.writes:
            if last_read[r] is not None: # WAR
                deps.append(Dependency(last_read[r], s, 'WAR'))
            if last_write[r] is not None: # WAW
                deps.append(Dependency(last_write[r], s, 'WAW'))
            last_write[r] = s
    return deps

def print_loop(l, stmts):
    s = 'for ...\n'
    stm = ''
    for stmt in stmts:
        stm += str(stmt) + '\n'
    s = s + textwrap.indent(stm, '  ')
    print(s)

l = ForLoop(0, {'ag': None, 'bg': None, 'a': None, 'b': None, 'c': None, 'scale': None, 'vg': None, 'v': None, 'o': None})
stmts = [LoadSync('bg', 'b'), ProdArrive(['b']), LoadSync('vg', 'v'), ConsWait(['b']), MMASync('a', 'b', 'c'), ProdArrive(['v']), OnlineSoftmax('c', 'scale'), RescaleO('c', 'scale'), ConsWait(['v']), MMASync('c', 'v', 'o')]
enumerate_statements(stmts)
deps = add_dependencies(stmts, l)
print_loop(l, stmts)
for d in deps:
    print(d)
# you'll get MMA --> MMA(WAR)
# if we had buffers, we should have a dependency vector of size [nstages]

# Let's try to add wait and arrive
# we can try to hack it so arrive is like reading from the buffers, wait is like writing
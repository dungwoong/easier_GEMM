
```
for ...:
    Qs = Load(Q, method='tma') # shared
    Ks = Load(K, method='tma') # shared
    p = Gemm(Q, K) # output regs
    scale = OnlineSoftmax(p) # output regs
    o_acc = Rescale(scale, o_acc) # output regs
    Vs = Load(V) # shared
    o_acc += Gemm(p, Vs, o_acc) # output regs
```

So first, high-level graph. then we must create a lowering to an equivalent loop program. This lowering process may be difficult
THEN, we can use e-graphs to figure out memory, resulting in the code here.
THEN, this will lower even more and we can do warp specialization etc.

For example, Load(Q, method='tma') expands into an instant load and barrier arrive, and we can choose to warp specialize or reorder it
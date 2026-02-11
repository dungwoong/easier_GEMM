TODOs for figuring out kernel:
- permuting global matrices, if needed(that can come from higher-level)
- deciding attrs like row/col major when initializing gemm(can come from higher-level)
- get gridsize
- get cluster size: start with 1 for now
- sort out the block stuff, we should simplify to warpids if possible. ALSO have to deal with producer loop which has two scopes technically
    - the kernel class can create these scopes/annotate the objects so everytime it encounters a scope, annotate the scope and increment the counter
    - kernel can also check that the counter doesn't go over the number of threads(maybe) and add an assert statement in the `__call__` right before the kernel(?)
    - Finally, we're going to have warpid as a variable inside the kernel. Let's simplify to warpid when possible
    - if you have a scope in a scope(kernel counts as a scope) and both have constant nthreads then maybe you can check that threads is valid
    - we can traverse in order and implement this as a visitor
- [x] k iter undecided
    - this could just be passed as a constant for our purposes, mirage etc. do this
- if loops have increment instructions we need to apply it
    - we just have to track the instructions and yeah
- [x] get kernel args
    - whatever args in `__call__` are referenced in the kernel, we can populate this in a visitor?
- If we're going to populate tx, we need to do so in __call__ and we also need tma atoms/tensors
- actually in the setup we have to declare tma atom/tensor and get tx count for stuff
- we should store a source node or smth in case we need it(when we add higher-level nodes)
- add an implicit declaration so we can just "declare" bidx, bidy etc. if needed? Actually we might want to explicitly call cute.block_idx or whatever


When we move stuff around
- we cannot move stuff from kernel to `__call__`. We don't know what datatypes we're moving so that is final. We can move stuff around in kernel though

Later on
- If we're doing multiple ops in one we should fuse the math e.g. doing multiple elementwise we need to generate a fused op for that

## Add Types
- so most things, we can actually use the cutlass typing system. It's just for blocks I want to keep track of metadata tbh.
- Actually, for MMA, loads, computations we should store data types(e.g. when initializing an MMA or computation)
- we can make dummy types and I think Type() is still good since isinstance won't work on e.g. cutlass.Float32
- still need to do checks with these types but at least type system is implemented(kinda)
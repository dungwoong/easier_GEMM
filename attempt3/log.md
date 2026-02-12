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

## Egg
- export e-graph?
- export entire e-graph?
- scoring stuff?
- serialization and custom extraction
- analysis?
- reprod tensat-MCTS

## other
- triton has convert-layout operations that do stuff
- triton tile sizes are known compile-time, matrix sizes aren't
- other MLCompilers you know matrix size but not tile size
- what layouts are possible might depend on warp config, tile size etc. that might be the biggest problem
- modelling synchronization could be difficult
- making more decisions at the e-graph level would be useful
- consider what's possible in cutedsl, since we can't go lower than that
- main problem is fusing incompatible layouts or something like that and checking validity of those

checks
- types match, sizes match, compatible layouts. I can add checks as I go
- within passes you can make sure an optimization is doable

## additions
- scopes specify nthreads and specific operations can specify the number of threads participating
- even like warpgroup.regalloc and stuff can do this
- but if nthreads is dynamic, then we can't tell
- we can add these checks once we have some sort of kernel up and running
- we ALSO have to specify memory regions for stuff and read/write to them to model the lower-level dependencies
    - only RuntimeFuncNodes will have reads/writes, but we need to know how to add these reads/writes to them...

## TODOs
- pipeline needs consumer/producer groups I think
- [DONE] need pipeline stage
- we can add a layout permute function for global matrix
- pipeline: if pipeline extent is 1 we can have it generate differently so it makes an mbarrier instead yknow

Tracking threads
- we can infer how many threads come from the type of certain variables(e.g. the tiledMMA or the pipeline)

Tracking reads/writes
- the type classes could have functions that accept an access, and they store what last accessed them so then all we have to do is use these different types

Scopes
- we could attach a minimum scope to everything e.g. things that must be in the kernel as a flag, so we can move setup things around.
- we just don't touch runtime things

Adding tile scheduler
- probably best to abstract the grid thing away in a tile scheduler, so we can write a generic one for GEMM or smth
- this will get rid of all problems related to finding what tile to do

We can probably just start transforming everything to the equivalent cutedsl stuff
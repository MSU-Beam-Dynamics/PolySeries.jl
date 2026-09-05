# Composition memory and performance review

The retained-image implementation builds one full `N`-coefficient polynomial
per source monomial through its highest active degree. Full-order sources
therefore retain `N²` coefficient slots even though there are only `N-1`
multiplications. Sparse source coefficients previously avoided accumulation,
but did not avoid building their monomial images.

For six variables and order ten, `N = binomial(16, 10) = 8008`:

| Strategy | Coefficient slots | Float64 coefficient storage |
| --- | ---: | ---: |
| Retain all images | 64,128,064 | 489.26 MiB |
| Depth-first workspace | 88,088 | 688.19 KiB |

These are calculated buffer sizes, excluding descriptors, inputs, output, and
object metadata. The new workspace retains one image per tree depth, including
the root, for `(order + 1) * N` slots. Child/sibling indices and pruning flags
add O(N) metadata. Default `compose!` creates this workspace; its measured total
allocation at N=8008 is 788,912 bytes (770.42 KiB). Passing a reusable workspace
eliminates allocation during Float64 composition.

## Implementation

`CompositionWorkspace(desc, T)` stores depth buffers and the parent tree's
adjacency lists. Each call marks nonzero coefficients in active degree blocks,
then marks their ancestors in one reverse pass. A depth-first traversal builds
only the required images, accumulating each before reusing its buffer for a
sibling. The multiplication count is at most one per required non-root node.
For `x₁¹⁰` in the six-variable descriptor, only 11 nodes are required, including
the root, instead of 8,008.

The pruning scan is still O(N); it is not proportional only to the number of
nonzero coefficients. Workspaces are sized for the descriptor's full order,
including when a particular source is constant or low order. They must be
owned by one concurrent caller, and output/input aliasing is unsupported.
BigFloat scalar operations can allocate even with a reusable workspace.

Enzyme automatically takes the retained-image path. Reusing intermediate
storage and pruning numerically zero coefficients are intentionally confined
to ordinary evaluation: reverse differentiation needs primal dependencies,
including dependencies through coefficients whose values are zero. Its
worst-case memory cost remains O(N²). Reducing the reverse-mode cost would
require a separately validated differentiation rule or checkpointing scheme.

## Measurements

Julia 1.12.7, aarch64, apple-m3. See
[the benchmark script](benchmark_composition.jl) and
[complete CSV results](benchmark_composition_results.csv). Times are the minimum
of five warm runs, with descriptor and reusable-workspace setup excluded.
The ordinary strategy includes its per-call workspace allocation. These short
local measurements are indicative rather than a throughput guarantee.

| Source / map | Strategy | Bytes per call | Time |
| --- | --- | ---: | ---: |
| 4 variables, order 6, dense / identity | Retained | 394,944 | 30.63 µs |
| Same | Ordinary | 15,200 | 20.17 µs |
| Same | Reused workspace | 0 | 18.46 µs |
| 6 variables, order 10, dense / identity | Ordinary | 788,912 | 8.45 ms |
| Same | Reused workspace | 0 | 8.31 ms |
| 6 variables, order 10, dense / shifted | Reused workspace | 0 | 31.58 ms |
| 6 variables, order 10, x₁¹⁰ / identity | Reused workspace | 0 | 11.83 µs |
| 6 variables, order 10, x₁¹⁰ / shifted | Reused workspace | 0 | 19.13 µs |

The N=8008 retained baseline was not executed; its storage above is calculated.
The script's `--large-baseline` option enables it. Runtime gains depend on source
sparsity and the substitution map; small dense cases can see minor setup overhead.
Depth-first accumulation also changes floating-point summation order.

## Validation

Tests cover existing composition examples, poisoned inactive storage, sparse
branch counts, reuse across sparse/constant/zero sources, descriptor mismatches,
order zero, dense nonlinear substitutions compared with retained images, and
Float64/BigFloat coefficients. Allocation checks enforce zero bytes for a warm
Float64 workspace call and reject a return to quadratic default allocation.
Enzyme checks cover derivatives through zero source coefficients and zero
substitution centers, including the explicit workspace overload.

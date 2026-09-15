# Output-coefficient multiplication

This continues `performance_tweak` from `c11343a` and targets independent dense
products rather than only squaring. The arithmetic API and version remain
unchanged.

## Selected implementation

The new kernel groups all input monomial pairs contributing to each output
coefficient. It accumulates them in one scalar and writes the coefficient once.
An off-diagonal pair contributes `a[i]*b[j] + a[j]*b[i]`; a diagonal contributes
once. A descriptor-owned read-only plan stores offsets, paired input indices,
and diagonal indices.

The production dispatcher enables the plan only for full Float32/Float64
products with one variable at orders 6–63 or two variables at orders 6–20.
These descriptors use UInt16 indices. The internal builder also supports Int32
for experiments, including a test above index 65,535. No plan is built for
other descriptors. Existing specialized squaring takes precedence when selected.

A short prefix check falls back to the previous kernel after finding four
positions where both operands are numerically zero. That avoids much of the
wasted work for conservative full masks whose coefficient arrays are mostly
zero. It is a heuristic for performance, not a source of coefficient activity:
both kernels compute the full product. Degree holes always retain the existing
mask-aware kernels, so inactive storage is never read by the new kernel.

The plan builder runs under the existing descriptor-cache lock. Descriptor
memory-limit checks include the plan and its count/cursor construction scratch.
The existing real Enzyme multiplication rules still calculate the mathematical
convolution derivative, independently of the primal kernel choice. Complex AD
retains its existing path.

## Experiments and limitations

The unrestricted output-coefficient layout was slower than the existing kernel
at six and eight variables. The likely explanation is that reading two index
streams and gathering both inputs outweighed the reduction in output writes;
this is an inference from the kernels and timings, not hardware-counter profiling. Smaller indices helped
slightly. SIMD reductions and four/eight independent accumulators did not close
the gap reliably. This is why the layout is selected narrowly rather than used
for all dense products.

Two-variable products at very high orders also lost the advantage. The current
selection stops at order 20. Products with three or more variables, other
coefficient types, and most sparse workloads continue to use the prior kernels.
Performance conclusions are specific to the measured workloads and hardware;
this is not a new comparison with GTPSA.

## Final measurements

Measured September 15, 2026, on Apple M3, Julia 1.12.7, Enzyme 0.13.173,
with one Julia thread. The [full results](../benchmarks/output_product_results.csv)
record the source hashes, minimum/median times, and allocations. Representative
Float64 warmed medians against `c11343a` are:

| Operation | Variables/order | Before (µs) | After (µs) | Speedup |
|---|---|---:|---:|---:|
| Independent dense product | 1/8 | 0.0703 | 0.0404 | 1.74× |
| Independent dense product | 1/20 | 0.2900 | 0.1057 | 2.74× |
| Independent dense product | 1/63 | 3.2619 | 0.6808 | 4.79× |
| Independent dense product | 2/6 | 0.1407 | 0.1023 | 1.37× |
| Independent dense product | 2/12 | 0.9237 | 0.5795 | 1.59× |
| Independent dense product | 2/20 | 4.5327 | 3.2969 | 1.37× |
| Independent dense product | 6/8 | 29.2360 | 29.3057 | ≈1× |
| Reverse product gradient | 1/63 | 7.4625 | 5.2868 | 1.41× |
| Reverse product gradient | 2/12 | 3.1987 | 2.9583 | 1.08× |
| Reverse product gradient | 2/20 | 13.3214 | 12.6786 | 1.05× |

All ordinary in-place measurements allocated zero bytes. The prefix classifier
added roughly 2–11 ns to several small zero-heavy cases; it therefore has a
measurable relative cost when the original operation takes only tens of
nanoseconds. Selected reverse-AD cases were faster but allocated 32 additional
bytes (one allocation) per call. The six-variable AD samples varied too much
to claim an improvement there.

Extra cached plan payloads were 4,744 bytes for 1/63, 4,502 bytes for 2/12, and
23,438 bytes for 2/20. Isolated plan construction takes a few microseconds.
Measured complete cold descriptor construction was approximately 2.10 ms,
2.05 ms, and 10.5 ms respectively in both implementations; its timing variation
was larger than the isolated plan cost. Plans also add temporary construction
allocations, recorded separately in the results.

The [layout comparisons](../benchmarks/output_layout_results.csv) preserve the
evidence for retaining the old higher-dimensional kernel, including compact
indices and multiple-accumulator variants that were not selected.

## Reproduction

Using the development environment described in [the first-stage report](performance_tweak.md),
run from the repository root:

```sh
julia --project=benchmarks benchmarks/benchmark_output_product.jl
julia --project=benchmarks benchmarks/benchmark_output_layouts.jl
```

The first script compares the working tree against an isolated source snapshot
of `c11343a`. It supports `--ordinary-only`, `--ad-only`, and an alternative
`POLYSERIES_BENCH_BASE`. Dense inputs have independent sine/cosine coefficient
vectors; a second fixture leaves many numeric zeros while keeping full masks.
The script records warmed execution times, Julia heap allocations, cold
descriptor construction measurements, extra table storage, and isolated plan
construction time. Compilation is excluded from the multiplication timings.
Cold construction uses five measured cache misses after one warm-up; those
measurements are noisier than isolated plan construction.

The second script compares the underlying schedule kernel against Int32 and
UInt16 output plans with scalar, SIMD, and multiple-accumulator reductions. It
checks each result before measuring. Both scripts record source hashes and
runtime details.

## Validation

The full suite passed **41,231 checks** with four Julia threads, including the
existing Enzyme regressions and executable README/documentation examples.

New tests check BigFloat coefficient references, plan pair counts, both index
widths, index-range rejection, aliased outputs, poisoned degree gaps, dense
masks containing numeric zeros, allocation behavior, and type inference. AD
tests cover independent coefficient directions, entirely inactive prebuilt
inputs, forward/reverse modes, aliases, and compiled derivatives reused across
descriptors with different plans.

# Dedicated squaring

This continues the `performance_tweak` work from commit `e2259a0`.

## Implementation

`mul!(out, p, p)` uses a dedicated kernel when the descriptor has at least
128 coefficients and the active degrees extend beyond a constant/linear
polynomial. The same path benefits `p*p` and squaring steps in integer powers.
The existing alias handling also makes it available to `mul!(p,p,p)`.
The dispatch requires the same coefficient buffer and matching masks; it does
not compare polynomial values or mistake different masks for the same operand.

For each pair of different monomials, the kernel computes one product and adds
it twice. A diagonal product is added once. It doubles the computed product
rather than doubling an input coefficient first, avoiding premature overflow.
The kernel visits only valid active degree pairs, including masks with holes.
It introduces no coefficient buffers or descriptor tables.

Small descriptors retain their previous kernel because the initial benchmarks
showed that specialized-loop overhead could exceed the arithmetic savings.
General multiplication of distinct buffers retains its existing kernel.

The real Enzyme multiplication primitive can also use this primal kernel.
Its existing mathematical rules still differentiate both arguments, including
independent tangent directions and aliased inputs. Complex AD keeps its existing
differentiated implementation.

## Measurements

Measured September 14, 2026 (local time), on Apple M3, Julia 1.12.7,
Enzyme 0.13.173, with one Julia thread. The baseline is `e2259a0`.
Times below are warmed medians; the complete results (`benchmarks/squaring_results.csv`)
also record minimum times, allocation counts, and source hashes.

| Operation | Variables/order | Before (µs) | After (µs) | Time reduction |
|---|---|---:|---:|---:|
| Dense square | 4/6 | 0.960 | 0.853 | 11.1% |
| Dense square | 6/6 | 4.519 | 4.125 | 8.7% |
| Dense square | 6/8 | 28.625 | 25.641 | 10.4% |
| Dense square | 6/10 | 139.583 | 122.084 | 12.5% |
| Dense square | 6/12 | 563.625 | 487.896 | 13.4% |
| Dense square | 8/8 | 162.105 | 142.416 | 12.1% |
| Independent dense product | 6/8 | 29.250 | 29.222 | ≈0% |
| Independent dense product | 6/12 | 570.500 | 575.438 | ≈0% |
| Square reverse gradient | 6/6 | 36.056 | 31.993 | 11.3% |
| Square reverse gradient | 6/8 | 230.875 | 193.875 | 16.0% |

All ordinary in-place cases allocated zero bytes. General dense multiplication
stayed within about 2% of the baseline. With many numerical zeros, squaring at
six or eight variables improved by roughly 8–12% in runtime; the smaller
four-variable case was essentially unchanged. Small descriptors retain the
old path, and small measured differences there should be treated as noise.

The reverse-AD medians improved, but their samples varied more: minimum times
improved only about 3–5% in the six-variable cases. Do not extrapolate the median
AD speedups to every workload. These reverse calls allocated an additional
64 bytes (two allocations), for example 124,048 versus 123,984 bytes at 6/8.

## Output-degree ordering and blocking

The internal squaring kernel supports column blocks for measurement. The
benchmark compares the original degree-pair order with schedules grouped by
output degree, and compares complete columns with blocks of 64 and 256 entries.
Alternative ordering shares the index matrices; it does not duplicate them.

Initial comparisons did not establish a consistent additional gain across
descriptors. These alternatives therefore remain benchmark candidates; the
production dispatcher uses the original ordering and complete columns. This
also avoids adding unused metadata to every descriptor.
The final layout measurements (`benchmarks/squaring_layout_results.csv`)
retain those comparisons separately from the before/after production results.

## Reproduction

From the repository root, using the development environment described in
[the first-stage report](performance_tweak.md):

```sh
julia --project=benchmarks benchmarks/benchmark_multiplication.jl
julia --project=benchmarks benchmarks/benchmark_multiplication.jl --layouts-only
```

`--ordinary-only` and `--ad-only` select subsets. The default baseline is
`e2259a0`; `POLYSERIES_BENCH_BASE` overrides it. The script loads the baseline
in a separate module without changing the checkout, pairs ordinary measurements
by descriptor, and records source hashes and package versions.

Ordinary fixtures use independent sine/cosine coefficient vectors, their
squares, and dense masks with many numerically zero coefficients. This separates
general multiplication from squaring, which the previous dense benchmark did
not distinguish. Reverse AD differentiates a constant plus a selected highest-
degree coefficient of the square. Timings exclude initial compilation and
descriptor construction; allocations are Julia heap allocations.

## Correctness coverage

The complete suite passed **41,039 checks** on Julia 1.12.7 with four threads,
including the Enzyme regressions and executable README/documentation examples.

The new tests compare squaring with independent exponent convolution for
Float32, Float64, ComplexF64, and BigFloat, with fully active, empty, and
separated degree masks and deliberately poisoned inactive storage. They also
cover order 63, aliased outputs, two wrappers sharing a buffer with different
masks, and products that would overflow if an operand were doubled too early.

Selected coefficients at six variables/order twelve are checked against
BigFloat references, including the alternative orderings and block sizes.
Allocation checks cover ordinary and aliased in-place squaring. Enzyme checks
include independent coefficient tangents, entirely inactive prebuilt inputs,
forward and batched forward modes, and reverse differentiation with aliases.

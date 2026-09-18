# Performance rebuild: first stage

This work lives on `performance_tweak`, based on commit `9c2d667`.
The public arithmetic API and descriptor ownership remain compatible. The
package version remains 0.1.0; this branch has not been submitted for release.

## Changes

- Multiplication by an affine polynomial uses coefficient shifts. A quadratic
  factor with at most four nonzero coefficients uses a sparse factor kernel.
  Dense multiplication and matching contiguous degree masks retain their
  existing kernels. Every ordinary kernel respects inactive degree blocks.
- Composition with `g[i] = shift[i] + x[i]` uses a triangular coefficient
  translation instead of constructing monomial images. Identity composition
  preserves the source's degree mask exactly.
- `CompositionPlan(f)` snapshots a fixed source and records the needed
  monomial traversal once. Repeated general composition can reuse that plan,
  an output, and a `CompositionWorkspace`. Descriptor-owned traversal links
  are shared; mutable image buffers remain separate for each workspace.
- Real Float32/Float64 multiplication and `exp` have explicit forward and
  reverse Enzyme rules. Multiplication uses the transpose of coefficient
  convolution; the exponential rule uses `D exp(f)[df] = exp(f) * df`.
  These rules preserve derivatives of numerically zero coefficients and cache
  primal data when the caller may overwrite it before the reverse sweep.
- `examples/08_parameter_gradients.jl` demonstrates differentiating selected
  coefficients with respect to 1,000 parameters and checks an analytic result.

## Validation

On Julia 1.12.7 with four Julia threads, the complete test entry point passed
40,509 checks, including Enzyme, executable README snippets, and standalone
examples. An additional independent multivariate binomial-reference test passed
16 checks and is now included in that entry point. The Documenter build passed.

New coverage includes poisoned inactive storage, separated active degrees,
Float32/Float64/ComplexF64/BigFloat ordinary arithmetic, order-63 boundaries,
aliasing, plan snapshots, zero-allocation workspace reuse, zero coefficient
tangents, overwritten primal buffers, and forward/batched/reverse AD. Selected
coefficient gradients are checked with 100 and 1,000 parameters.

## Measurements

Measured September 14, 2026, on Apple M3, Julia 1.12.7, Enzyme 0.13.173,
with one Julia thread. These are warmed median times from the final run.
The complete results (`benchmarks/rebuild_results.csv`) include every case,
allocation counts, and hashes of the measured source files.

| Operation | Case | Before (µs) | After (µs) | Speedup |
|---|---|---:|---:|---:|
| Dense multiplication | 6 variables, order 8 | 29.417 | 28.965 | 1.02× |
| Dense × affine | 6 variables, order 8 | 8.017 | 1.129 | 7.10× |
| Dense × sparse quadratic | 6 variables, order 8 | 15.858 | 4.140 | 3.83× |
| Coordinate-shift composition | 6 variables, order 8 | 4,896.958 | 39.448 | 124.14× |
| Nonlinear composition | 6 variables, order 8 | 15,691.958 | 10,964.958 | 1.43× |
| Sparse-source nonlinear composition, with plan | 6 variables, order 8 | 18.589 | 12.465 | 1.49× |
| Ordinary `exp!` | 6 variables, order 8 | 49.135 | 49.198 | 1.00× |
| Multiplication reverse gradient | 6 variables, order 6 | 76.312 | 34.708 | 2.20× |
| Exponential reverse gradient | 6 variables, order 6 | 150.083 | 25.813 | 5.81× |
| Selected-coefficient reverse gradient | 100 parameters | 37.750 | 4.580 | 8.24× |
| Selected-coefficient reverse gradient | 1,000 parameters | 39.312 | 6.551 | 6.00× |

Across the tested descriptors, affine multiplication improved 4–7×,
coordinate shifts 13–124×, and nonlinear composition 1.27–1.62×. Ordinary
dense multiplication, multiplication of two affine factors, and ordinary
`exp!` were essentially unchanged; a few percent difference is not evidence
of a reliable gain. All measured ordinary in-place cases allocated zero bytes.
For the 1,000-parameter gradient, Julia heap allocation fell from 31,552 to
13,216 bytes per call. The plan comparison includes a prebuilt plan; construction
cost is excluded, so it describes repeated evaluation of a fixed source.

## Reproduce the measurements

From the repository root, prepare a development-only environment and run:

```sh
julia --project=benchmarks -e 'using Pkg; Pkg.develop(path="."); Pkg.add(["BenchmarkTools", "Enzyme"])'
julia --project=benchmarks benchmarks/benchmark_rebuild.jl
```

The script loads a source snapshot of `9c2d667` in a separate module and compares
it with the working tree in one process. It does not change the checkout.
`POLYSERIES_BENCH_BASE` selects another baseline; `--ordinary-only` and
`--ad-only` select subsets. Timings exclude first-call compilation and descriptor
or workspace construction. Each case warms up first, then uses up to 250 samples
with a 0.3-second sampling budget. Reported allocations are Julia heap allocations.
The output records library versions and source hashes.

## Scope and remaining work

These measurements compare PolySeries with its own starting commit, not with a
new GTPSA run. Gains depend on polynomial support, dimensions, and order. Dense
multiplication is intentionally unchanged and remains a target for further work.

Ordinary general composition still needs roughly `(order + 1) * N` coefficient
slots in its workspace. Reverse composition retains the existing monomial-image
path, which can require O(N²) coefficient storage. A checkpointed composition
pullback is a separate next step. Complex AD and functions other than `exp`
continue to use the existing differentiated implementation. The new rules are
validated for first-order parameter AD; nested AD is not established here.

A plan owns a source snapshot: construct a new plan when the source changes.
Treat internal plan and descriptor arrays as read-only. Plans can be shared,
but simultaneous evaluations need separate mutable workspaces.

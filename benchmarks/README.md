# PolySeries.jl Benchmarks

This directory contains performance benchmarks for the TPSA package.

## Benchmark Files

### benchmark_basic.jl
Benchmarks fundamental TPSA operations:
- Addition and subtraction
- Multiplication
- Scalar multiplication
- Power operations

### benchmark_mathfunctions.jl
Benchmarks mathematical functions:
- Exponential and logarithm
- Trigonometric functions (sin, cos)
- Hyperbolic functions (sinh, cosh)
- Square root
- High-order powers

### benchmark_multiplication.jl
Detailed multiplication performance analysis across different problem sizes.

### benchmark_GTPSA.jl
The head-to-head comparison with [GTPSA.jl](https://github.com/bmad-sim/GTPSA.jl).
Requires the `benchmarks` environment, which pins GTPSA:

```bash
julia --project=benchmarks benchmarks/benchmark_GTPSA.jl          # full run
julia --project=benchmarks benchmarks/benchmark_GTPSA.jl --quick  # short budget
```

Four sections, each writing its own CSV: a composed Henon map
(`benchmark_results.csv`), multiplication (`benchmark_mul_results.csv`), the
math functions (`benchmark_mathfunc_results.csv`) and composition
(`benchmark_compose_gtpsa_results.csv`). Every ratio is GTPSA divided by
PolySeries, so a value above 1.0 means PolySeries is faster. Each CSV opens
with `#` provenance lines recording the date, machine, Julia version and both
package versions; read them with `comment="#"` (CSV.jl) or `comment='#'`
(pandas).

Three properties matter when interpreting the output:

- **The Henon map genuinely composes.** Each iteration substitutes the previous
  result, so the state densifies just as it does in real map tracking, and the
  reported `active_degree_fraction` column shows how far it got. A map that
  restarts from linear variables every step never leaves the sparse
  degree-0..2 corner and flatters any implementation that skips inactive
  degrees.
- **Sparse and dense operands are reported separately.** Sparse inputs favour
  active-degree tracking; dense inputs are the worst case. Quoting one number
  without saying which regime produced it is misleading.
- **Nothing a caller would hoist is inside a timed region.** Rotation constants
  are precomputed and the in-place paths reuse a `PSWorkspace`, so the
  measurement is of the kernels rather than of scratch allocation.

Times are the minimum over a sampling budget (2 s per measurement, 0.4 s under
`--quick`). Any operation that a given GTPSA version does not support is
reported as `NaN` rather than aborting the run, and every skip is written to
`benchmark_skips.log` with its error message. `--only=compose,math` re-runs a
subset of the sections.

PolySeries composition is called as `PolySeries.compose` / `PolySeries.compose!`
in this script: GTPSA exports `compose!` too, so under `using PolySeries, GTPSA`
the bare name is ambiguous and throws. `mul!` needs no such qualification
because both packages extend `LinearAlgebra.mul!` — one shared binding rather
than two competing exports.

### benchmark_composition.jl
Compares retained images, ordinary depth-first evaluation, and a reusable
`CompositionWorkspace` for dense and sparse sources with identity and shifted
maps. It uses only Julia's built-in timing/allocation tools. Run from the root:

```bash
julia --compiled-modules=existing --project=. benchmarks/benchmark_composition.jl
```

Descriptor and workspace construction and compilation are outside timing.
CSV output reports the minimum of five warm runs and allocated bytes per call.
The 8,008-monomial retained baseline is skipped by default; use
`--large-baseline` to include its roughly 489 MiB coefficient allocation.

### benchmark_ad_sparsity.jl
Measures the steady-state time and allocation cost of ordinary execution,
Enzyme forward mode, and Enzyme reverse mode after inactive coefficient
derivatives are preserved. It compares poisoned inactive inputs with fully
materialized zero inputs from 9 through 8,008 coefficients.

`benchmark_ad_sparsity_ordinary.jl` runs the ordinary subset in a fresh process
that never loads Enzyme, for a strict measurement of the non-AD path.

## Running Benchmarks

**Prerequisites:**
```julia
using Pkg
Pkg.add("BenchmarkTools")
```

**Run a benchmark:**
```bash
julia --project=. benchmarks/benchmark_basic.jl
```

Or from Julia REPL:
```julia
using Pkg
Pkg.activate(".")
include("benchmarks/benchmark_basic.jl")
```

## Benchmark Results

Results will show:
- **Median time**: Most representative timing
- **Mean time**: Average over many runs
- **Allocations**: Number of memory allocations
- **Memory**: Total memory allocated

## Performance Tips

For accurate benchmarks:
1. Close other applications
2. Run benchmarks multiple times
3. Warm up the JIT compiler (BenchmarkTools does this automatically)
4. Use `@benchmark` from BenchmarkTools for reliable statistics

## Notes

- `benchmark_GTPSA.jl` needs the `benchmarks` environment (it depends on GTPSA
  and BenchmarkTools); the other scripts run under the package environment.
- Results may vary by hardware and Julia version, which is why the CSVs carry
  provenance headers. Re-run a benchmark before quoting it against code that
  has changed since the recorded date.

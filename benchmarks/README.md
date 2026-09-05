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

## Development Benchmarks

Historical and development-related benchmarks have been moved to `../dev_scripts/`:
- GTPSA comparison benchmarks
- Internal implementation benchmarks
- Allocation profiling scripts
- Three-way comparison tests

## Notes

- All benchmarks use the simplified API with `set_descriptor!()`
- BenchmarkTools automatically handles warm-up and statistical analysis
- Results may vary by hardware and Julia version

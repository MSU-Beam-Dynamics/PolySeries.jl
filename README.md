# PolySeries.jl

[![CI](https://github.com/MSU-Beam-Dynamics/PolySeries.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/MSU-Beam-Dynamics/PolySeries.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/MSU-Beam-Dynamics/PolySeries.jl/graph/badge.svg)](https://codecov.io/gh/MSU-Beam-Dynamics/PolySeries.jl)

**Truncated Power Series Algebra for Julia**

PolySeries.jl computes multivariate Taylor expansions of arbitrary functions to high orders. It overloads all standard arithmetic operators and transcendental functions so that code written for ordinary `Float64` scalars also works for `CTPS` objects (struct of PolySeries) — producing exact Taylor series rather than single numbers.

## Highlights

- **Automatic differentiation through order 63** — all partial derivatives up to the chosen order emerge as coefficients of the series.
- **[Enzyme.jl compatible](examples/07_enzyme_ad.jl)** — differentiate through TPSA computations to get sensitivities of Taylor coefficients w.r.t. scalar design parameters.
- **Sparse degree-mask representation** — only active degree blocks are touched; constant-only inputs have near-zero overhead.
- **Lazy-zero allocation** — temporaries use `undef` memory; the `degree_mask` invariant ensures garbage outside the active range is never read.
- **Zero-allocation in-place API** — `mul!`, `add!`, `scaleadd!`, `pow!`, etc., plus `PSWorkspace` for pool-based temporary management.
- **`@tpsa` macro** — compiles an arithmetic expression into an optimal in-place call sequence, borrowing workspace slots automatically.
- **Thread-safe** — task-local defaults and separate workspaces documented and tested.

## Installation

```julia
using Pkg
Pkg.add("PolySeries")          # once registered; until then:
Pkg.add(url="https://github.com/MSU-Beam-Dynamics/PolySeries.jl")
```

## Minimal example

<!-- readme-test -->
```julia
using PolySeries

set_descriptor!(2, 6)    # 2 variables, max order 6

x = CTPS(0.0, 1)         # variable x
y = CTPS(0.0, 2)         # variable y

f = exp(x) * sin(y)      # Taylor series of e^x sin(y) through order 6

# Extract the coefficient of x¹y¹ (i.e., ∂²f/∂x∂y|₀ / 1!1!)
coefficient = element(f, [1, 1])
println(coefficient)          # → 1.0
@assert coefficient == 1.0
```

## Documentation

| Page | Description |
|------|-------------|
| [Home](https://msu-beam-dynamics.github.io/PolySeries.jl/dev/) | Overview and quick start |
| [Tutorial](https://msu-beam-dynamics.github.io/PolySeries.jl/dev/tutorial/) | Step-by-step walkthrough of all key features |
| [API Reference](https://msu-beam-dynamics.github.io/PolySeries.jl/dev/api/) | Complete function and type documentation |


## Quick reference

<!-- readme-test -->
```julia
using PolySeries

# Explicit descriptors keep examples independent of task-local state.
desc = PSDesc(2, 4)
x = CTPS(0.0, 1, desc)
y = CTPS(0.0, 2, desc)

# Construction
c = CTPS(3.14, desc)             # scalar constant
z = CTPS(Float64, desc)           # all-zero CTPS

# Allocating arithmetic (returns new CTPS)
f = (1 + x) * (1 + y)
f + x;  f - y;  f * x;  -f;  f^2;  2.0*f

# Math functions
exp(x); log(1 + x); sqrt(1 + x); pow(1 + x, 3)
sin(x); cos(x); tan(x); asin(x/2); acos(x/2)
sinh(x); cosh(x)

# Coefficient access
cst(f)                           # constant term
element(f, [1, 0])               # coefficient of x¹ y⁰
findindex(f, [1, 0])             # integer index of that monomial

# In-place arithmetic (zero allocation)
out = CTPS(Float64, desc)
mul!(out, x, y);  add!(out, x, y);  sub!(out, x, y)
scale!(out, x, 2.0);  scaleadd!(out, 2.0, x, -1.0, y)

# In-place math
sin!(out, x);  cos!(out, x);  exp!(out, x)
log!(out, 1 + x);  sqrt!(out, 1 + x);  pow!(out, 1 + x, 3)
sinh!(out, x); cosh!(out, x)

# Workspace pool
ws = PSWorkspace(desc, 16)
out = borrow!(ws)
mul!(out, x, y)
@assert element(out, [1, 1]) == 1.0
release!(ws, out)

# @tpsa macro — compiles expression into zero-alloc in-place code
θ = 0.2
@tpsa ws out = cos(θ)*x + sin(θ)*(y + x^2)
@assert element(out, [1, 0]) ≈ cos(θ)
```

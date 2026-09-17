# PolySeries.jl

[![CI](https://github.com/MSU-Beam-Dynamics/PolySeries.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/MSU-Beam-Dynamics/PolySeries.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/MSU-Beam-Dynamics/PolySeries.jl/graph/badge.svg)](https://codecov.io/gh/MSU-Beam-Dynamics/PolySeries.jl)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://msu-beam-dynamics.github.io/PolySeries.jl/dev/)

**Truncated Power Series Algebra for Julia**

PolySeries.jl computes multivariate Taylor expansions through a chosen total degree. It overloads arithmetic operators and supported transcendental functions for `CTPS` objects, producing Taylor coefficients with the rounding behavior of the coefficient type.

## Highlights

- **Automatic differentiation to high order** — all partial derivatives up to the chosen order (at most 63) emerge as coefficients of the series. Multiplication tables grow like `binomial(2nv + order, order)/2`, so very high orders are practical for a few variables; `PSDesc` refuses descriptors above a configurable memory limit instead of exhausting memory.
- **[Enzyme.jl compatible](examples/07_enzyme_ad.jl)** — differentiate through TPSA computations to get sensitivities of Taylor coefficients w.r.t. scalar design parameters.
- **[Selected-coefficient parameter gradients](examples/08_parameter_gradients.jl)** — compute a weighted coefficient gradient with 1,000 parameters in one reverse pass.
- **Planned composition** — direct coordinate translations and reusable `CompositionPlan` snapshots for fixed sources evaluated at changing maps.
- **Sparse degree-mask representation** — only active degree blocks are touched; constant-only inputs have near-zero overhead.
- **Lazy-zero allocation** — temporaries use `undef` memory; the `degree_mask` invariant ensures garbage outside the active range is never read.
- **Zero-allocation in-place API** — `mul!` (a method of `LinearAlgebra.mul!`), `add!`, `scaleadd!`, `pow!`, etc., plus `PSWorkspace` for pool-based temporary management.
- **`@tpsa` macro** — compiles supported arithmetic expressions into in-place calls, borrowing workspace slots automatically.
- **Thread-safe** — task-local defaults and separate workspaces documented and tested.

## Installation

```julia
using Pkg
Pkg.add("PolySeries")   # once registered; until then:
# Pkg.add(url="https://github.com/MSU-Beam-Dynamics/PolySeries.jl")
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
temporary = borrow!(ws)
mul!(temporary, x, y)
@assert element(temporary, [1, 1]) == 1.0
release!(ws, temporary)

# @tpsa macro — compiles expression into zero-alloc in-place code
θ = 0.2
@tpsa ws out = cos(θ)*x + sin(θ)*(y + x^2)
@assert element(out, [1, 0]) ≈ cos(θ)
@assert element(out, [0, 1]) ≈ sin(θ)
@assert element(out, [2, 0]) ≈ sin(θ)
@assert ws.sp == length(ws.bufs)
```

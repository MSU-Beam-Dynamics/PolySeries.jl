# PolySeries.jl

**Truncated Power Series Algebra for Julia**

PolySeries.jl computes multivariate Taylor expansions of arbitrary functions to high orders. It overloads all standard arithmetic operators and transcendental functions so that code written for ordinary `Float64` scalars also works for `CTPS` objects (struct of PolySeries) — producing exact Taylor series rather than single numbers.

## Highlights

- **Automatic differentiation to any order** — all partial derivatives up to the chosen order emerge as coefficients of the series.
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
Pkg.add(url="https://github.com/your-org/PolySeries.jl")
```

## Minimal example

```julia
using PolySeries

set_descriptor!(2, 6)    # 2 variables, max order 6

x = CTPS(0.0, 1)         # variable x
y = CTPS(0.0, 2)         # variable y

f = exp(x) * sin(y)      # Taylor series of e^x sin(y) through order 6

# Extract the coefficient of x¹ y² (i.e., ∂³f/∂x ∂y²|₀ / 1! 2!)
println(element(f, [1, 2]))   # → -0.5
```

## Documentation

| Page | Description |
|------|-------------|
| [Home](https://msu-beam-dynamics.github.io/PolySeries.jl/dev/) | Overview and quick start |
| [Tutorial](https://msu-beam-dynamics.github.io/PolySeries.jl/dev/tutorial/) | Step-by-step walkthrough of all key features |
| [API Reference](https://msu-beam-dynamics.github.io/PolySeries.jl/dev/api/) | Complete function and type documentation |


## Quick reference

```julia
# Default descriptor (local to this task)
set_descriptor!(nv, order)
desc = get_descriptor()
clear_descriptor!()

# Explicit descriptor construction (also works without a task default)
desc = PSDesc(2, 4)
x = CTPS(0.0, 1, desc)
# x keeps desc even if the task default changes later

# Construction
x  = CTPS(0.0, 1)       # variable 1 expanded around 0.0
c  = CTPS(3.14)          # scalar constant
z  = CTPS(Float64)       # all-zero CTPS

# Allocating arithmetic (returns new CTPS)
f + g;  f - g;  f * g;  -f;  f^n;  2.0*f

# Math functions
exp(f); log(f); sqrt(f); pow(f, n)
sin(f); cos(f); tan(f); asin(f); acos(f)
sinh(f); cosh(f)

# Coefficient access
cst(f)                           # constant term
element(f, [1, 0])               # coefficient of x¹ y⁰
findindex(f, [1, 0])             # integer index of that monomial

# In-place arithmetic (zero allocation)
mul!(out, a, b);  add!(out, a, b);  sub!(out, a, b)
scale!(out, a, s);  scaleadd!(out, s1, a, s2, b)

# In-place math
sin!(out, f);  cos!(out, f);  exp!(out, f)
log!(out, f);  sqrt!(out, f);  pow!(out, f, n)
sinh!(out, f); cosh!(out, f)

# Workspace pool
ws = PSWorkspace(desc, 16)
t  = borrow!(ws)
# ... use t ...
release!(ws, t)

# @tpsa macro — compiles expression into zero-alloc in-place code
@tpsa ws  nx = cos(θ)*x + sin(θ)*(y + x^2 - z^2)
```

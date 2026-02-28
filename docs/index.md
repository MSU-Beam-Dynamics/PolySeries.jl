# TPSA.jl Documentation

TPSA.jl implements **Truncated Power Series Algebra** — a technique for computing
multivariate Taylor expansions of arbitrary functions to user-specified order.
It overloads Julia's standard arithmetic operators and mathematical functions so
that code written for plain numbers automatically computes exact Taylor series
when given `CTPS` inputs.

## Overview

### What is TPSA?

A `CTPS` (Coefficient-based Taylor Power Series) object represents a function as
a truncated multivariate polynomial:

$$f(x_1, \dotsc, x_n) \approx \sum_{|\alpha| \le d} c_\alpha\, x_1^{\alpha_1} \cdots x_n^{\alpha_n}$$

where $|\alpha| = \alpha_1 + \dotsb + \alpha_n$ and $d$ is the chosen maximum order.

All $\binom{n+d}{d}$ coefficients are stored contiguously, ordered first by total
degree, then lexicographically within each degree.

### What can you do with it?

- **Automatic differentiation to arbitrary order** — partial derivatives of any
  order appear directly as (rescaled) coefficients.
- **Nonlinear map propagation** — push a truncated series through a sequence of
  operations exactly (no finite-difference error).
- **Computing Jacobians, Hessians, and higher-order tensors** without writing
  symbolic formulas.
- **Beam dynamics / perturbation theory** — the original use case; every TPSA
  coefficient encodes a transfer-matrix element or perturbation coefficient.

## Getting started

See the **[Tutorial](tutorial.md)** for a step-by-step walkthrough.

## Installation

```julia
using Pkg
Pkg.add("TPSA")
```

## Basic workflow

```julia
using TPSA

# 1. Register the global descriptor (once per thread / session)
set_descriptor!(3, 6)          # 3 independent variables, max order 6

# 2. Create variables
x = CTPS(0.0, 1)               # x₁, expansion point 0
y = CTPS(0.0, 2)               # x₂
z = CTPS(0.0, 3)               # x₃

# 3. Compute — identical syntax to scalar code
f = exp(x) * sin(y + z^2)

# 4. Inspect coefficients
println(cst(f))                 # constant term f(0,0,0)
println(element(f, [1,0,0]))   # ∂f/∂x|₀
println(element(f, [0,1,0]))   # ∂f/∂y|₀
println(element(f, [1,1,0]))   # ∂²f/(∂x ∂y)|₀
```

## Key types

`CTPS{T}` is the only type end users construct directly.  It holds the coefficient
vector `c::Vector{T}` (length `desc.N`) and tracks which degree blocks are active
via a `degree_mask` bitmask.  Everything else (`TPSADesc`, `TPSAWorkspace`) is
either obtained from helper functions (`get_descriptor()`, `TPSAWorkspace(desc, n)`)
or used only in the advanced zero-allocation API.

## Key functions

### Descriptor management

| Function | Description |
|----------|-------------|
| `set_descriptor!(nv, order)` | Create and register a global descriptor |
| `get_descriptor()` | Retrieve the current thread-local descriptor |
| `clear_descriptor!()` | Remove the current descriptor |

### Construction

| Expression | Result |
|-----------|--------|
| `CTPS(a, i)` | Variable $x_i$ expanded around $a$ |
| `CTPS(a)` | Constant $a$ |
| `CTPS(Float64)` | All-zero series (use as a pre-allocated output slot) |

### Arithmetic operators

All operators create a new `CTPS`:

```
f + g,  f - g,  f * g,  -f,  f^n   (n::Int)
f + a,  a + f,  f - a,  a - f,  a*f,  f*a   (a::Real)
```

### Mathematical functions

| Allocating | In-place | Notes |
|-----------|---------|-------|
| `exp(f)` | `exp!(out, f)` | |
| `log(f)` | `log!(out, f)` | requires `cst(f) > 0` |
| `sqrt(f)` | `sqrt!(out, f)` | requires `cst(f) > 0` |
| `pow(f, n)` | `pow!(out, f, n)` | integer `n` |
| `sin(f)` | `sin!(out, f)` | |
| `cos(f)` | `cos!(out, f)` | |
| `tan(f)` | — | |
| `asin(f)` | — | |
| `acos(f)` | — | |
| `sinh(f)` | `sinh!(out, f)` | |
| `cosh(f)` | `cosh!(out, f)` | |

### In-place arithmetic (zero allocation)

| Function | Effect |
|---------|--------|
| `add!(out, a, b)` | `out = a + b` |
| `add!(out, a, s)` | `out = a + s` (scalar `s`) |
| `sub!(out, a, b)` | `out = a - b` |
| `mul!(out, a, b)` | `out = a * b` |
| `scale!(out, a, s)` | `out = s * a` |
| `scaleadd!(out, s1, a, s2, b)` | `out = s1*a + s2*b` (fused) |
| `addto!(a, b)` | `a += b` |
| `subfrom!(a, b)` | `a -= b` |
| `copy!(dest, src)` | Copy active range of `src` into `dest` |
| `zero!(a)` | Zero all coefficients |

### Coefficient access

| Function | Description |
|---------|-------------|
| `cst(f)` | Constant term (`f.c[1]`) |
| `element(f, exps)` | Coefficient for monomial with exponent vector `exps` |
| `findindex(f, exps)` | Integer index of the monomial (use with `f.c[idx]`) |

`exps` is a `Vector{Int}` of length `nv`; entry `i` is the power of variable $x_i$.

To iterate over all active monomials, use `TPSA.getindexmap(desc.polymap, i)` which
returns a view `[degree, e₁, e₂, …, eₙ]` for coefficient index `i`.

## Zero-allocation patterns

### Workspace pool

```julia
ws = TPSAWorkspace(desc, 16)   # pool of 16 pre-allocated Float64 CTPS

t1 = borrow!(ws)
t2 = borrow!(ws)
mul!(t1, a, a)           # t1 = a²
mul!(t2, b, b)           # t2 = b²
sub!(out, t1, t2)        # out = a² - b²
release!(ws, t1)
release!(ws, t2)
```

`release!` zeros only the active degree range (O(active)), not the full `N`-element
vector, so it is near-free for small active sets.

### `@tpsa` macro

Compiles an arithmetic expression into the equivalent zero-allocation in-place call
sequence.  Workspace slots are borrowed and released automatically.

```julia
@tpsa ws  nx = cos(θ)*x1 + sin(θ)*(x2 + x1^2 - x3^2)
```

Supported operations: `+`, `-`, `*`, unary `-`, `^n` (Int),
`sin`, `cos`, `exp`, `log`, `sqrt`, `sinh`, `cosh`.

## Performance notes

### Coefficient count

$$N = \binom{nv + d}{nv}$$

| `nv` | `order` | $N$ |
|------|---------|-----|
| 2 | 6 | 28 |
| 2 | 10 | 66 |
| 4 | 6 | 210 |
| 4 | 10 | 1 001 |
| 6 | 10 | 8 008 |

Choose the minimum order that captures the physics you care about.

### Allocation cost

Under **lazy-zero** allocation, creating a temporary `CTPS` costs only a `malloc`
(no `memset`). The `degree_mask` bitmask tracks which degree blocks have been
written; reads outside active blocks never occur.

## Enzyme / AD interoperability

TPSA.jl is compatible with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)
(and by extension with Zygote and other source-transformation AD tools that can
differentiate through Julia's heap allocations).

### What this enables

| Level | Tool | What you get |
|-------|------|--------------|
| Phase-space variables | TPSA | Exact Taylor coefficients up to the chosen order |
| Design parameters | Enzyme | First-order sensitivities of any map coefficient |

For example, in beam physics:  TPSA computes the 6-th order transfer map;
Enzyme gives you $\partial c_{ijk}/\partial\theta$ for any lattice parameter $\theta$.

### Minimal example

```julia
using TPSA, Enzyme

# set_descriptor! must be called OUTSIDE the differentiated function.
set_descriptor!(1, 3)         # ← outside

# f(x₀) = exp(x₀)  —  x₀ is both the expansion center and the parameter.
# Analytically: d/dx₀ exp(x₀) = exp(x₀).

function exp_value(x0::Float64)
    t = CTPS(x0, 1)           # expansion around x₀
    return cst(exp(t))         # = exp(x₀)
end

grad = Enzyme.gradient(Reverse, exp_value, 1.0)   # returns (exp(1),) ≈ (2.718,)
```

### Setup

`using TPSA, Enzyme` is all that's needed. The `TPSAEnzymeExt` package
extension is loaded automatically and registers the required `inactive_type`
rules for all TPSA-internal types (`TPSADesc`, `DescPool`, `PolyMap`, etc.) —
no user-side setup required.

### Rules

1. **Call `set_descriptor!` OUTSIDE the differentiated function.** Enzyme does
   not re-execute task-local-storage mutations in its reverse pass.  Set the
   descriptor once before calling `Enzyme.gradient` / `Enzyme.jacobian`.

2. **Use the expansion center as the differentiation parameter:**
   `CTPS(x0, var_index)` where `x0` is the scalar parameter.

3. **Use the allocating forms** inside the differentiated function:
   `exp`, `sin`, `cos`, `log`, `sqrt`, `sinh`, `cosh`, `+`, `-`, `*`, `/`, `^`.

4. **Avoid `!`-variants inside differentiated code.** The in-place functions
   (`exp!`, `sin!`, `mul!`, etc.) write into workspace-pool slots, which
   involves mutation that Enzyme cannot trace through.

See [`examples/07_enzyme_ad.jl`](../examples/07_enzyme_ad.jl) for worked examples
including multi-output Jacobians and finite-difference verification.

## Contents

- [Tutorial](tutorial.md) — step-by-step examples
- [API Reference](api.md) — complete function and type documentation
- [`examples/`](../examples/) — runnable Julia scripts
- [`benchmarks/`](../benchmarks/) — performance measurement scripts

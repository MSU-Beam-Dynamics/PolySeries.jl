# PolySeries.jl Documentation

PolySeries.jl implements **Truncated Power Series Algebra** — a technique for computing
multivariate Taylor expansions of arbitrary functions to user-specified order.
It overloads Julia's arithmetic operators and the mathematical functions listed
below for `CTPS` objects, so code written against that operator set computes
truncated Taylor series when given `CTPS` inputs (a `CTPS` is not a `Number`,
so generic code that relies on `zero(x)`, comparisons or `convert` is not
supported). Coefficients retain the rounding behavior of their numeric type.

## Overview

### What is TPSA?

A `CTPS` (Coefficient-based Taylor Power Series) object represents a function as
a truncated multivariate polynomial:

$$f(x_1, \dotsc, x_n) \approx \sum_{|\alpha| \le d} c_\alpha\, x_1^{\alpha_1} \cdots x_n^{\alpha_n}$$

where $|\alpha| = \alpha_1 + \dotsb + \alpha_n$ and $d$ is the chosen maximum order.

All $\binom{n+d}{d}$ coefficients are stored contiguously, ordered first by total
degree, then lexicographically within each degree.

### What can you do with it?

- **Automatic differentiation through order 63** — partial derivatives through
  the chosen descriptor order appear directly as rescaled coefficients.
- **Nonlinear map propagation** — push a truncated series through a sequence of
  operations without finite-difference approximations.
- **Computing Jacobians, Hessians, and higher-order tensors** without writing
  symbolic formulas.
- **Beam dynamics / perturbation theory** — the original use case; every TPSA
  coefficient encodes a transfer-matrix element or perturbation coefficient.

## Getting started

See the **[Tutorial](tutorial.md)** for a step-by-step walkthrough.

## Installation

```julia
using Pkg
Pkg.add("PolySeries")   # once registered; until then:
# Pkg.add(url="https://github.com/MSU-Beam-Dynamics/PolySeries.jl")
```

## Basic workflow

```julia
using PolySeries

# 1. Register this task's default descriptor
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

`CTPS{T}` stores coefficients and an active-degree mask. Construct a
`PSDesc(nv, order)` explicitly to specify its polynomial space, or use a
task-local default. `PSWorkspace` and `CompositionWorkspace` provide reusable
temporary storage. A polynomial retains its construction descriptor.

## Key functions

### Descriptor management

| Function | Description |
|----------|-------------|
| `set_descriptor!(nv, order)` | Create and register this task's default descriptor |
| `get_descriptor()` | Retrieve the current task-local descriptor |
| `clear_descriptor!()` | Remove the current descriptor |

### Construction

| Expression | Result |
|-----------|--------|
| `CTPS(a, i)` | Variable $x_i$ expanded around $a$ |
| `CTPS(a)` | Constant $a$ |
| `CTPS(Float64)` | All-zero series (use as a pre-allocated output slot) |

### Arithmetic operators

All operators create a new `CTPS`:

```text
f + g,  f - g,  f * g,  f / g,  -f,  f^n   (n::Int)
f + a,  a + f,  f - a,  a - f,  a*f,  f*a,  f/a,  a/f   (a::Number, converted to the coefficient type)
```

Division and `inv` require a nonzero constant term in the divisor
(`DomainError` otherwise). Coefficient types are floating point (`Float64`,
`Float32`, `BigFloat`, and their `Complex` forms); mixing two coefficient types
in one expression is not supported.

### Mathematical functions

| Allocating | In-place | Notes |
|-----------|---------|-------|
| `exp(f)` | `exp!(out, f)` | |
| `log(f)` | `log!(out, f)` | positive real or nonzero complex constant |
| `sqrt(f)` | `sqrt!(out, f)` | positive real or nonzero complex constant |
| `pow(f, n)` | `pow!(out, f, n)` | integer `n`; in-place form requires `n ≥ 0` |
| `sin(f)` | `sin!(out, f)` | |
| `cos(f)` | `cos!(out, f)` | |
| `tan(f)` | — | |
| `asin(f)` | `asin!(out, f)` | real constant must have absolute value below 1; complex branch points ±1 are rejected |
| `acos(f)` | `acos!(out, f)` | real constant must have absolute value below 1; complex branch points ±1 are rejected |
| `sinh(f)` | `sinh!(out, f)` | |
| `cosh(f)` | `cosh!(out, f)` | |

### In-place arithmetic (zero allocation)

| Function | Effect |
|---------|--------|
| `add!(out, a, b)` | `out = a + b` |
| `add!(out, a, s)` | `out = a + s` (scalar `s` of the coefficient type `T`) |
| `sub!(out, a, b)` | `out = a - b` |
| `mul!(out, a, b)` | `out = a * b` |
| `scale!(out, a, s)` | `out = s * a` (scalar `s` of the coefficient type `T`) |
| `scaleadd!(out, s1, a, s2, b)` | `out = s1*a + s2*b` (fused) |
| `addto!(a, b)` | `a += b` |
| `subfrom!(a, b)` | `a -= b` |
| `copy!(dest, src)` | Copy active range of `src` into `dest` |
| `zero!(a)` | Reset to the zero polynomial and clear its activity mask |

### Coefficient access

| Function | Description |
|---------|-------------|
| `cst(f)` | Safely read the constant coefficient |
| `element(f, exps)` | Safely read the coefficient for exponent vector `exps` |
| `findindex(f, exps)` | Return the internal storage index of a monomial |

`exps` is a `Vector{Int}` with one nonnegative power per variable. A leading
total degree is also accepted when it matches their sum. The degree must not
exceed the descriptor order; malformed vectors raise `ArgumentError`.
The raw `f.c` buffer is lazily initialized, so inactive degree blocks must be
read through `cst` or `element` rather than indexed directly.

To iterate over all monomials, `decomposite(i - 1, desc.nv)` returns the
exponent vector `[degree, e₁, e₂, …, eₙ]` for storage index `i` (`1 ≤ i ≤ desc.N`);
pass its tail to `element` to read the coefficient safely.

## Zero-allocation patterns

### Workspace pool

```julia
ws = PSWorkspace(desc, 16)   # pool of 16 pre-allocated Float64 CTPS

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

Under lazy initialization, constructors initialize the coefficient blocks they
activate rather than necessarily clearing the full buffer. Public accessors
return zero for inactive blocks. Allocation costs depend on the constructor,
coefficient type, and whether Enzyme is differentiating the call.

## Enzyme / AD interoperability

PolySeries.jl is compatible with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl).

### What this enables

| Level | Tool | What you get |
|-------|------|--------------|
| Phase-space variables | TPSA | Exact Taylor coefficients up to the chosen order |
| Design parameters | Enzyme | First-order sensitivities of any map coefficient |

For example, in beam physics:  TPSA computes the 6-th order transfer map;
Enzyme gives you $\partial c_{ijk}/\partial\theta$ for any lattice parameter $\theta$.

### Minimal example

```julia
using PolySeries, Enzyme

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

`using PolySeries, Enzyme` is all that's needed. The `PolySeriesEnzymeExt` package
extension is loaded automatically and registers the required `inactive_type`
rules for all TPSA-internal types (`PSDesc`, `DescPool`, `PolyMap`, etc.) —
no user-side setup required.

### Rules

1. **Call `set_descriptor!` OUTSIDE the differentiated function.** Enzyme does
   not re-execute task-local-storage mutations in its reverse pass.  Set the
   descriptor once before calling `Enzyme.gradient` / `Enzyme.jacobian`.

2. **Use the expansion center as the differentiation parameter:**
   `CTPS(x0, var_index)` where `x0` is the scalar parameter.

3. **Use the allocating forms** inside the differentiated function:
   `exp`, `sin`, `cos`, `log`, `sqrt`, `sinh`, `cosh`, `+`, `-`, `*`, `/`, `^`.

4. **Keep differentiable buffers local to the differentiated call.**
   In-place operations are not categorically unsupported: tests cover
   forward and reverse differentiation through aliased `exp!`, `sin!`,
   `cos!`, and nonnegative `pow!` calls. Internal math temporaries use
   independent allocations during AD. These checks do not establish support
   for arbitrary mutation of a shared `PSWorkspace`.

Prebuilt polynomials may have zero-valued coefficients with nonzero derivatives.
During AD, arithmetic materializes sparse inputs through mask-aware coefficient
reads so those derivatives are retained without reading inactive storage.
This can increase memory and work compared with ordinary sparse arithmetic.
When only the evaluation point varies, mark the polynomial as fixed:
`Enzyme.gradient(Reverse, Enzyme.Const(f), x0)`.

See `examples/07_enzyme_ad.jl` in the package directory for worked examples
including comparisons with analytic derivatives.

## Contents

- [Tutorial](tutorial.md) — step-by-step examples
- [API Reference](api.md) — complete function and type documentation
- `examples/` — runnable Julia scripts
- `benchmarks/` — performance measurement scripts

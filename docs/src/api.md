# API Reference

Complete reference for all public types and functions in PolySeries.jl.

---

## Types

### `CTPS{T}`

The core type representing a truncated power series with coefficient type `T`.

```julia
struct CTPS{T}
    c           :: Vector{T}       # coefficient vector, length desc.N
    descriptor_id :: Int         # stable descriptor identity
    degree_mask :: Ref{UInt64}     # bitmask — bit k set iff degree-k block is active
end
```

The `p.desc` property resolves the descriptor captured by `p` at construction.
It does not consult the current task default. The numeric handle keeps constant
metadata separate from differentiable coefficient storage in Enzyme.

The coefficient vector is ordered by total degree, then lexicographically:
index 1 is the constant term; indices 2…nv+1 are the degree-1 (linear) terms; and so on.

**Constructors:**

```julia
CTPS(a::Real, var_index::Int)   # variable x_{var_index} expanded around a
CTPS(a::Real)                   # scalar constant a
CTPS(T::Type)                   # all-zero series (use as pre-allocated output)
```

The constructors above use the descriptor registered with `set_descriptor!`.

Alternatively, pass a descriptor explicitly without changing the task default:

```julia
desc = PSDesc(2, 4)
x = CTPS(0.0, 1, desc)
c = CTPS(3.0, desc)
out = CTPS(Float64, desc)
```

Each polynomial retains its descriptor. Changing or clearing a task's default
does not change existing polynomials. Operations on incompatible descriptors
throw `DimensionMismatch` before mutating coefficient buffers.

---

### `PSDesc`

Descriptor holding shared metadata for a TPSA space `(nv, order)`.

```julia
mutable struct PSDesc
    const id        :: Int             # stable context identity
    const nv        :: Int             # number of variables
    const order     :: Int             # maximum total degree
    const N         :: Int             # total coefficient count = C(nv+order, nv)
    const Nd        :: Vector{Int}     # number of monomials per degree
    const off       :: Vector{Int}     # start index per degree
    const polymap   :: PolyMap         # index → exponent mapping
    const exp_to_idx :: Dict           # exponent → index
    const mul       :: ...             # multiplication schedules
    const comp_plan :: ...             # composition plan
end
```

Obtain via `get_descriptor()` after calling `set_descriptor!`.
`PSDesc(nv, order)` also returns a cached descriptor without setting a default.

Supported orders are **0–63**, with **1–127 variables** and at most
`typemax(Int32)` coefficients. Invalid sizes throw `ArgumentError` before table
allocation and leave the current default unchanged. These are representation
limits; practical sizes also depend on available memory. Transcendental series
coefficients are computed in the coefficient type, including `BigFloat`, without
machine-integer factorials.
Its fields cannot be reassigned; treat the shared index tables as read-only.

**Useful fields:**
- `desc.N` — total number of coefficients per series
- `desc.nv`, `desc.order` — space parameters
- `desc.off[d+1]` — 1-based start of degree-`d` block in coefficient vector
- `PolySeries.getindexmap(desc.polymap, i)` — exponent info for coefficient index `i`

---

### `PSWorkspace`

Pre-allocated pool of `CTPS{Float64}` objects for zero-allocation in-place code.

```julia
ws = PSWorkspace(desc::PSDesc, n::Int = 32)
```

Creates a pool of `n` CTPS slots of the same shape as `desc`.

---

### `PolyMap`

Internal mapping structure (accessed via `PSDesc`).

```julia
struct PolyMap
    dim       :: Int             # nv
    max_order :: Int             # order
    map       :: Matrix{UInt8}   # map[i, 1] = total degree; map[i, v+1] = exp of var v
end
```

---

## Descriptor management

```julia
set_descriptor!(nv::Int, order::Int)
```
Create a descriptor for a space with `nv` variables and maximum degree `order`,
and register it as the task-local default.

```julia
desc = get_descriptor()
```
Return the current task-local descriptor.  Throws if none is registered.

```julia
clear_descriptor!()
```
Deregister the task-local descriptor.

---

## Construction

```julia
CTPS(a::Real, var_index::Int)
```
Create the identity map variable $x_i$ at expansion point $a$:
`c[1] = a`, `c[var_index + 1] = 1.0`, all other coefficients zero.

```julia
CTPS(a::Real)
```
Create a CTPS whose only non-zero coefficient is the constant term:
`c[1] = a`.

```julia
CTPS(T::Type)
```
Allocate an all-zero CTPS with coefficient type `T`.
Equivalent to a pre-allocated output slot.

```julia
CTPS(src::CTPS{T})
```
Deep copy of `src` (copies only the active degree range).

---

## Arithmetic operators

All operators below return a new `CTPS`; inputs are not modified.

```julia
+(f, g), -(f, g), *(f, g)    # CTPS × CTPS
+(f, a), -(f, a), *(f, a)    # CTPS × scalar (a::Real)
+(a, f), -(a, f), *(a, f)    # scalar × CTPS
-(f)                          # unary negation
^(f, n::Int)                  # integer power (binary exponentiation)
```

---

## Composition

```julia
h = compose(f, g)
compose!(out, f, g)
workspace = CompositionWorkspace(f.desc, Float64)
compose!(out, f, g, workspace)
```

These compute `f(g[1], …, g[nv])`, truncated to the descriptor order.
All polynomials and the workspace must have the same descriptor and coefficient
type. The output must not alias `f` or any member of `g`. A workspace can be
reused across different polynomials and substitution maps, but must not be
shared by concurrent calls.

Ordinary composition visits the parent-monomial tree depth first and retains
only one image per depth. It needs `(order + 1) * N` coefficient slots plus
O(N) metadata, and skips branches that do not contribute to nonzero source
coefficients. Inactive coefficient storage is never read. For `Float64`, the
workspace overload allocates zero bytes after setup; types such as `BigFloat`
still allocate during scalar arithmetic.

Enzyme differentiation automatically uses a separate retained-image path.
That path preserves derivative dependencies at zero-valued coefficients and
still has O(N²) worst-case coefficient storage. The ordinary workspace memory
bound does **not** apply to reverse differentiation. Reuse changes the order of
floating-point accumulation, so final rounding may differ from the retained
path.

---

## Mathematical functions

### Allocating (return new `CTPS`)

| Function | Mathematical meaning |
|---------|---------------------|
| `exp(f)` | $e^f$ |
| `log(f)` | $\ln f$ — requires `cst(f) > 0` |
| `sqrt(f)` | $\sqrt{f}$ — requires `cst(f) > 0` |
| `pow(f, n::Int)` | $f^n$ — equivalent to `f^n` |
| `sin(f)` | $\sin f$ |
| `cos(f)` | $\cos f$ |
| `tan(f)` | $\tan f$ |
| `asin(f)` | $\arcsin f$ |
| `acos(f)` | $\arccos f$ |
| `sinh(f)` | $\sinh f$ |
| `cosh(f)` | $\cosh f$ |

### In-place (`!` variants — write into pre-allocated `out::CTPS`)

```julia
exp!(out, f);   log!(out, f);   sqrt!(out, f);   pow!(out, f, n::Int)
sin!(out, f);   cos!(out, f)
sinh!(out, f);  cosh!(out, f)
```

The unary in-place functions support using the same polynomial as input and
output, for example `exp!(p, p)` or `sin!(p, p)`. This also applies to `log!`,
`sqrt!`, `cos!`, `sinh!`, `cosh!`, `asin!`, `acos!`, and `pow!` with nonnegative
integer exponents. Input coefficients are preserved before the output is
cleared. Ordinary Float64 calls reuse internal pool storage when slots are
available; Enzyme differentiation uses independently allocated temporaries.

The `!` variants use `_zero_active!` internally: only the active degree range is
zeroed before recomputing, giving O(active) cost instead of O(N).

---

## In-place arithmetic

The following functions write their result into a **pre-allocated** first argument
and return `nothing`.  No heap allocation occurs.

```julia
add!(out::CTPS, a::CTPS, b::CTPS)        # out = a + b
add!(out::CTPS, a::CTPS, s::T)           # out = a + s  (scalar s)
sub!(out::CTPS, a::CTPS, b::CTPS)        # out = a - b
mul!(out::CTPS, a::CTPS, b::CTPS)        # out = a * b
scale!(out::CTPS, a::CTPS, s::T)         # out = s * a
scaleadd!(out, s1, a, s2, b)             # out = s1*a + s2*b  (fused, single pass)
addto!(a::CTPS, b::CTPS)                 # a += b  (modifies a)
subfrom!(a::CTPS, b::CTPS)              # a -= b  (modifies a)
copy!(dest::CTPS, src::CTPS)             # dest ← src  (active range only)
zero!(a::CTPS)                           # zero all N coefficients
```

`scaleadd!` is the preferred primitive for linear combinations; it avoids a
temporary and processes both inputs in a single pass.

---

## Workspace pool

```julia
ws = PSWorkspace(desc, n)    # create pool of n Float64 CTPS slots
t  = borrow!(ws)               # get a zeroed CTPS from the pool
release!(ws, t)                # return t to pool (active range zeroed)
```

`borrow!` pops from a stack of pre-allocated objects.  `release!` calls
`_zero_active!` on the returned slot (O(active) zeroing) and pushes it back.

**The pool is not thread-safe.** Create one workspace per concurrent task.

---

## `@tpsa` macro

```julia
@tpsa ws  lhs = expr
```

Compiles `expr` into a sequence of zero-allocation in-place calls, writing the
final result into the pre-allocated `CTPS` object `lhs`.  Temporaries are
borrowed from `ws::PSWorkspace` and released as soon as they are no longer
needed, including when evaluation throws. Slots borrowed by the caller remain
untouched; the output may be partially written on failure.

**Supported operations in `expr`:**
`+`, `-`, `*`, unary `-`, `^n` (integer `n`),
`sin`, `cos`, `exp`, `log`, `sqrt`, `sinh`, `cosh`.
Scalars (`Real`) may appear as either operand to `+`, `-`, `*`, and as arguments to the supported unary functions.

**Restrictions:**
- `lhs` must be a pre-allocated `CTPS` object.
- `lhs` must not appear on the right-hand side (no self-referential expressions).
- The workspace `ws` must not be shared across threads.

---

## Coefficient access

```julia
cst(f::CTPS) -> T
```
Return the constant term `f.c[1]`.

```julia
element(f::CTPS, exps::Vector{Int}) -> T
```
Return the coefficient of the monomial $x_1^{e_1} \cdots x_{nv}^{e_{nv}}$
where `exps = [e_1, …, e_nv]` (length `nv`, no degree prefix).

```julia
findindex(f::CTPS, exps::Vector{Int}) -> Int
```
Return the 1-based index into `f.c` for the monomial described by `exps`.
The index can be reused with `f.c[idx]` directly.

```julia
PolySeries.getindexmap(desc.polymap, i::Int) -> view
```
Return a view `[degree, e₁, e₂, …, e_nv]` for coefficient index `i`.
Useful for iterating over all monomials:

```julia
for i in 1:desc.N
    v = f.c[i]; iszero(v) && continue
    row = PolySeries.getindexmap(desc.polymap, i)
    # row[1] = total degree; row[k+1] = exponent of variable k
end
```

---

## Assign / reassign

```julia
assign!(ctps::CTPS, a::Real)
```
Set the constant term to `a` without touching other coefficients.

```julia
reassign!(ctps::CTPS, a::Real, var_index::Int)
```
Reset `ctps` to the variable $x_{var\_index}$ at expansion point `a`
(fills the full coefficient vector with zeros, then sets `c[1] = a`, `c[var_index+1] = 1`).

---

## Descriptor utilities

```julia
PolySeries.decomposite(n::Int, dim::Int) -> Vector{Int}
```
Decompose coefficient index `n` (0-based) into an exponent vector of length `dim+1`:
entry 1 is the total degree, entries 2…dim+1 are per-variable exponents.

---

## See also

- [Tutorial](tutorial.md) — worked examples for all features
- [Home](index.md) — overview, performance notes, and Enzyme interoperability guide
- `examples/` — complete runnable scripts
- `examples/07_enzyme_ad.jl` — nested AD with Enzyme
- `benchmarks/` — performance measurement scripts

---

## Enzyme / AD interoperability

PolySeries.jl is compatible with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl).
Enzyme can differentiate *through* TPSA computations, giving exact
first-order sensitivities of any Taylor coefficient with respect to scalar
"design" parameters.

### Coefficient access on an existing polynomial

`cst` and `element` have explicit differentiation rules: a zero-valued primal
coefficient can still have a nonzero tangent. These reads continue to respect
the primal storage mask, including when inactive storage is uninitialized.

```julia
using PolySeries, Enzyme
set_descriptor!(1, 3)
p = CTPS(0.0, 1)                 # Constructed before entering Enzyme
dp = Enzyme.make_zero(p)         # Independent, fully initialized shadow
dp.c[1] = 1.0                   # Unit constant-coefficient tangent
Enzyme.autodiff(Forward, cst, Duplicated(p, dp)) # (1.0,)
```

Enzyme coefficient shadows are dense: initialize every tangent coefficient,
even when its corresponding primal degree is inactive. `Enzyme.make_zero(p)`
provides initialized zero storage and an independent full degree mask, so
returned gradients expose contributions at degrees absent from the primal.

### Setup

`using PolySeries, Enzyme` is sufficient. The `PolySeriesEnzymeExt` package extension
loads automatically and registers `inactive_type` rules for all TPSA-internal
types — no call to `EnzymeRules.inactive_type` is needed in user code.

### Compatible operations

All allocating forms are Enzyme-compatible:

```julia
# ✓ Arithmetic operators
f + g,  f - g,  f * g,  f / g,  -f,  f^n
f + a,  a + f,  f - a,  a*f  ...

# ✓ Math functions (allocating)
exp(f), log(f), sqrt(f), pow(f, n)
sin(f), cos(f), tan(f), asin(f), acos(f)
sinh(f), cosh(f)
```

### Incompatible operations (do not differentiate through)

```julia
# ✗ In-place / pool-based variants — mutate shared workspace slots
exp!(out, f);  sin!(out, f);  mul!(out, a, b) ...
```

Use the allocating forms instead when building a function intended for
Enzyme differentiation.

### Usage pattern

```julia
using PolySeries, Enzyme

# set_descriptor! must be called OUTSIDE the differentiated function.
set_descriptor!(1, 4)          # ← outside

# Use the expansion center as the parameter.
function map_coeff(x0::Float64)
    t  = CTPS(x0, 1)           # x0 is the parameter
    return element(exp(t), [3])  # 3rd Taylor coefficient = exp(x0)/6
end

# First derivative w.r.t. x0
grad = Enzyme.gradient(Reverse, map_coeff, 0.3)   # (exp(0.3)/6,)

# Multi-variable: set descriptor outside, use expansion center as parameter
set_descriptor!(2, 3)
function f_mv(x0::Float64)
    x = CTPS(x0, 1);  y = CTPS(0.5, 2)   # y expansion center fixed
    return cst(sin(x) * cos(y) + exp(x))
end
Enzyme.gradient(Reverse, f_mv, 0.7)
```

See `examples/07_enzyme_ad.jl` in the package directory for a complete
worked example with finite-difference verification.

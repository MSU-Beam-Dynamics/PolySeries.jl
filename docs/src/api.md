# API Reference

This page is generated from the public docstrings in PolySeries.jl. The raw
`CTPS.c` vector is internal, lazily initialized storage. Read coefficients with
[`cst`](@ref) and [`element`](@ref); use [`findindex`](@ref) only when a storage
index is required.

```@docs
PolySeries
```

## Types and construction

```@docs
CTPS
PSDesc
PSWorkspace
CompositionWorkspace
CompositionPlan
```

## Descriptor management

```@docs
set_descriptor!
get_descriptor
clear_descriptor!
```

## Coefficient access

```@docs
cst
element
findindex
```

The usual form contains one nonnegative exponent per variable. For example,
`element(p, [2, 1])` reads the coefficient of ``x_1^2x_2``. A leading total
degree is also accepted when it equals the sum of the exponents. Invalid lengths,
negative exponents, inconsistent prefixes, and degrees beyond the descriptor
order raise `ArgumentError`. An inactive degree block represents numerical zeros
even though its backing memory need not be initialized.

## Arithmetic

The standard operators `+`, `-`, `*`, `/`, and integer `^` allocate their
result. The following functions write to existing storage:

```@docs
add!
addto!
sub!
subfrom!
scale!
scaleadd!
copy!
zero!
mul!(::CTPS, ::CTPS, ::CTPS)
pow
pow!
```

`mul!` is a method of `LinearAlgebra.mul!`, so it can be used alongside
`LinearAlgebra` without qualification. In-place scalar arguments (`add!`,
`scale!`, `scaleadd!`) must have the coefficient type `T` exactly; the
allocating operators accept any `Number` and convert.

## Errors

`DomainError` signals an expansion center where the function is singular:
division by a series with zero constant term, `log` at zero, real `sqrt` of a
negative constant, and real `asin`/`acos` with `|constant| ≥ 1`.
`ArgumentError` signals invalid arguments (exponent vectors, variable indices,
negative in-place powers, descriptor sizes beyond the limit, releasing a
workspace slot twice, a composition result that aliases its inputs).
`DimensionMismatch` signals operands built on different descriptors or a
substitution map with the wrong number of polynomials.

## Mathematical functions

The allocating forms are `exp`, `log`, `sqrt`, `sin`, `cos`, `tan`, `asin`,
`acos`, `sinh`, and `cosh`. Their in-place counterparts are:

`sqrt` and `sqrt!` require a nonzero constant coefficient (positive for real
coefficients). Expansion about zero is unsupported and raises `DomainError`.

`asin`/`acos` and their in-place forms reject complex branch points at `±1`.
For other complex constants on a branch cut, the sign of the imaginary zero
selects the local analytic continuation matching Julia's scalar function.
Evaluating that continuation across the cut need not match the scalar function
on its opposite side.

```@docs
exp!
log!
sqrt!
inv!
div!
sin!
cos!
sincos!
tan!
asin!
acos!
sinh!
cosh!
```

Every elementary function is evaluated by a degree-block recurrence — one block
convolution per function (`tan`, `asin` and `acos` use three) rather than one
series product per order — and produces a result whose active degrees are
exactly those reachable from the active degrees of the argument. `sincos!`
computes both trigonometric series in one pass for the cost of one; `div!`
divides two series with a single recurrence and no intermediate inverse.

## Composition

```@docs
compose
compose!
```

Ordinary composition can reuse a [`CompositionWorkspace`](@ref). The workspace
keeps one monomial image per tree depth and prunes source branches whose
coefficients are zero. Enzyme uses a separate retained-image implementation so
reverse differentiation preserves dependencies through zero-valued
coefficients.

Coordinate translations `g[i] = x[i] + shift[i]` use a direct coefficient
translation strategy in ordinary execution. Identity substitutions preserve
the source's degree mask exactly. Other translations may conservatively mark
zero blocks active, without changing their mathematical coefficients.

For a fixed source evaluated at many maps, [`CompositionPlan`](@ref) snapshots
the source and caches its traversal. Reuse a workspace with
`compose!(out, plan, g, workspace)`; changing the original source does not
change the plan. Construct a new plan to use updated source coefficients.

## Temporary workspaces and expression lowering

```@docs
borrow!
release!
@tpsa
```

A `PSWorkspace` belongs to one caller at a time. The `@tpsa` macro releases all
temporaries it borrows, including when evaluation throws. Its coefficient type
must match the polynomials: use `PSWorkspace(desc, 16, ComplexF64)` for complex
coefficients or `PSWorkspace(desc, 16, Float32)` for single precision. Omitting
the type preserves the `Float64` default. Borrowing a slot does not allocate;
elementary functions may still allocate internal scratch buffers for types
other than `Float64`.

## Index decomposition

```@docs
decomposite
```

## Enzyme interoperability

Load `Enzyme` beside `PolySeries` to activate the package extension:

```julia
using PolySeries, Enzyme

set_descriptor!(1, 3)
f(a) = cst(exp(CTPS(a, 1)))
Enzyme.gradient(Forward, f, 0.0) # (1.0,)
```

Set the task's default descriptor before entering differentiated code, or pass
an explicit descriptor to every constructor. Existing polynomials retain their
descriptor if the task default changes. `cst` and `element` preserve tangent
activity at zero-valued coefficients, including for polynomials constructed
before differentiation.

Use `Enzyme.make_zero(p)` when constructing a tangent for an existing `CTPS`.
It creates independent, initialized coefficient storage and a full tangent
degree mask.

Real `Float32`/`Float64` polynomial multiplication uses explicit forward and
reverse coefficient-convolution rules. The reverse rule applies the transpose
of truncated convolution, retaining adjoints for zero-valued coefficients and
saving operands when later mutation requires it. Complex multiplication keeps
the existing Enzyme path. Real `exp`/`exp!` also use a mathematical rule:
their forward derivative multiplies by the primal exponential, and their
reverse derivative applies the corresponding coefficient correlation.
Ordinary elementary-function evaluation retains its graded recurrences.

When a differentiated closure captures a fixed polynomial or composition plan,
annotate the closure with `Enzyme.Const`, for example
`Enzyme.gradient(Reverse, Enzyme.Const(loss), parameters)`. Only do this when
the captured values are fixed, rather than differentiation parameters.

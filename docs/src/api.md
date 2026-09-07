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

Exponent vectors contain one exponent per variable. For example,
`element(p, [2, 1])` reads the coefficient of ``x_1^2x_2``. An inactive degree
block represents numerical zeros even though its backing memory need not be
initialized.

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
mul!
pow
pow!
```

## Mathematical functions

The allocating forms are `exp`, `log`, `sqrt`, `sin`, `cos`, `tan`, `asin`,
`acos`, `sinh`, and `cosh`. Their in-place counterparts are:

```@docs
exp!
log!
sqrt!
sin!
cos!
asin!
acos!
sinh!
cosh!
```

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

## Temporary workspaces and expression lowering

```@docs
borrow!
release!
@tpsa
```

A `PSWorkspace` belongs to one caller at a time. The `@tpsa` macro releases all
temporaries it borrows, including when evaluation throws.

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

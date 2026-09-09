# Public API docstrings kept together so the reference manual can render every
# exported binding with Documenter's @docs blocks.

@doc """
    PSDesc(nv::Int, order::Int)

Return the cached descriptor for polynomials in `nv` variables through total
degree `order`. Supported orders are 0–63. A polynomial retains the descriptor
with which it was constructed.
""" PSDesc

@doc """
    PSWorkspace(desc::PSDesc, n::Int=32)

Create a pool of `n` reusable `CTPS{Float64}` temporaries for `desc`. A
workspace must not be shared by concurrent calls.
""" PSWorkspace

@doc """
    cst(p::CTPS)

Return the constant coefficient of `p`. This accessor respects lazy inactive
degree storage and returns numerical zero when the constant block is inactive.
""" cst

@doc """
    element(p::CTPS, exponents::Vector{Int})

Return the coefficient of the monomial described by `exponents`. Pass one
nonnegative exponent per variable, or prefix those exponents with their total
degree. The total degree must not exceed the descriptor order. This is the safe
public coefficient accessor; inactive degree blocks return numerical zero.
""" element

@doc """
    findindex(p::CTPS, exponents::Vector{Int}) -> Int

Return the storage index for a monomial. `exponents` accepts the same validated
formats as [`element`](@ref). Use `element` to read the coefficient because
inactive entries in `p.c` may be uninitialized.
""" findindex

@doc """
    add!(out, a, b)

Write `a + b` to `out`, where `b` may be a compatible `CTPS` or scalar.
""" add!

@doc """
    addto!(a::CTPS, b::CTPS)

Accumulate `b` into `a` in place.
""" addto!

@doc """
    sub!(out::CTPS, a::CTPS, b::CTPS)

Write `a - b` to `out`.
""" sub!

@doc """
    subfrom!(a::CTPS, b::CTPS)

Subtract `b` from `a` in place.
""" subfrom!

@doc """
    scale!(p::CTPS, factor)
    scale!(out::CTPS, p::CTPS, factor)

Scale a polynomial in place, or write a scaled polynomial to `out`.
""" scale!

@doc """
    scaleadd!(out, a, p, b, q)

Write the fused linear combination `a*p + b*q` to `out`.
""" scaleadd!

@doc """
    copy!(out::CTPS, p::CTPS)

Copy the active coefficients and mask from `p` to compatible `out`.
""" copy!

@doc """
    zero!(p::CTPS)

Set `p` to the zero polynomial by clearing its active coefficient blocks.
""" zero!

@doc """
    mul!(out::CTPS, a::CTPS, b::CTPS)

Write the truncated product `a*b` to `out`.
The output may alias either or both inputs; aliasing uses temporary storage.
""" mul!

@doc """
    compose!(out, f, substitutions)
    compose!(out, f, substitutions, workspace)

Write `f(substitutions...)` to `out`. The output must not alias `f` or a
substitution. Pass a [`CompositionWorkspace`](@ref) to reuse scratch storage.
""" compose!

@doc """
    pow(p::CTPS, exponent::Int)

Return `p` raised to an integer power. Negative powers exponentiate the
reciprocal, which requires a nonzero constant coefficient.
""" pow

@doc """
    pow!(out::CTPS, p::CTPS, exponent::Int)

Write `p^exponent` to `out` for a nonnegative integer exponent. Aliasing
`out === p` is supported.
""" pow!

@doc """
    exp!(out::CTPS, p::CTPS)

Write the exponential series of `p` to `out`. Input/output aliasing is supported.
""" exp!

@doc """
    log!(out::CTPS, p::CTPS)

Write the natural logarithm series of `p` to `out`. Input/output aliasing is
supported.
""" log!

@doc """
    sqrt!(out::CTPS, p::CTPS)

Write the square-root series of `p` to `out`. Input/output aliasing is supported.
The constant coefficient must be nonzero (positive for real coefficients);
expansion about zero is unsupported and throws `DomainError`.
""" sqrt!

@doc """
    sin!(out::CTPS, p::CTPS)

Write the sine series of `p` to `out`. Input/output aliasing is supported.
""" sin!

@doc """
    cos!(out::CTPS, p::CTPS)

Write the cosine series of `p` to `out`. Input/output aliasing is supported.
""" cos!

@doc """
    asin!(out::CTPS, p::CTPS)

Write the inverse-sine series of `p` to `out`. Input/output aliasing is supported.
Complex centers at ±1 raise `DomainError`. On complex branch cuts, signed
imaginary zero selects the continuation matching the scalar `asin` value.
""" asin!

@doc """
    acos!(out::CTPS, p::CTPS)

Write the inverse-cosine series of `p` to `out`. Input/output aliasing is supported.
Complex centers at ±1 raise `DomainError`. On complex branch cuts, signed
imaginary zero selects the continuation matching the scalar `acos` value.
""" acos!

@doc """
    sinh!(out::CTPS, p::CTPS)

Write the hyperbolic-sine series of `p` to `out`. Input/output aliasing is
supported.
""" sinh!

@doc """
    cosh!(out::CTPS, p::CTPS)

Write the hyperbolic-cosine series of `p` to `out`. Input/output aliasing is
supported.
""" cosh!

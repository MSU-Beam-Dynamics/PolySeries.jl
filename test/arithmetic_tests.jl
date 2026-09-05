# Accuracy tests for all TPSA arithmetic operations
#
# Tests cover:
#   - CTPS constructors (constant, variable, zero)
#   - element() / findindex() / cst() public accessors
#   - +, -, unary-, * (all overloads inc. scalar), /
#   - pow / ^
#   - In-place: add!, addto!, sub!, subfrom!, scale!, scaleadd!, copy!, zero!
#   - Order truncation
#   - Complex arithmetic
#   - Dense multiplication with exact expected coefficients

# ─── helpers ─────────────────────────────────────────────────────────────────

# Coefficient of x1^e1 * x2^e2 using the public element() API
_coeff2(c::CTPS, e1, e2) = element(c, [e1 + e2, e1, e2])

# Set coefficient of monomial [deg, e1, e2] directly (no public assign!-by-index)
function _setcoeff!(c::CTPS, e1, e2, val)
    idx = findindex(c, [e1 + e2, e1, e2])
    c.c[idx] = val
end

# ─── CTPS constructor accuracy ───────────────────────────────────────────────

@testset "Constructor accuracy" begin
    set_descriptor!(2, 4)

    # CTPS(a) — constant term only
    p = CTPS(3.7)
    @test cst(p) ≈ 3.7
    @test _coeff2(p, 1, 0) ≈ 0.0
    @test _coeff2(p, 0, 1) ≈ 0.0

    # CTPS(a, n) — constant + linear variable n using global descriptor
    x = CTPS(1.5, 1)     # 1.5 + δx₁
    @test cst(x) ≈ 1.5
    @test _coeff2(x, 1, 0) ≈ 1.0
    @test _coeff2(x, 0, 1) ≈ 0.0
    @test _coeff2(x, 2, 0) ≈ 0.0

    y = CTPS(0.0, 2)     # 0 + δx₂
    @test cst(y) ≈ 0.0
    @test _coeff2(y, 1, 0) ≈ 0.0
    @test _coeff2(y, 0, 1) ≈ 1.0

    # CTPS(T) — zero CTPS
    z = CTPS(Float64)
    @test all(iszero, z.c)
    @test z.degree_mask[] == UInt64(0)
end

# ─── element / findindex / cst ────────────────────────────────────────────────

@testset "element and findindex consistency" begin
    set_descriptor!(2, 4)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    # Build a known polynomial using CTPS(Float64) which zero-initialises ALL
    # coefficient positions, so every element() call returns a meaningful value.
    p = CTPS(Float64)   # zeros(Float64, N)
    p.c[1]                           = 5.0   # constant
    p.c[2]                           = 3.0   # x₁
    p.c[3]                           = 2.0   # x₂
    p.c[findindex(p, [2, 2, 0])]     = 7.0   # x₁²
    # x₁x₂ and x₂² remain 0.0 from zeros-init
    PolySeries.update_degree_mask!(p)

    # cst returns constant term
    @test cst(p) ≈ 5.0

    # element with degree-first indexmap
    @test element(p, [0, 0, 0]) ≈ 5.0
    @test element(p, [1, 1, 0]) ≈ 3.0    # x₁ coefficient
    @test element(p, [1, 0, 1]) ≈ 2.0    # x₂ coefficient
    @test element(p, [2, 2, 0]) ≈ 7.0    # x₁² coefficient
    @test element(p, [2, 1, 1]) ≈ 0.0    # x₁x₂ coefficient (zero from zeros-init)
    @test element(p, [2, 0, 2]) ≈ 0.0    # x₂² coefficient (zero from zeros-init)

    # findindex roundtrip: element == c[findindex]
    for (e1, e2) in [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3)]
        idx = findindex(p, [e1+e2, e1, e2])
        @test p.c[idx] == element(p, [e1+e2, e1, e2])
    end

    # element with nv-length indexmap (degree inferred)
    @test element(p, [1, 0]) ≈ 3.0   # [e1, e2] only
    @test element(p, [0, 1]) ≈ 2.0
    @test element(p, [2, 0]) ≈ 7.0
end

# ─── addition / subtraction ───────────────────────────────────────────────────

@testset "Addition accuracy" begin
    set_descriptor!(2, 4)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    # (2 + 3x₁ + 4x₂) + (5 + 6x₁ - x₂) = 7 + 9x₁ + 3x₂
    a = 2.0 + 3.0*x + 4.0*y
    b = 5.0 + 6.0*x - 1.0*y
    s = a + b
    @test _coeff2(s, 0, 0) ≈ 7.0
    @test _coeff2(s, 1, 0) ≈ 9.0
    @test _coeff2(s, 0, 1) ≈ 3.0
    # Note: degree-2 coefficients are outside the active range of s (degrees 0–1
    # only), so element() would read uninitialized memory — no check here.

    # CTPS + scalar (right)
    t = a + 10.0
    @test _coeff2(t, 0, 0) ≈ 12.0
    @test _coeff2(t, 1, 0) ≈ 3.0

    # scalar + CTPS (left)
    t2 = 10.0 + a
    @test _coeff2(t2, 0, 0) ≈ 12.0
end

@testset "Subtraction accuracy" begin
    set_descriptor!(2, 4)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    a = 3.0 + 2.0*x + 5.0*y
    b = 1.0 + 4.0*x + 2.0*y

    # a - b = 2 - 2x₁ + 3x₂
    d = a - b
    @test _coeff2(d, 0, 0) ≈  2.0
    @test _coeff2(d, 1, 0) ≈ -2.0
    @test _coeff2(d, 0, 1) ≈  3.0

    # Unary minus
    neg = -a
    @test _coeff2(neg, 0, 0) ≈ -3.0
    @test _coeff2(neg, 1, 0) ≈ -2.0
    @test _coeff2(neg, 0, 1) ≈ -5.0

    # CTPS - scalar
    t1 = a - 1.0
    @test _coeff2(t1, 0, 0) ≈ 2.0
    @test _coeff2(t1, 1, 0) ≈ 2.0

    # scalar - CTPS
    t2 = 10.0 - a
    @test _coeff2(t2, 0, 0) ≈  7.0
    @test _coeff2(t2, 1, 0) ≈ -2.0
    @test _coeff2(t2, 0, 1) ≈ -5.0
end

# ─── scalar multiplication ────────────────────────────────────────────────────

@testset "Scalar multiplication accuracy" begin
    set_descriptor!(2, 4)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    p = 2.0 + 3.0*x + 4.0*y

    # scalar * CTPS
    t1 = 5.0 * p
    @test _coeff2(t1, 0, 0) ≈ 10.0
    @test _coeff2(t1, 1, 0) ≈ 15.0
    @test _coeff2(t1, 0, 1) ≈ 20.0

    # CTPS * scalar
    t2 = p * 5.0
    @test _coeff2(t2, 0, 0) ≈ 10.0
    @test _coeff2(t2, 1, 0) ≈ 15.0
    @test _coeff2(t2, 0, 1) ≈ 20.0

    # *(-1) == unary minus
    t3 = p * (-1.0)
    @test _coeff2(t3, 0, 0) ≈ -2.0
    @test _coeff2(t3, 1, 0) ≈ -3.0
    @test _coeff2(t3, 0, 1) ≈ -4.0
end

# ─── multiplication operator (full coefficient checks) ────────────────────────

@testset "Multiplication operator accuracy" begin
    set_descriptor!(2, 4)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    # (1 + 2x₁)(1 + 3x₂) = 1 + 2x₁ + 3x₂ + 6x₁x₂
    a = 1.0 + 2.0*x
    b = 1.0 + 3.0*y
    p = a * b
    @test _coeff2(p, 0, 0) ≈ 1.0
    @test _coeff2(p, 1, 0) ≈ 2.0
    @test _coeff2(p, 0, 1) ≈ 3.0
    @test _coeff2(p, 1, 1) ≈ 6.0
    @test _coeff2(p, 2, 0) ≈ 0.0
    @test _coeff2(p, 0, 2) ≈ 0.0

    # (x₁ + x₂)² = x₁² + 2x₁x₂ + x₂²
    # s = x + y has zero constant (degree_mask excludes degree-0), so check only
    # the non-zero degree-2 coefficients that ARE written by the multiplication.
    s = x + y
    q = s * s
    @test _coeff2(q, 2, 0) ≈ 1.0
    @test _coeff2(q, 1, 1) ≈ 2.0
    @test _coeff2(q, 0, 2) ≈ 1.0
end

# ─── commutativity of multiplication ─────────────────────────────────────────

@testset "Multiplication commutativity" begin
    set_descriptor!(2, 4)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    a = 1.0 + 2.0*x + 3.0*y + 0.5*(x*x) - (x*y)
    b = 4.0 - x + 5.0*y + 2.0*(y*y)

    ab = a * b
    ba = b * a
    @test ab.c ≈ ba.c
end

# ─── division ─────────────────────────────────────────────────────────────────

@testset "Division accuracy" begin
    set_descriptor!(2, 4)
    x = CTPS(0.0, 1)

    # (1 + x) / 2 = 0.5 + 0.5x
    d = (1.0 + x) / 2.0
    @test _coeff2(d, 0, 0) ≈ 0.5
    @test _coeff2(d, 1, 0) ≈ 0.5

    # scalar / CTPS: 1 / (1 + x) = 1 - x + x² - x³ + x⁴ (geometric series)
    inv1 = 1.0 / (1.0 + x)
    @test _coeff2(inv1, 0, 0) ≈  1.0
    @test _coeff2(inv1, 1, 0) ≈ -1.0
    @test _coeff2(inv1, 2, 0) ≈  1.0
    @test _coeff2(inv1, 3, 0) ≈ -1.0
    @test _coeff2(inv1, 4, 0) ≈  1.0

    # CTPS / CTPS: (1 + x) / (1 + x) = 1
    q = (1.0 + x) / (1.0 + x)
    @test _coeff2(q, 0, 0) ≈ 1.0
    @test _coeff2(q, 1, 0) ≈ 0.0  atol=1e-14
    @test _coeff2(q, 2, 0) ≈ 0.0  atol=1e-14
    @test _coeff2(q, 3, 0) ≈ 0.0  atol=1e-14

    # CTPS / scalar: (6 + 3x) / 3 = 2 + x
    r = (6.0 + 3.0*x) / 3.0
    @test _coeff2(r, 0, 0) ≈ 2.0
    @test _coeff2(r, 1, 0) ≈ 1.0
end

# ─── pow / ^ ─────────────────────────────────────────────────────────────────

@testset "pow and ^ accuracy" begin
    set_descriptor!(2, 6)
    x = CTPS(0.0, 1)

    # (1 + x)^0 = 1
    p0 = (1.0 + x)^0
    @test _coeff2(p0, 0, 0) ≈ 1.0
    @test _coeff2(p0, 1, 0) ≈ 0.0  atol=1e-15

    # (1 + x)^1 = 1 + x
    p1 = (1.0 + x)^1
    @test _coeff2(p1, 0, 0) ≈ 1.0
    @test _coeff2(p1, 1, 0) ≈ 1.0

    # (1 + x)^2 = 1 + 2x + x²
    p2 = (1.0 + x)^2
    @test _coeff2(p2, 0, 0) ≈ 1.0
    @test _coeff2(p2, 1, 0) ≈ 2.0
    @test _coeff2(p2, 2, 0) ≈ 1.0

    # (1 + x)^4 with C(4,k) coefficients
    p4 = (1.0 + x)^4
    for (k, expected) in enumerate([1.0, 4.0, 6.0, 4.0, 1.0])
        @test _coeff2(p4, k-1, 0) ≈ expected   atol=1e-13
    end

    # (1 + x)^5 — binomial coefficients [1,5,10,10,5,1]
    p5 = (1.0 + x)^5
    for (k, expected) in enumerate([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        @test _coeff2(p5, k-1, 0) ≈ expected   atol=1e-13
    end

    # Negative integer exponent: (2 + x)^(-1) = (1/2) * 1/(1 + x/2)
    #   = sum_{k≥0} (-1)^k x^k / 2^{k+1}
    pn1 = (2.0 + x)^(-1)
    for k in 0:5
        expected = (-1.0)^k / 2.0^(k+1)
        @test _coeff2(pn1, k, 0) ≈ expected   rtol=1e-12
    end

    # pow! in-place matches operator
    r = CTPS(Float64)
    pow!(r, 1.0 + x, 3)
    p3 = (1.0 + x)^3
    for e1 in 0:3
        @test _coeff2(r, e1, 0) ≈ _coeff2(p3, e1, 0)   atol=1e-14
    end
end

# ─── in-place arithmetic ──────────────────────────────────────────────────────

@testset "add! accuracy" begin
    set_descriptor!(2, 4)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    p = 1.0 + x;  q = 2.0 + y;  r = CTPS(Float64)
    add!(r, p, q)
    @test _coeff2(r, 0, 0) ≈ 3.0
    @test _coeff2(r, 1, 0) ≈ 1.0
    @test _coeff2(r, 0, 1) ≈ 1.0
end

@testset "addto! accuracy" begin
    set_descriptor!(2, 4)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    # addto!(a, b)  →  a += b
    a = 1.0 + 2.0*x
    b = 3.0 + 4.0*y
    addto!(a, b)
    @test _coeff2(a, 0, 0) ≈ 4.0
    @test _coeff2(a, 1, 0) ≈ 2.0
    @test _coeff2(a, 0, 1) ≈ 4.0
end

@testset "sub! accuracy" begin
    set_descriptor!(2, 4)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    p = 5.0 + 3.0*x;  q = 2.0 + y;  r = CTPS(Float64)
    sub!(r, p, q)
    @test _coeff2(r, 0, 0) ≈  3.0
    @test _coeff2(r, 1, 0) ≈  3.0
    @test _coeff2(r, 0, 1) ≈ -1.0
end

@testset "subfrom! accuracy" begin
    set_descriptor!(2, 4)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    # subfrom!(a, b)  →  a -= b
    a = 5.0 + 3.0*x
    b = 2.0 + x + y
    subfrom!(a, b)
    @test _coeff2(a, 0, 0) ≈  3.0
    @test _coeff2(a, 1, 0) ≈  2.0
    @test _coeff2(a, 0, 1) ≈ -1.0
end

@testset "scale! accuracy" begin
    set_descriptor!(2, 4)
    x = CTPS(0.0, 1)

    # 2-arg: scale!(a, s) → a *= s
    a = 1.0 + 2.0*x
    scale!(a, 3.0)
    @test _coeff2(a, 0, 0) ≈ 3.0
    @test _coeff2(a, 1, 0) ≈ 6.0

    # 3-arg: scale!(dest, src, s) → dest = src * s
    set_descriptor!(2, 4); x = CTPS(0.0, 1)
    src = 2.0 + 4.0*x
    dst = CTPS(Float64)
    scale!(dst, src, 0.5)
    @test _coeff2(dst, 0, 0) ≈ 1.0
    @test _coeff2(dst, 1, 0) ≈ 2.0
end

@testset "scaleadd! accuracy" begin
    # scaleadd!(result, a, c1, b, c2)  →  result = a*c1 + b*c2
    set_descriptor!(2, 4)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    c1 = 1.0 + 2.0*x
    c2 = 3.0 + 4.0*y
    result = CTPS(Float64)
    # result = 2.0*c1 + 0.5*c2  = (2 + 4x) + (1.5 + 2y) = 3.5 + 4x + 2y
    scaleadd!(result, 2.0, c1, 0.5, c2)
    @test _coeff2(result, 0, 0) ≈ 3.5
    @test _coeff2(result, 1, 0) ≈ 4.0
    @test _coeff2(result, 0, 1) ≈ 2.0
end

# ─── copy! and zero! ─────────────────────────────────────────────────────────

@testset "copy! and zero! accuracy" begin
    set_descriptor!(2, 4)
    x = CTPS(0.0, 1)
    src = 3.0 + 2.0*x + x^2
    dst = CTPS(Float64)

    copy!(dst, src)
    @test _coeff2(dst, 0, 0) ≈ 3.0
    @test _coeff2(dst, 1, 0) ≈ 2.0
    @test _coeff2(dst, 2, 0) ≈ 1.0
    @test _coeff2(dst, 0, 1) ≈ 0.0

    # Mutating src after copy must not affect dst (deep copy)
    src.c[1] = 99.0
    @test _coeff2(dst, 0, 0) ≈ 3.0

    # zero! clears all
    zero!(dst)
    @test all(iszero, dst.c)
    @test dst.degree_mask[] == UInt64(0)
end

# ─── order truncation ─────────────────────────────────────────────────────────

@testset "Order truncation" begin
    # With order-3 descriptor, degree-4+ coefficients must be absent (zero-length vector)
    # Use element() with an index ≤ order; test that degree > order gives correct boundary.
    set_descriptor!(1, 3)
    x = CTPS(0.0, 1)

    # exp(x) truncated at order 3: coefficients match 1/k! for k=0..3
    e = PolySeries.exp(x)
    for k in 0:3
        @test element(e, [k, k]) ≈ 1.0 / factorial(k)   rtol=1e-14
    end
    # descriptor has exactly N = C(1+3,1) = 4 coefficients
    @test length(e.c) == 4

    # (1 + x)^5 — coefficients beyond degree 3 are absent
    p = (1.0 + x)^5
    @test element(p, [0, 0]) ≈  1.0  # C(5,0) = 1
    @test element(p, [1, 1]) ≈  5.0  # C(5,1) = 5
    @test element(p, [2, 2]) ≈ 10.0  # C(5,2) = 10
    @test element(p, [3, 3]) ≈ 10.0  # C(5,3) = 10  ← last stored term
    @test length(p.c) == 4           # no degree-4 or 5 entries exist
end

# ─── dense multiplication exact coefficients ──────────────────────────────────

@testset "Dense multiplication exact coefficients" begin
    # (1 + x₁ + x₂ + x₁² + x₁x₂ + x₂²) * (1 + x₁)
    # = 1 + 2x₁ + x₂ + 2x₁² + 2x₁x₂ + x₂² + x₁³ + x₁²x₂ + x₁x₂²
    nv = 2; order = 4
    set_descriptor!(nv, order)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    a = 1.0 + x + y + x^2 + x*y + y^2
    b = 1.0 + x
    p = a * b

    @test _coeff2(p, 0, 0) ≈ 1.0
    # degree 1
    @test _coeff2(p, 1, 0) ≈ 2.0   # 2x₁
    @test _coeff2(p, 0, 1) ≈ 1.0   # x₂
    # degree 2: a_deg1 * x₁ + a_deg2 * 1
    #   x₁·x₁ = x₁²; x₂·x₁ = x₁x₂; (x₁²+x₁x₂+x₂²)·1
    #   → 2x₁² + 2x₁x₂ + x₂²
    @test _coeff2(p, 2, 0) ≈ 2.0
    @test _coeff2(p, 1, 1) ≈ 2.0
    @test _coeff2(p, 0, 2) ≈ 1.0
    # degree 3: a_deg2 * x₁ = x₁³ + x₁²x₂ + x₁x₂²
    @test _coeff2(p, 3, 0) ≈ 1.0
    @test _coeff2(p, 2, 1) ≈ 1.0
    @test _coeff2(p, 1, 2) ≈ 1.0
end

# ─── complex arithmetic ───────────────────────────────────────────────────────

@testset "Complex arithmetic" begin
    set_descriptor!(2, 3)
    xc = CTPS(0.0 + 0.0im, 1)
    yc = CTPS(0.0 + 0.0im, 2)

    # (1 + 2i + x₁) + (3 + x₁) = (4 + 2i) + 2x₁
    a = (1.0 + 2.0im) + xc
    b = 3.0 + xc
    s = a + b
    @test real(cst(s)) ≈ 4.0
    @test imag(cst(s)) ≈ 2.0
    @test element(s, [1, 1, 0]) ≈ 2.0 + 0.0im

    # Scalar multiplication: (2+3i) * x₁
    t = (2.0 + 3.0im) * xc
    @test real(element(t, [1, 1, 0])) ≈ 2.0
    @test imag(element(t, [1, 1, 0])) ≈ 3.0

    # (ix₁)(ix₂) = i²·x₁x₂ = -x₁x₂
    p = (1.0im * xc) * (1.0im * yc)
    @test real(element(p, [2, 1, 1])) ≈ -1.0
    @test imag(element(p, [2, 1, 1])) ≈  0.0  atol=1e-15
end

# ─── mixed-type scalar operations ─────────────────────────────────────────────

@testset "Mixed-type scalar operations" begin
    set_descriptor!(2, 3)
    x = CTPS(0.0, 1)

    # Int scalar: x + 2  →  2 + δx₁
    ri = x + 2
    @test _coeff2(ri, 0, 0) ≈ 2.0
    @test _coeff2(ri, 1, 0) ≈ 1.0

    # x * 3: the constant is outside the active degree range of x (degree-mask
    # excludes degree-0 since x has zero constant), so only check the linear term.
    ri2 = x * 3
    @test _coeff2(ri2, 1, 0) ≈ 3.0

    # Float32 scalar
    rf = x + Float32(1.5)
    @test _coeff2(rf, 0, 0) ≈ 1.5
end

@testset "Lazy degree blocks: cst/element/math functions on zero-constant series" begin
    # Coefficient blocks whose degree bit is clear are uninitialized memory.
    # `cst` and `element` must go through the degree mask, and the math
    # functions must expand about the *mathematical* constant term (zero here),
    # not about whatever `c[1]` happens to hold. Allocate and discard series in
    # between so that stale heap blocks are likely to be reused.
    set_descriptor!(2, 4)
    for trial in 1:100
        junk = [CTPS(0.7 + 0.01trial, 1) * CTPS(1.3, 2) + 5.0 for _ in 1:8]
        x = CTPS(0.0, 1)                    # pure variable: degree-0 bit clear
        @test cst(x) == 0.0
        @test element(x, [0, 0, 0]) == 0.0
        y = x / (CTPS(2.0, 2) + 1.0)        # mul! output band starts at degree 1
        @test cst(y) == 0.0
        @test cst(x * x) == 0.0
        # asin(x) = x + x^3/6 + ...; sin(x) = x - x^3/6; sqrt(1 + x) constant 1
        @test element(asin(x), [1, 1, 0]) ≈ 1.0 atol=1e-14
        @test cst(asin(x)) == 0.0
        @test element(asin(x), [3, 3, 0]) ≈ 1/6 atol=1e-14
        @test element(sin(x), [3, 3, 0]) ≈ -1/6 atol=1e-14
        @test cst(exp(x)) ≈ 1.0 atol=1e-14
        @test element(tan(x), [1, 1, 0]) ≈ 1.0 atol=1e-14
        @test cst(tan(x)) == 0.0
        length(junk) == 8 || error()
    end
end

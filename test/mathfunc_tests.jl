# Accuracy tests for all TPSA math functions
#
# Strategy:
#   1. Coefficient test  — compare every Taylor coeff c[k] against the
#      analytically-known derivative d^k f(a0) / k!
#   2. Pointwise test    — evaluate the polynomial at a0+h for several h
#      and compare against Base.fn(a0+h); uses relative error.
#   3. In-place parity   — fn!(result, x) must produce the same coefficients
#      as fn(x) to within floating-point rounding.
#
# Expansion point a0 = 0.5 — valid for log, sqrt, inv, asin, acos (|a0|<1).

# ─── reference coefficient formulas ──────────────────────────────────────────

_ref_exp(a0, order)  = [exp(a0) / factorial(k) for k in 0:order]

function _ref_log(a0, order)
    v = Vector{Float64}(undef, order + 1)
    v[1] = log(a0)
    for k in 1:order; v[k+1] = (-1)^(k+1) / (k * a0^k); end
    v
end

function _ref_sqrt(a0, order)
    v = Vector{Float64}(undef, order + 1)
    v[1] = sqrt(a0)
    binom = 1.0
    for k in 1:order
        binom *= (0.5 - (k-1)) / k
        v[k+1] = binom * a0^(0.5 - k)
    end
    v
end

_ref_inv(a0, order) = [(-1)^k / a0^(k+1) for k in 0:order]

function _ref_sin(a0, order)
    sa, ca = sin(a0), cos(a0)
    cycle  = [sa, ca, -sa, -ca]
    [cycle[mod(k,4)+1] / factorial(k) for k in 0:order]
end

function _ref_cos(a0, order)
    sa, ca = sin(a0), cos(a0)
    cycle  = [ca, -sa, -ca, sa]
    [cycle[mod(k,4)+1] / factorial(k) for k in 0:order]
end

function _ref_sinh(a0, order)
    sa, ca = sinh(a0), cosh(a0)
    [iseven(k) ? sa/factorial(k) : ca/factorial(k) for k in 0:order]
end

function _ref_cosh(a0, order)
    sa, ca = sinh(a0), cosh(a0)
    [iseven(k) ? ca/factorial(k) : sa/factorial(k) for k in 0:order]
end

# ─── helper: evaluate 1-var CTPS at a0+h ─────────────────────────────────────

function _polyval(y::CTPS, h::Float64)
    val = 0.0; hk = 1.0
    for k in 0:y.desc.order
        val += real(element(y, [k])) * hk
        hk  *= h
    end
    val
end

# ─── tests ───────────────────────────────────────────────────────────────────

const MATHFUNC_ORDER  = 8
const MATHFUNC_A0     = 0.5
const COEFF_TOL       = 1e-10
const POINTWISE_TOL   = 1e-6
const INPLACE_TOL     = 1e-14
const TEST_HS         = [0.001 * i for i in 1:5]

@testset "Math Functions" begin

    @testset "exp" begin
        set_descriptor!(1, MATHFUNC_ORDER)
        x    = CTPS(MATHFUNC_A0, 1)
        y    = PolySeries.exp(x)
        ref  = _ref_exp(MATHFUNC_A0, MATHFUNC_ORDER)
        for k in 0:MATHFUNC_ORDER
            @test element(y, [k]) ≈ ref[k+1]  atol=COEFF_TOL
        end
        for h in TEST_HS
            @test _polyval(y, h) ≈ exp(MATHFUNC_A0 + h)  rtol=POINTWISE_TOL
        end
    end

    @testset "exp!" begin
        set_descriptor!(1, MATHFUNC_ORDER)
        x = CTPS(MATHFUNC_A0, 1);  r = CTPS(Float64);  exp!(r, x)
        ref = _ref_exp(MATHFUNC_A0, MATHFUNC_ORDER)
        for k in 0:MATHFUNC_ORDER
            @test element(r, [k]) ≈ ref[k+1]  atol=COEFF_TOL
        end
    end

    @testset "log" begin
        set_descriptor!(1, MATHFUNC_ORDER)
        x    = CTPS(MATHFUNC_A0, 1)
        y    = PolySeries.log(x)
        ref  = _ref_log(MATHFUNC_A0, MATHFUNC_ORDER)
        for k in 0:MATHFUNC_ORDER
            @test element(y, [k]) ≈ ref[k+1]  atol=COEFF_TOL
        end
        for h in TEST_HS
            @test _polyval(y, h) ≈ log(MATHFUNC_A0 + h)  rtol=POINTWISE_TOL
        end
    end

    @testset "log!" begin
        set_descriptor!(1, MATHFUNC_ORDER)
        x = CTPS(MATHFUNC_A0, 1);  r = CTPS(Float64);  log!(r, x)
        ref = _ref_log(MATHFUNC_A0, MATHFUNC_ORDER)
        for k in 0:MATHFUNC_ORDER
            @test element(r, [k]) ≈ ref[k+1]  atol=COEFF_TOL
        end
    end

    @testset "sqrt" begin
        set_descriptor!(1, MATHFUNC_ORDER)
        x    = CTPS(MATHFUNC_A0, 1)
        y    = PolySeries.sqrt(x)
        ref  = _ref_sqrt(MATHFUNC_A0, MATHFUNC_ORDER)
        for k in 0:MATHFUNC_ORDER
            @test element(y, [k]) ≈ ref[k+1]  atol=COEFF_TOL
        end
        for h in TEST_HS
            @test _polyval(y, h) ≈ sqrt(MATHFUNC_A0 + h)  rtol=POINTWISE_TOL
        end
    end

    @testset "sqrt!" begin
        set_descriptor!(1, MATHFUNC_ORDER)
        x = CTPS(MATHFUNC_A0, 1);  r = CTPS(Float64);  sqrt!(r, x)
        ref = _ref_sqrt(MATHFUNC_A0, MATHFUNC_ORDER)
        for k in 0:MATHFUNC_ORDER
            @test element(r, [k]) ≈ ref[k+1]  atol=COEFF_TOL
        end
    end

    @testset "inv" begin
        set_descriptor!(1, MATHFUNC_ORDER)
        x    = CTPS(MATHFUNC_A0, 1)
        y    = inv(x)
        ref  = _ref_inv(MATHFUNC_A0, MATHFUNC_ORDER)
        for k in 0:MATHFUNC_ORDER
            @test element(y, [k]) ≈ ref[k+1]  atol=COEFF_TOL
        end
        for h in TEST_HS
            @test _polyval(y, h) ≈ 1.0 / (MATHFUNC_A0 + h)  rtol=POINTWISE_TOL
        end
    end

    @testset "sin" begin
        set_descriptor!(1, MATHFUNC_ORDER)
        x    = CTPS(MATHFUNC_A0, 1)
        y    = PolySeries.sin(x)
        ref  = _ref_sin(MATHFUNC_A0, MATHFUNC_ORDER)
        for k in 0:MATHFUNC_ORDER
            @test element(y, [k]) ≈ ref[k+1]  atol=COEFF_TOL
        end
        for h in TEST_HS
            @test _polyval(y, h) ≈ sin(MATHFUNC_A0 + h)  rtol=POINTWISE_TOL
        end
    end

    @testset "sin!" begin
        set_descriptor!(1, MATHFUNC_ORDER)
        x = CTPS(MATHFUNC_A0, 1);  r = CTPS(Float64);  sin!(r, x)
        ref = _ref_sin(MATHFUNC_A0, MATHFUNC_ORDER)
        for k in 0:MATHFUNC_ORDER
            @test element(r, [k]) ≈ ref[k+1]  atol=COEFF_TOL
        end
    end

    @testset "cos" begin
        set_descriptor!(1, MATHFUNC_ORDER)
        x    = CTPS(MATHFUNC_A0, 1)
        y    = PolySeries.cos(x)
        ref  = _ref_cos(MATHFUNC_A0, MATHFUNC_ORDER)
        for k in 0:MATHFUNC_ORDER
            @test element(y, [k]) ≈ ref[k+1]  atol=COEFF_TOL
        end
        for h in TEST_HS
            @test _polyval(y, h) ≈ cos(MATHFUNC_A0 + h)  rtol=POINTWISE_TOL
        end
    end

    @testset "cos!" begin
        set_descriptor!(1, MATHFUNC_ORDER)
        x = CTPS(MATHFUNC_A0, 1);  r = CTPS(Float64);  cos!(r, x)
        ref = _ref_cos(MATHFUNC_A0, MATHFUNC_ORDER)
        for k in 0:MATHFUNC_ORDER
            @test element(r, [k]) ≈ ref[k+1]  atol=COEFF_TOL
        end
    end

    @testset "tan" begin
        set_descriptor!(1, MATHFUNC_ORDER)
        x = CTPS(MATHFUNC_A0, 1)
        y = PolySeries.tan(x)
        for h in TEST_HS
            @test _polyval(y, h) ≈ tan(MATHFUNC_A0 + h)  rtol=POINTWISE_TOL
        end
        # Regression: the constant term used to be accumulated into an
        # uninitialized slot, giving a random cst(tan(x)). Stress the allocator
        # so a stale heap block is likely to be reused.
        for _ in 1:200
            junk = [CTPS(MATHFUNC_A0 + 0.1k, 1) * CTPS(0.3, 1) for k in 1:8]
            @test cst(PolySeries.tan(CTPS(MATHFUNC_A0, 1))) ≈ tan(MATHFUNC_A0) rtol=1e-14
            length(junk) == 8 || error()
        end
    end

    @testset "sinh" begin
        set_descriptor!(1, MATHFUNC_ORDER)
        x    = CTPS(MATHFUNC_A0, 1)
        y    = PolySeries.sinh(x)
        ref  = _ref_sinh(MATHFUNC_A0, MATHFUNC_ORDER)
        for k in 0:MATHFUNC_ORDER
            @test element(y, [k]) ≈ ref[k+1]  atol=COEFF_TOL
        end
        for h in TEST_HS
            @test _polyval(y, h) ≈ sinh(MATHFUNC_A0 + h)  rtol=POINTWISE_TOL
        end
    end

    @testset "sinh!" begin
        set_descriptor!(1, MATHFUNC_ORDER)
        x = CTPS(MATHFUNC_A0, 1);  r = CTPS(Float64);  sinh!(r, x)
        ref = _ref_sinh(MATHFUNC_A0, MATHFUNC_ORDER)
        for k in 0:MATHFUNC_ORDER
            @test element(r, [k]) ≈ ref[k+1]  atol=COEFF_TOL
        end
    end

    @testset "cosh" begin
        set_descriptor!(1, MATHFUNC_ORDER)
        x    = CTPS(MATHFUNC_A0, 1)
        y    = PolySeries.cosh(x)
        ref  = _ref_cosh(MATHFUNC_A0, MATHFUNC_ORDER)
        for k in 0:MATHFUNC_ORDER
            @test element(y, [k]) ≈ ref[k+1]  atol=COEFF_TOL
        end
        for h in TEST_HS
            @test _polyval(y, h) ≈ cosh(MATHFUNC_A0 + h)  rtol=POINTWISE_TOL
        end
    end

    @testset "cosh!" begin
        set_descriptor!(1, MATHFUNC_ORDER)
        x = CTPS(MATHFUNC_A0, 1);  r = CTPS(Float64);  cosh!(r, x)
        ref = _ref_cosh(MATHFUNC_A0, MATHFUNC_ORDER)
        for k in 0:MATHFUNC_ORDER
            @test element(r, [k]) ≈ ref[k+1]  atol=COEFF_TOL
        end
    end

    @testset "asin" begin
        set_descriptor!(1, MATHFUNC_ORDER)
        x = CTPS(MATHFUNC_A0, 1)
        y = asin(x)
        for h in TEST_HS
            @test _polyval(y, h) ≈ asin(MATHFUNC_A0 + h)  rtol=POINTWISE_TOL
        end
    end

    @testset "acos" begin
        set_descriptor!(1, MATHFUNC_ORDER)
        x = CTPS(MATHFUNC_A0, 1)
        y = acos(x)
        for h in TEST_HS
            @test _polyval(y, h) ≈ acos(MATHFUNC_A0 + h)  rtol=POINTWISE_TOL
        end
    end

    @testset "in-place parity" begin
        # fn!(r, x) must match fn(x) exactly to within rounding across all fns
        set_descriptor!(1, MATHFUNC_ORDER)
        x = CTPS(MATHFUNC_A0, 1)
        for (fn, fn!) in [(PolySeries.exp, exp!), (PolySeries.log, log!), (PolySeries.sqrt, sqrt!),
                          (PolySeries.sin, sin!), (PolySeries.cos, cos!),
                          (PolySeries.sinh, sinh!), (PolySeries.cosh, cosh!)]
            y = fn(x)
            r = CTPS(Float64);  fn!(r, x)
            for k in 0:MATHFUNC_ORDER
                @test element(r, [k]) ≈ element(y, [k])  atol=INPLACE_TOL
            end
        end
    end

    @testset "multi-variable composition" begin
        # sin(0.3 + 0.5*x1 + 0.2*x2^2) evaluated at x2=0, check x1-slice
        set_descriptor!(2, 6)
        x1 = CTPS(0.0, 1)
        x2 = CTPS(0.0, 2)
        g  = 0.3 + 0.5 * x1 + 0.2 * x2^2
        y  = PolySeries.sin(g)
        # Horner evaluation using only x1^k terms (x2 exponent = 0)
        ord = y.desc.order
        for h in TEST_HS
            val = sum(real(element(y, [k, 0])) * h^k for k in 0:ord)
            @test val ≈ sin(0.3 + 0.5 * h)  rtol=1e-4
        end
    end

end

# Mixed monomials (x¹y¹, x²y³, …) are never produced by a single-variable
# slice. Analytic identities give exact expectations for every coefficient of
# a genuinely multivariate argument without any hand computation.
@testset "Multi-variable analytic identities" begin
    for (nv, order) in ((2, 6), (3, 4))
        desc = PSDesc(nv, order)
        vars = [CTPS(0.0, v, desc) for v in 1:nv]
        s = foldl(+, vars)                       # x + y (+ z)
        rest = foldl(+, vars[2:end])             # y (+ z)
        one_p = CTPS(1.0, desc)
        allc(p) = [element(p, [Int(desc.polymap.map[i, v]) for v in 2:nv + 1]) for i in 1:desc.N]
        same(p, q; atol=1e-12) = isapprox(allc(p), allc(q); atol=atol)

        @testset "nv=$nv order=$order" begin
            @test same(exp(s), foldl(*, exp.(vars)))
            @test same(exp(s) * exp(-s), one_p)
            @test same(sin(s), sin(vars[1]) * cos(rest) + cos(vars[1]) * sin(rest))
            @test same(cos(s), cos(vars[1]) * cos(rest) - sin(vars[1]) * sin(rest))
            @test same(sin(s)^2 + cos(s)^2, one_p)
            @test same(cosh(s)^2 - sinh(s)^2, one_p)
            @test same(sinh(s), sinh(vars[1]) * cosh(rest) + cosh(vars[1]) * sinh(rest))
            @test same(tan(s) * cos(s), sin(s); atol=1e-11)
            @test same(log(foldl(*, (1.0 .+ vars))), foldl(+, log.(1.0 .+ vars)); atol=1e-11)
            @test same(exp(log(1.0 + s)), 1.0 + s; atol=1e-11)
            @test same(sqrt(1.0 + s)^2, 1.0 + s)
            @test same(sqrt(1.0 + s) * sqrt(1.0 + s), 1.0 + s)
            @test same(inv(1.0 + s) * (1.0 + s), one_p)
            @test same((1.0 + s) / (1.0 + s), one_p)
            # Inverse trigonometric series through composition with the forward one.
            u = 0.5 * s
            @test same(sin(asin(u)), u; atol=1e-11)
            @test same(cos(acos(u)), u; atol=1e-11)
            w = 0.3 + 0.4 * s                    # nonzero expansion centre
            @test same(sin(asin(w)), w; atol=1e-11)
            @test same(cos(acos(w)), w; atol=1e-11)
            # Exact mixed coefficients of exp(x+y+…): 1/(e₁! e₂! …).
            if nv == 2
                @test element(exp(s), [2, 3]) ≈ 1 / (factorial(2) * factorial(3))
                @test element(exp(s), [1, 1]) ≈ 1.0
                @test element(sin(s), [1, 2]) ≈ -1 / 2      # -(x+y)³/6 → coefficient of x y²
            else
                @test element(exp(s), [1, 1, 2]) ≈ 1 / factorial(2)
                @test element(exp(s), [1, 1, 1]) ≈ 1.0
                @test element(cos(s), [2, 1, 1]) ≈ 1 / 2    # (x+y+z)⁴/24 → 12/24
            end
        end
    end
end

@testset "sincos! and reachability of degree blocks" begin
    desc = PSDesc(2, 6)
    x = CTPS(0.0, 1, desc); y = CTPS(0.0, 2, desc)
    allc(p) = [element(p, [Int(desc.polymap.map[i, v]) for v in 2:3]) for i in 1:desc.N]

    f = 0.3 + 0.5 * x + 0.2 * y^2
    s = CTPS(Float64, desc); c = CTPS(Float64, desc)
    sincos!(s, c, f)
    @test allc(s) ≈ allc(sin(f)) atol=1e-14
    @test allc(c) ≈ allc(cos(f)) atol=1e-14
    @test allc(s) ≈ allc(sin(0.3) * cos(f - 0.3) + cos(0.3) * sin(f - 0.3)) atol=1e-13
    # Either output may alias the argument; the two outputs may not alias each other.
    s2 = CTPS(f); c2 = CTPS(Float64, desc)
    sincos!(s2, c2, s2)
    @test allc(s2) ≈ allc(sin(f)) atol=1e-14
    @test allc(c2) ≈ allc(cos(f)) atol=1e-14
    c3 = CTPS(f); s3 = CTPS(Float64, desc)
    sincos!(s3, c3, c3)
    @test allc(c3) ≈ allc(cos(f)) atol=1e-14
    @test_throws ArgumentError sincos!(s, s, f)
    @test_throws DimensionMismatch sincos!(s, CTPS(Float64, PSDesc(2, 4)), f)
    @test (@allocated sincos!(s, c, f)) == 0

    # An even argument reaches only even degrees: the odd blocks of the result
    # stay inactive instead of being written as zeros.
    g = x^2 + y^2
    even = UInt64(0b1010101)                        # degrees 0, 2, 4, 6
    e = exp(g)
    @test e.degree_mask[] == even
    @test element(e, [2, 0]) ≈ 1.0 && element(e, [4, 0]) ≈ 0.5 && element(e, [2, 2]) ≈ 1.0
    @test element(e, [6, 0]) ≈ 1 / 6 && element(e, [1, 0]) == 0.0 && element(e, [3, 0]) == 0.0
    sh = sinh(g); ch = cosh(g)
    @test sh.degree_mask[] == even && ch.degree_mask[] == even
    @test element(sh, [2, 0]) ≈ 1.0 && element(sh, [6, 0]) ≈ 1 / 6 && element(sh, [4, 0]) == 0.0
    @test element(ch, [4, 0]) ≈ 0.5 && element(ch, [2, 0]) == 0.0
    out = CTPS(1.0, 1, desc)                        # stale full content is irrelevant
    exp!(out, exp(out))
    exp!(out, g)
    @test out.degree_mask[] == even
    @test allc(out) ≈ allc(e) atol=1e-14
    # Constant arguments produce constants with a degree-0 mask only.
    k = CTPS(0.7, desc)
    for h in (exp, sin, cos, sinh, cosh)
        r = h(k)
        @test r.degree_mask[] == UInt64(1) && cst(r) ≈ h(0.7)
    end
end

@testset "inv!, div!, tan! and the @tpsa extensions" begin
    desc = PSDesc(2, 5)
    x = CTPS(0.0, 1, desc); y = CTPS(0.0, 2, desc)
    allc(p) = [element(p, [Int(desc.polymap.map[i, v]) for v in 2:3]) for i in 1:desc.N]
    a = 1.5 + x + 0.5 * y^2
    b = 2.0 - 0.3 * x * y + y
    out = CTPS(Float64, desc)

    @test allc(inv!(out, b)) ≈ allc(inv(b)) atol=1e-14
    @test allc(inv(b) * b) ≈ allc(CTPS(1.0, desc)) atol=1e-14
    @test allc(div!(out, a, b)) ≈ allc(a / b) atol=1e-14
    @test allc((a / b) * b) ≈ allc(a) atol=1e-13
    @test allc(a / b) ≈ allc(a * inv(b)) atol=1e-13
    @test allc(3.0 / b) ≈ allc(3.0 * inv(b)) atol=1e-14
    @test allc(tan!(out, a)) ≈ allc(tan(a)) atol=1e-13
    # Near the pole at π/2, tan(a)*cos(a) cancels large coefficients, so
    # testing its small residual with an absolute tolerance is ill-conditioned.
    # Independently generate tan's derivative polynomials with exact integers:
    # P₀(t)=t, Pₙ₊₁(t)=(1+t²)Pₙ′(t). Only the scalar tan uses BigFloat.
    reference = setprecision(BigFloat, 256) do
        t = tan(BigFloat(1.5))
        derivative = BigInt[0, 1] # ascending powers of t
        coeffs = BigFloat[]
        for n in 0:desc.order
            push!(coeffs, evalpoly(t, derivative) / factorial(big(n)))
            next = zeros(BigInt, length(derivative) + 1)
            for k in 1:length(derivative)-1
                next[k] += k * derivative[k + 1]
                next[k + 2] += k * derivative[k + 1]
            end
            derivative = next
        end
        # Substitute h=x+y²/2 into Σₙ tan⁽ⁿ⁾(1.5)hⁿ/n! analytically.
        map(1:desc.N) do idx
            i, j = Int(desc.polymap.map[idx, 2]), Int(desc.polymap.map[idx, 3])
            isodd(j) && return 0.0
            m = j ÷ 2
            Float64(coeffs[i + m + 1] * binomial(big(i + m), m) / big(2)^m)
        end
    end
    # Check every coefficient; a vector norm could hide errors in small terms.
    for actual in (allc(tan(a)), allc(tan!(out, a)))
        @test all(isapprox.(actual, reference; rtol=32eps(Float64), atol=0.0))
    end

    # Aliasing: every argument position.
    p = CTPS(b); inv!(p, p);          @test allc(p) ≈ allc(inv(b)) atol=1e-14
    p = CTPS(a); div!(p, p, b);       @test allc(p) ≈ allc(a / b) atol=1e-14
    p = CTPS(b); div!(p, a, p);       @test allc(p) ≈ allc(a / b) atol=1e-14
    p = CTPS(b); div!(p, p, p);       @test allc(p) ≈ allc(CTPS(1.0, desc)) atol=1e-14
    p = CTPS(a); tan!(p, p);          @test allc(p) ≈ allc(tan(a)) atol=1e-13
    p = CTPS(a); sqrt!(p, p);         @test allc(p) ≈ allc(sqrt(a)) atol=1e-14
    p = CTPS(a); log!(p, p);          @test allc(p) ≈ allc(log(a)) atol=1e-14
    q = 0.2 + 0.4 * x - 0.1 * y
    p = CTPS(q); asin!(p, p);         @test allc(p) ≈ allc(asin(q)) atol=1e-14
    p = CTPS(q); acos!(p, p);         @test allc(p) ≈ allc(acos(q)) atol=1e-14

    # Zero allocation for the whole in-place family.
    for (f!, arg) in ((inv!, b), (tan!, a), (sqrt!, a), (log!, a), (asin!, q), (acos!, q))
        f!(out, arg)
        @test (@allocated f!(out, arg)) == 0
    end
    div!(out, a, b)
    @test (@allocated div!(out, a, b)) == 0

    # Domain errors happen before anything is written or borrowed.
    pool = desc._pools[Threads.threadid()]
    capacity = pool.sp
    z = CTPS(0.0, 1, desc)
    before = copy(out.c); mask = out.degree_mask[]
    @test_throws DomainError inv!(out, z)
    @test_throws DomainError div!(out, a, z)
    @test out.c == before && out.degree_mask[] == mask
    @test pool.sp == capacity

    # @tpsa: division and the newly lowered unary calls, zero allocation.
    ws = PSWorkspace(desc, 8)
    r = CTPS(Float64, desc)
    @tpsa ws r = a / b + tan(x) - asin(0.5 * x) + acos(0.5 * y) / 2.0 + atan(x - y)
    @test allc(r) ≈ allc(a / b + tan(x) - asin(0.5 * x) + acos(0.5 * y) / 2.0 + atan(x - y)) atol=1e-13
    @test ws.sp == length(ws.bufs)
    @tpsa ws r = 2.0 / b
    @test allc(r) ≈ allc(2.0 / b) atol=1e-14
    tpsa_div_tan!(ws, r, a, b, x) = (@tpsa ws r = a / b + tan(x); r)
    tpsa_div_tan!(ws, r, a, b, x)                 # compile outside the measurement
    @test (@allocated tpsa_div_tan!(ws, r, a, b, x)) == 0
    # Scalar sub-expressions are evaluated as numbers and borrow no slot:
    # `(1+2)*(3-1)*x` needs none, `cos(μ)*x + sin(μ)*y` needs two.
    tiny = PSWorkspace(desc, 1)
    μ = 0.3
    @tpsa tiny r = (1.0 + 2.0) * (3.0 - 1.0) * x
    @test allc(r) ≈ allc(6.0 * x)
    two = PSWorkspace(desc, 2)
    @tpsa two r = cos(μ) * x + sin(μ) * y
    @test allc(r) ≈ allc(cos(μ) * x + sin(μ) * y)
    @test two.sp == 2
    @tpsa two r = 2.5                                  # a bare number on the rhs
    @test cst(r) == 2.5 && r.degree_mask[] == UInt64(1)
    # Complex scalars therefore reach a complex lhs intact: `2.0im` is the call
    # `2.0 * im`, which used to be forced through a Float64 slot (InexactError).
    cdesc = PSDesc(1, 3)
    cx = CTPS(0.0 + 0.0im, 1, cdesc)
    cr = CTPS(ComplexF64, cdesc)
    cws = PSWorkspace(cdesc, 4, ComplexF64)
    @tpsa cws cr = (1.0 + 2.0im) * cx
    @test element(cr, [1]) == 1.0 + 2.0im
    @tpsa cws cr = cx / (2.0im) - (0.5 + 0.5im)
    @test element(cr, [1]) == -0.5im && cst(cr) == -0.5 - 0.5im
    @test cws.sp == 4
end

@testset "atan and atan!" begin
    desc = PSDesc(2, 6)
    x = CTPS(0.0, 1, desc); y = CTPS(0.0, 2, desc)
    allc(p) = [element(p, [Int(desc.polymap.map[i, v]) for v in 2:3]) for i in 1:desc.N]
    one_p = CTPS(1.0, desc)

    # Exact single-variable coefficients: (-1)^k / (2k+1) on odd degrees only.
    a = atan(x)
    @test [element(a, [k, 0]) for k in 0:6] ≈ [0, 1, 0, -1/3, 0, 1/5, 0]
    @test a.degree_mask[] == UInt64(0b0101011)          # constant, then odd degrees only

    # Round trips and identities, at zero and at nonzero real centers.
    for f in (0.5 * x + 0.3 * y^2, 0.7 + x - 0.4 * y, -2.0 + 0.2 * x * y + y)
        @test allc(tan(atan(f))) ≈ allc(f) atol=1e-12
        # d/dx atan(f) = f_x / (1 + f²), checked on the x-derivative coefficients.
        af = atan(f); g = one_p + f * f
        lhs = [element(af, [k + 1, 0]) * (k + 1) for k in 0:5]            # ∂/∂x, y = 0 slice
        fx  = [element(f, [k + 1, 0]) * (k + 1) for k in 0:5]
        q   = inv(g)                                                    # 1/(1+f²)
        # multiply the two slices as truncated series in x
        conv = [sum(fx[j+1] * element(q, [k - j, 0]) for j in 0:k) for k in 0:5]
        @test lhs ≈ conv atol=1e-11
        @test cst(af) ≈ atan(cst(f))
    end
    # atan(tan(w)) = w when |w0| < π/2.
    w = 0.6 + 0.3 * x - 0.2 * y
    @test allc(atan(tan(w))) ≈ allc(w) atol=1e-12
    # Odd function.
    @test allc(atan(-x - 0.5 * y)) ≈ -allc(atan(x + 0.5 * y)) atol=1e-14
    # atan(f) + atan(1/f) = ±π/2 for a series with nonzero constant term.
    f = 2.0 + x + y^2
    @test allc(atan(f) + atan(inv(f))) ≈ allc(CTPS(pi / 2, desc)) atol=1e-12

    # In-place forms, aliasing, zero allocation, pool accounting.
    out = CTPS(Float64, desc)
    @test allc(atan!(out, w)) ≈ allc(atan(w)) atol=1e-14
    p = CTPS(w); atan!(p, p)
    @test allc(p) ≈ allc(atan(w)) atol=1e-14
    atan!(out, w)
    @test (@allocated atan!(out, w)) == 0
    pool = desc._pools[Threads.threadid()]
    capacity = pool.sp
    for _ in 1:3 * PolySeries.CTPS_POOL_SIZE
        atan!(out, w)
    end
    @test pool.sp == capacity

    # Complex coefficients: atan(i·z) = i·atanh(z) has coefficients 1/(2k+1) on
    # odd degrees, all purely imaginary.
    cdesc = PSDesc(1, 5)
    z = CTPS(0.0 + 0.0im, 1, cdesc)
    ai = atan(im * z)
    @test [element(ai, [k]) for k in 0:5] ≈ [0, im, 0, im / 3, 0, im / 5] atol=1e-14
    # A complex center off the cut follows the scalar constant and round-trips.
    zc = CTPS(0.3 + 0.4im, 1, cdesc)
    @test cst(atan(zc)) == atan(0.3 + 0.4im)
    @test [element(tan(atan(zc)), [k]) for k in 0:5] ≈ [0.3 + 0.4im, 1, 0, 0, 0, 0] atol=1e-12

    # Float32 and BigFloat keep their type.
    @test atan(CTPS(0.5f0, 1, PSDesc(1, 4))) isa CTPS{Float32}
    setprecision(BigFloat, 256) do
        ab = atan(CTPS(BigFloat(0), 1, PSDesc(1, 7)))
        @test element(ab, [7]) ≈ -BigFloat(1) / 7 rtol=BigFloat("1e-65")
    end
end

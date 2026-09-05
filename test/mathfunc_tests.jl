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

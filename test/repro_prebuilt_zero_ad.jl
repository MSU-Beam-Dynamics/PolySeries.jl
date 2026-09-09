# Regression coverage for Enzyme arithmetic on entirely inactive prebuilt
# polynomials, including poisoned inactive storage.
using Test, PolySeries, Enzyme

@testset "Prebuilt structurally zero polynomial derivatives" begin
    p = CTPS(0.0, PSDesc(1, 3))
    tangent = CTPS(1.0, p.desc)
    @test cst(Enzyme.gradient(Reverse, q -> cst(q * 2.0), p)[1]) == 2.0
    @test cst(Enzyme.gradient(Reverse, q -> cst(exp(q)), p)[1]) == 1.0
    @test Enzyme.autodiff(Forward, q -> cst(q * 2.0), Duplicated(p, tangent))[1] == 2.0
    @test Enzyme.autodiff(Forward, q -> cst(exp(q)), Duplicated(p, tangent))[1] == 1.0

    poisoned = CTPS(Float64, p.desc)
    fill!(poisoned.c, NaN)
    @test cst(Enzyme.gradient(Reverse, q -> cst(q * 2.0), poisoned)[1]) == 2.0
    @test cst(Enzyme.gradient(Reverse, q -> cst(exp(q)), poisoned)[1]) == 1.0
    @test all(isnan, poisoned.c)
end

@testset "Prebuilt inactive coefficients across arithmetic" begin
    desc = PSDesc(1, 3)
    cases = (
        ("addition", q -> q + 2.0, 1.0),
        ("subtraction", q -> q - 2.0, 1.0),
        ("scalar minus polynomial", q -> 2.0 - q, -1.0),
        ("negation", q -> -q, -1.0),
        ("polynomial addition", q -> q + CTPS(2.0, q.desc), 1.0),
        ("polynomial subtraction", q -> CTPS(2.0, q.desc) - q, -1.0),
        ("polynomial multiplication", q -> q * CTPS(2.0, q.desc), 2.0),
        ("division", q -> q / 2.0, 0.5),
        ("exponential", exp, 1.0),
        ("sine", sin, 1.0),
        ("cosine", cos, 0.0),
        ("tangent", tan, 1.0),
        ("hyperbolic sine", sinh, 1.0),
        ("hyperbolic cosine", cosh, 0.0),
        ("arcsine", asin, 1.0),
        ("arccosine", acos, -1.0),
        ("square", q -> q^2, 0.0),
        ("shifted logarithm", q -> log(q + 2.0), 0.5),
        ("shifted square root", q -> sqrt(q + 4.0), 0.25),
        ("exp!", q -> begin out=CTPS(Float64,q.desc); exp!(out,q); out end, 1.0),
        ("scale!", q -> begin out=CTPS(Float64,q.desc); scale!(out,q,2.0); out end, 2.0),
        ("copy!", q -> begin out=CTPS(Float64,q.desc); PolySeries.copy!(out,q); out end, 1.0),
        ("addto!", q -> begin out=CTPS(Float64,q.desc); addto!(out,q); out end, 1.0),
        ("subfrom!", q -> begin out=CTPS(Float64,q.desc); subfrom!(out,q); out end, -1.0),
        ("scaleadd!", q -> begin out=CTPS(Float64,q.desc); scaleadd!(out,2.0,q,3.0,q); out end, 5.0),
        ("composition", q -> compose(q,[CTPS(0.0,1,q.desc)]), 1.0),
    )
    for (name, transform, expected) in cases, degree in (0, 2)
        @testset "$name degree $degree" begin
            p = CTPS(Float64, desc)
            fill!(p.c, NaN)
            tangent = CTPS(Float64, desc)
            tangent.c[degree+1] = 1.0
            PolySeries.update_degree_mask!(tangent)
            f = q -> element(transform(q), [degree])
            @test Enzyme.autodiff(Forward, f, Duplicated(p,tangent))[1] ≈ expected
            grad = Enzyme.gradient(Reverse, f, p)[1]
            @test [element(grad,[k]) for k in 0:3] ≈ [k == degree ? expected : 0.0 for k in 0:3]
            @test p.degree_mask[] == 0
            @test all(isnan,p.c)
        end
    end
    p = CTPS(Float64, desc)
    fill!(p.c, NaN)
    grad = Enzyme.gradient(Reverse, q -> q(0.25), p)[1]
    @test [element(grad,[k]) for k in 0:3] ≈ [0.25^k for k in 0:3]
    x = CTPS(0.0, 1, desc)
    fixed = Enzyme.Const(x*x)
    @test Enzyme.gradient(Reverse, fixed, 0.5)[1] ≈ 1.0
    @test Enzyme.autodiff(Forward, fixed, Duplicated(0.5,1.0))[1] ≈ 1.0
end

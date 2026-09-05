using Test, PolySeries, Enzyme

@testset "Enzyme accessors on prebuilt polynomials" begin
    @test Base.get_extension(PolySeries, :PolySeriesEnzymeExt) !== nothing
    desc = set_descriptor!(1, 3)
    for a in (0.0, 0.5)
        p = CTPS(a, 1)
        tangent = CTPS(1.0)
        @test Enzyme.autodiff(Forward, cst, Duplicated(p, tangent))[1] ≈ 1.0
        @test cst(Enzyme.gradient(Reverse, cst, p)[1]) ≈ 1.0
        @test Enzyme.autodiff(Forward, q -> cst(exp(q)), Duplicated(p, tangent))[1] ≈ exp(a)
        @test cst(Enzyme.gradient(Reverse, q -> cst(exp(q)), p)[1]) ≈ exp(a)
    end
    p = CTPS(0.0, 1)
    p.c[1] = NaN # Inactive primal storage must never be read by the AD rules.
    @test cst(p) == 0.0
    @test Enzyme.autodiff(Forward, cst, Duplicated(p, CTPS(1.0)))[1] == 1.0
    @test cst(Enzyme.gradient(Reverse, cst, p)[1]) == 1.0
    @test p.degree_mask[] == UInt64(2)
    @test isnan(p.c[1])

    for degree in (0, 2, 3)
        coefficient = q -> element(q, [degree])
        p = CTPS(Float64)
        fill!(p.c, NaN) # Every primal degree is inactive.
        tangent = CTPS(Float64)
        tangent.c[degree+1] = 1.0
        PolySeries.update_degree_mask!(tangent)
        @test coefficient(p) == 0.0
        @test Enzyme.autodiff(Forward, coefficient, Duplicated(p, tangent))[1] == 1.0
        grad = Enzyme.gradient(Reverse, coefficient, p)[1]
        @test element(grad, [degree]) == 1.0
        @test all(element(grad, [d]) == (d == degree ? 1.0 : 0.0) for d in 0:3)
    end
end

@testset "Coefficient shadows are independent and reusable" begin
    d = set_descriptor!(2, 3)
    p = CTPS(0.0, 1)
    shadow = Enzyme.make_zero(p)
    @test shadow.c !== p.c
    @test shadow.degree_mask !== p.degree_mask
    @test shadow.desc === p.desc
    # Accumulate twice in the same inactive degree block, then reuse the seed.
    f = q -> element(q, [0, 2]) + 2element(q, [1, 1])
    Enzyme.autodiff(Reverse, f, Enzyme.Active, Duplicated(p, shadow))
    @test element(shadow, [0, 2]) == 1.0
    @test element(shadow, [1, 1]) == 2.0
    @test element(shadow, [2, 0]) == 0.0
    @test p.degree_mask[] == UInt64(2)
    Enzyme.autodiff(Reverse, f, Enzyme.Active, Duplicated(p, shadow))
    @test element(shadow, [0, 2]) == 2.0
    @test element(shadow, [1, 1]) == 4.0
    for width in (1, 2)
        seeds = ntuple(i -> CTPS(Float64(i), d), width)
        expected = width == 1 ? 1.0 : (1.0, 2.0)
        result = Enzyme.autodiff(Forward, cst, BatchDuplicated(p, seeds))[1]
        @test (width == 1 ? result : Tuple(result)) == expected
    end
    # Return both value and derivative, including a poisoned inactive primal.
    p.c[1] = NaN
    @test Enzyme.autodiff(Enzyme.ForwardWithPrimal, cst, Duplicated(p, CTPS(1.0, d))) == (1.0, 0.0)
end

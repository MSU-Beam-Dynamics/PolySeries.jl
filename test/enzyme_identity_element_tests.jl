# zero(p) and one(p) allocate initialized storage under AD, so a structural
# zero still carries a tangent slot; neither reads the source polynomial, so
# neither contributes to a derivative. iszero is a primal predicate.
using Test, PolySeries, Enzyme

@testset "Enzyme: zero, one and iszero" begin
    set_descriptor!(1, 4)

    cases = (
        # zero(p)/one(p) ignore p's values: the derivative comes only from the
        # rest of the expression.
        ("zero is constant in its source", a -> cst(zero(CTPS(a, 1))), a -> 0.0),
        ("one is constant in its source", a -> cst(one(CTPS(a, 1))), a -> 0.0),
        ("zero coefficient is constant", a -> element(zero(CTPS(a, 1)), [1]), a -> 0.0),
        # Identity elements inside a differentiated expression.
        ("additive identity", a -> cst(CTPS(a, 1) + zero(CTPS(a, 1))), a -> 1.0),
        ("multiplicative identity", a -> cst(CTPS(a, 1) * one(CTPS(a, 1))), a -> 1.0),
        ("annihilator", a -> cst(CTPS(a, 1) * zero(CTPS(a, 1))), a -> 0.0),
        ("identity in a linear coefficient", a -> element(a * one(CTPS(0.0, 1)), [0]), a -> 1.0),
        ("sum against zero", a -> element(exp(CTPS(a, 1)) + zero(CTPS(a, 1)), [2]),
            a -> exp(a) / 2),
        # A numerically zero coefficient still carries its tangent through the
        # accessor, even though iszero reports the primal as zero.
        ("branch on iszero", a -> begin
            p = CTPS(a, 1) - CTPS(a, 1)
            iszero(p) ? cst(CTPS(a, 1)) : 0.0
        end, a -> 1.0),
    )

    for (name, f, df) in cases
        @testset "$name" begin
            for a in (0.0, 0.3, -0.7)
                @test Enzyme.autodiff(Forward, f, Duplicated(a, 1.0))[1] ≈ df(a) atol=1e-12
                @test Enzyme.gradient(Reverse, f, a)[1] ≈ df(a) atol=1e-12
            end
        end
    end

    @testset "iszero reports the primal, not the tangent" begin
        # The series is numerically zero while its coefficients still carry a
        # tangent: the predicate is true, and the tangent reaches a read.
        cancelled(a) = CTPS(a, 1) - CTPS(a, 1)
        @test iszero(cancelled(0.25))
        probe(a) = element(cancelled(a) + CTPS(a, 1), [0])
        @test Enzyme.autodiff(Forward, probe, Duplicated(0.25, 1.0))[1] ≈ 1.0
        @test Enzyme.gradient(Reverse, probe, 0.25)[1] ≈ 1.0
    end

    @testset "zero and one hold their descriptor under AD" begin
        function widths(a)
            p = CTPS(a, 1)
            z = zero(p)
            o = one(p)
            return cst(o) * element(z + p, [1]) + cst(z)
        end
        @test Enzyme.autodiff(Forward, widths, Duplicated(0.4, 1.0))[1] ≈ 0.0
        @test widths(0.4) == 1.0
    end
end

# Analytic references exercise branches at exact zeros, not finite differences
# near zero. Run this file directly as well as through the full test runner.
using Test, PolySeries, Enzyme

function _enzyme_written_coefficient(a)
    p = CTPS(Float64) # Fully initialized storage before editing coefficients.
    p.c[2] = a
    PolySeries.update_degree_mask!(p)
    return element(p, [1])
end

@testset "Enzyme: zero-valued differentiable coefficients" begin
    set_descriptor!(1, 4)
    cases = (
        ("constant accessor", a -> cst(CTPS(a)), a -> 1.0),
        ("variable accessor", a -> cst(CTPS(a, 1)), a -> 1.0),
        ("constant coefficient", a -> element(CTPS(a, 1), [0]), a -> 1.0),
        ("constant exponential", a -> cst(exp(CTPS(a))), exp),
        ("variable exponential", a -> cst(exp(CTPS(a, 1))), exp),
        ("exponential coefficient", a -> element(exp(CTPS(a, 1)), [2]), a -> exp(a)/2),
        ("sine constant", a -> cst(sin(CTPS(a, 1))), cos),
        ("sine zero coefficient", a -> element(sin(CTPS(a, 1)), [2]), a -> -cos(a)/2),
        ("cosine zero coefficient", a -> element(cos(CTPS(a, 1)), [1]), a -> -cos(a)),
        ("asin constant", a -> cst(asin(CTPS(a, 1))), a -> 1/sqrt(1-a*a)),
        ("scalar addition", a -> cst(CTPS(0.0, 1) + a), a -> 1.0),
        ("scalar subtraction", a -> cst(a - CTPS(0.0, 1)), a -> 1.0),
        ("zero scale", a -> element(a * CTPS(0.0, 1), [1]), a -> 1.0),
        ("zero product", a -> element(CTPS(a) * CTPS(0.0, 1), [1]), a -> 1.0),
        ("zero product coefficient", a -> element(CTPS(a, 1) * CTPS(0.0, 1), [1]), a -> 1.0),
        ("zero power coefficient", a -> element(CTPS(a, 1)^2, [1]), a -> 2.0),
        ("evaluation", a -> (a * CTPS(0.0, 1))(0.25), a -> 0.25),
        ("composition", a -> element(compose(a * CTPS(0.0, 1), [CTPS(0.0, 1)]), [1]), a -> 1.0),
        ("coefficient mask update", _enzyme_written_coefficient, a -> 1.0),
        ("zero numerator", a -> cst(CTPS(a)/CTPS(2.0)), a -> 0.5),
        ("zero scalar numerator", a -> cst(a/CTPS(2.0)), a -> 0.5),
        ("cancellation at nonzero center", a -> cst(CTPS(a) - 0.5), a -> 1.0),
    )
    for (name, f, derivative) in cases
        @testset "$name" begin
            for a in (0.0, 0.5)
                @test Enzyme.gradient(Reverse, f, a)[1] ≈ derivative(a) atol=1e-12
                @test Enzyme.autodiff(Forward, f, Duplicated(a, 1.0))[1] ≈ derivative(a) atol=1e-12
            end
        end
    end
end

using Test, PolySeries, Enzyme

@testset "Enzyme differentiates aliased in-place math" begin
    for order in (2, 3)
        set_descriptor!(1, order)
        for (f!, expected) in ((exp!, exp), (sin!, a -> -sin(a)), (cos!, a -> -cos(a)),
                               (log!, a -> -1/a^2), (sinh!, sinh), (cosh!, cosh),
                               (asin!, a -> a/(1-a*a)^(3/2)),
                               (acos!, a -> -a/(1-a*a)^(3/2)),
                               ((out, p) -> mul!(out, p, p), a -> 2.0),
                               ((out, p) -> pow!(out, p, 2), a -> 2.0),
                               ((out, p) -> pow!(out, p, 3), a -> 6a))
            f = a -> begin
                p = CTPS(a, 1)
                f!(p, p)
                element(p, [1])
            end
            @test Enzyme.autodiff(Forward, f, Duplicated(0.5, 1.0))[1] ≈ expected(0.5)
            @test Enzyme.gradient(Reverse, f, 0.5)[1] ≈ expected(0.5)
        end
    end
end

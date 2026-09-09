using Test, PolySeries, Enzyme

@testset "Enzyme: normalized high-order terms" begin
    for (T, order, slope, f, f!, degree) in (
        (Float32, 40, 10, exp, exp!, 40),
        (Float32, 40, 10, sin, sin!, 39),
        (Float64, 63, 100000, exp, exp!, 63))
        desc = PSDesc(1, order)
        magnitude = big(slope)^(degree-1) // factorial(big(degree-1))
        expected = T(f === sin ? (-1)^((degree-1)÷2) * magnitude : magnitude)
        allocating = a -> element(f(a * CTPS(zero(T), 1, desc)), [degree])
        inplace = a -> begin
            x = a * CTPS(zero(T), 1, desc)
            out = CTPS(T, desc)
            f!(out, x)
            element(out, [degree])
        end
        for objective in (allocating, inplace)
            @test isapprox(Enzyme.autodiff(Forward, objective, Duplicated(T(slope), one(T)))[1],
                           expected; rtol=200eps(T), atol=0)
            @test isapprox(Enzyme.gradient(Reverse, objective, T(slope))[1],
                           expected; rtol=200eps(T), atol=0)
        end
    end
end

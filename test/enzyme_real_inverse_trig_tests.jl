using Test, PolySeries, Enzyme

@testset "Enzyme sensitivities of near-boundary inverse-trig coefficients" begin
    d = PSDesc(1, 3)
    for T in (Float32, Float64), sign in (-1, 1)
        center = sign * (T === Float32 ? T(0.9999) : T(0.99999999))
        reference = setprecision(BigFloat, 512) do
            a = BigFloat(center)
            root = sqrt((1-a)*(1+a))
            T(a/root^3) # d/da of the linear Taylor coefficient
        end
        for (f, factor) in ((asin, 1), (acos, -1))
            coefficient = a -> element(f(CTPS(a, 1, d)), [1])
            expected = factor*reference
            @test isapprox(Enzyme.autodiff(Forward, coefficient, Duplicated(center, one(T)))[1],
                           expected; rtol=64eps(T))
            @test isapprox(Enzyme.gradient(Reverse, coefficient, center)[1],
                           expected; rtol=64eps(T))
        end
    end
end

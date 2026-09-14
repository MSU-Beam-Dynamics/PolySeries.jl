using Test, PolySeries

@testset "Real inverse trig accuracy near branch points" begin
    setprecision(BigFloat, 256) do
        d = PSDesc(1, 3)
        for T in (Float32, Float64, BigFloat)
            positive = T === Float32 ? T[0.5, 0.9999, prevfloat(one(T))] :
                       T === Float64 ? T[0.5, 0.9999, 0.99999999, prevfloat(one(T))] :
                       T[0.5, one(T)-T(2)^(-120), prevfloat(one(T))]
            centers = vcat(T[0], positive, -positive)
            for a in centers
                # Independent closed-form coefficients at the exactly
                # represented input, evaluated with extra reference precision.
                reference = setprecision(BigFloat, 512) do
                    z = BigFloat(a)
                    root = sqrt((1-z)*(1+z))
                    [inv(root), z/(2root^3), (1+2z*z)/(6root^5)]
                end
                p = CTPS(a, 1, d)
                p.c[3:end] .= T(NaN) # Inactive degree blocks must remain unread.
                for (f, f!, sign) in ((asin, asin!, 1), (acos, acos!, -1))
                    out, aliased = CTPS(T, d), CTPS(p)
                    f!(out, p)
                    f!(aliased, aliased)
                    for result in (f(p), out, aliased)
                        @test cst(result) == f(a)
                        for n in 1:3
                            @test isapprox(element(result, [n]), T(sign*reference[n]);
                                           rtol=32eps(T), atol=zero(T))
                        end
                    end
                end
                @test cst(p) == a && element(p, [1]) == one(T)
                @test all(isnan, p.c[3:end])
            end
        end
    end
end

@testset "Factored real inverse trig retains zero-allocation kernels" begin
    d = PSDesc(1, 3)
    p, out = CTPS(0.99999999, 1, d), CTPS(Float64, d)
    for f! in (asin!, acos!)
        f!(out, p)
        @test (@allocated f!(out, p)) == 0
    end
end

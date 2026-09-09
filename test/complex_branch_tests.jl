using Test, PolySeries

@testset "Complex series follow scalar branches" begin
    for R in (Float32, Float64, BigFloat)
        T = Complex{R}
        desc = PSDesc(1, 5)
        tolerance = R === Float32 ? R(2e-5) : R === Float64 ? R(1e-12) : R("1e-65")
        step = R === BigFloat ? R("1e-12") : R(1e-3)
        centers = [T(re, im) for re in (-2, -0.5, 0.5, 2),
                   im in (R(0), -R(0), R(0.01), R(-0.01))]
        for a in centers
            p = CTPS(a, 1, desc)
            p.c[3:end] .= T(NaN) # Inactive storage must remain unread.
            # Independent scalar identity selects the branch of Base.asin.
            root = cos(asin(a))
            asin_coeffs = (inv(root), a/(2root^3), (1+2a^2)/(6root^5))
            for (f, f!, coefficients) in (
                    (asin, asin!, asin_coeffs),
                    (acos, acos!, map(-, asin_coeffs)),
                    (log, log!, (inv(a), -inv(2a^2), inv(3a^3))),
                    (sqrt, sqrt!, (inv(2sqrt(a)), -inv(8a*sqrt(a)), inv(16a^2*sqrt(a)))))
                @testset "$R $f at $a" begin
                    results = [f(p), CTPS(T, desc), CTPS(p)]
                    f!(results[2], p)
                    f!(results[3], results[3])
                    for q in results
                        # Signed zeros select opposite sides of complex cuts.
                        @test isequal(cst(q), f(a))
                        for degree in 1:3
                            @test element(q, [degree]) ≈ coefficients[degree] rtol=tolerance atol=tolerance
                        end
                        target = f(T(real(a) + step, imag(a)))
                        @test q(T(step)) ≈ target rtol=tolerance atol=tolerance
                    end
                end
            end
        end
    end
end

@testset "Inverse trig rejects complex branch points before mutation" begin
    for R in (Float32, Float64, BigFloat), a in (-one(R), one(R))
        desc = PSDesc(1, 3)
        p = CTPS(Complex(a, zero(R)), 1, desc)
        for (f,f!) in ((asin,asin!), (acos,acos!))
            @test_throws DomainError f(p)
            out = CTPS(Complex(R(7), R(2)), desc)
            original = copy(out.c)
            @test_throws DomainError f!(out, p)
            @test out.c == original
            original = copy(p.c)
            @test_throws DomainError f!(p, p)
            @test p.c == original
        end
    end
end

@testset "Acos constant avoids subtraction cancellation" begin
    for R in (Float32, Float64, BigFloat)
        desc = PSDesc(1, 3)
        a = prevfloat(one(R))
        for value in (a, Complex(a, zero(R)), Complex(a, -zero(R)))
            p = CTPS(value, 1, desc)
            @test isequal(cst(acos(p)), acos(value))
            acos!(p, p)
            @test isequal(cst(p), acos(value))
        end
    end
end

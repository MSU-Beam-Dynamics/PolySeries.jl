using Test, PolySeries

@testset "Descriptor order limits" begin
    original = set_descriptor!(1, 4)
    @test_throws ArgumentError set_descriptor!(1, 64)
    @test get_descriptor() === original
    @test_throws ArgumentError PSDesc(1, -1)
    for (nv, order) in ((0, 1), (-1, 1), (128, 1), (typemax(Int), 1),
                        (1, typemax(Int)), (127, 63))
        @test_throws ArgumentError set_descriptor!(nv, order)
        @test get_descriptor() === original
    end
    @test PSDesc(127, 1).N == 128
    constant = CTPS(2.0, PSDesc(1, 0))
    @test cst(exp(constant)) ≈ exp(2.0)
    desc = PSDesc(1, 63)
    x = CTPS(0.0, 1, desc)
    @test element(x^63, [63]) == 1.0
    @test element(x^64, [63]) == 0.0 # Valid truncation at the supported limit.
end

function scaled_entire_reference(f, n, slope)
    coeff = big(slope)^n // factorial(big(n))
    f === exp && return coeff
    f === sin && return isodd(n) ? (-1)^((n-1)÷2) * coeff : zero(coeff)
    f === cos && return iseven(n) ? (-1)^(n÷2) * coeff : zero(coeff)
    f === sinh && return isodd(n) ? coeff : zero(coeff)
    f === cosh && return iseven(n) ? coeff : zero(coeff)
end

@testset "Scaled high-order entire functions" begin
    for (T, order, slope) in ((Float32, 40, 10), (Float64, 63, 100000))
        desc = PSDesc(1, order)
        x = T(slope) * CTPS(zero(T), 1, desc)
        # Only degree one is active; higher storage must never be read.
        x.c[3:end] .= T(NaN)
        for (f, f!) in ((exp, exp!), (sin, sin!), (cos, cos!),
                        (sinh, sinh!), (cosh, cosh!))
            y = f(x)
            out = CTPS(T, desc)
            f!(out, x)
            aliased = CTPS(x)
            f!(aliased, aliased)
            for n in 0:order
                expected = T(scaled_entire_reference(f, n, slope))
                for result in (y, out, aliased)
                    @test isapprox(element(result, [n]), expected; rtol=100eps(T), atol=0)
                end
            end
        end
    end
end

# Exact independent coefficient references; factorials here use arbitrary-size
# integers and are evaluated only in the tests, never in the package kernels.
function order_reference(f, n)
    fac = factorial(big(n))
    f === exp && return 1 // fac
    f === sin && return isodd(n) ? (-big(1))^((n-1)÷2) // fac : big(0)//1
    f === cos && return iseven(n) ? (-big(1))^(n÷2) // fac : big(0)//1
    f === sinh && return isodd(n) ? 1 // fac : big(0)//1
    f === cosh && return iseven(n) ? 1 // fac : big(0)//1
    f === log && return n == 0 ? big(0)//1 : (-big(1))^(n+1)//n
    f === sqrt && return n == 0 ? big(1)//1 :
        (-big(1))^(n-1) * binomial(big(2n), n) // (big(4)^n * (2n-1))
    # asin(x) = Σ_{k≥0} (2k)! / (4^k (k!)² (2k+1)) x^{2k+1}
    f === asin && return isodd(n) ? binomial(big(n - 1), (n - 1) ÷ 2) // (big(4)^((n - 1) ÷ 2) * n) : big(0)//1
    error("Missing reference")
end

@testset "High-order coefficient recurrences" begin
    for T in (Float64, BigFloat), order in (21, 63)
        setprecision(BigFloat, 256) do
            desc = PSDesc(1, order)
            tolerance = T === BigFloat ? BigFloat("1e-65") : 1e-12
            for (f, f!) in ((exp, exp!), (sin, sin!), (cos, cos!),
                            (sinh, sinh!), (cosh, cosh!), (log, log!), (sqrt, sqrt!),
                            (asin, asin!))
                @testset "$T order=$order $f" begin
                    center = f in (log, sqrt) ? one(T) : zero(T)
                    # The asin coefficient recurrence is O(order²) scalar work;
                    # allow it a little more Float64 rounding at order 63.
                    tol = (f === asin && T === Float64) ? 1e-10 : tolerance
                    x = CTPS(center, 1, desc)
                    y = f(x)
                    out = CTPS(T, desc)
                    f!(out, x)
                    for n in 0:order
                        expected = T(order_reference(f, n))
                        @test element(y, [n]) ≈ expected rtol=tol atol=0
                        @test element(out, [n]) ≈ expected rtol=tol atol=0
                    end
                end
            end
            # tan satisfies y' = 1 + y². Compute its reference in exact rationals.
            coeffs = zeros(Rational{BigInt}, order + 1)
            for n in 0:order-1
                coeffs[n+2] = ((n == 0 ? 1 : 0) +
                    sum(coeffs[k+1]*coeffs[n-k+1] for k in 0:n)) / (n+1)
            end
            @testset "$T order=$order tan" begin
                y = tan(CTPS(zero(T), 1, desc))
                # Division of sin/cos compounds cancellation at high Float64 orders.
                for n in 0:order
                    @test element(y, [n]) ≈ T(coeffs[n+1]) rtol=(T === Float64 ? 1e-11 : tolerance) atol=0
                end
            end
        end
    end
end

@testset "Constants retain BigFloat precision" begin
    setprecision(BigFloat, 256) do
        desc = PSDesc(1, 4)
        x = CTPS(BigFloat(0), 1, desc)
        out = CTPS(BigFloat, desc)
        acos!(out, x)
        @test cst(acos(x)) ≈ acos(BigFloat(0)) rtol=BigFloat("1e-65")
        @test cst(out) ≈ acos(BigFloat(0)) rtol=BigFloat("1e-65")
        @test element(log(CTPS(BigFloat(1), 1, desc)), [3]) ≈
            BigFloat(1)/3 rtol=BigFloat("1e-65")
    end
end

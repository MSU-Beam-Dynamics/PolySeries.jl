using Test, PolySeries

@testset "In-place math preserves aliased input" begin
    for T in (Float64, BigFloat)
        desc = PSDesc(1, 5)
        x = CTPS(zero(T), 1, desc)
        source = T(0.5) + x + x^3
        # Poison the gaps to ensure only initialized input blocks are copied.
        for d in 0:desc.order
            if source.degree_mask[] & (UInt64(1) << d) == 0
                source.c[d+1] = T(NaN)
            end
        end
        for (f, f!) in ((exp, exp!), (sin, sin!), (cos, cos!), (log, log!),
                        (sqrt, sqrt!), (sinh, sinh!), (cosh, cosh!), (asin, asin!), (acos, acos!))
            expected = f(source)
            @testset "$T $f!" begin
                for shared_wrapper in (false, true)
                    input = CTPS(source)
                    output = shared_wrapper ? CTPS{T}(input.c, desc, Ref(input.degree_mask[])) : input
                    f!(output, input)
                    for degree in 0:desc.order
                        @test element(output, [degree]) ≈ element(expected, [degree]) rtol=1e-12
                    end
                end
            end
        end
        for exponent in (0, 1, 2, 3, 4, 5, 9)
            input = CTPS(source)
            expected = input^exponent
            pow!(input, input, exponent)
            @test all(element(input, [d]) ≈ element(expected, [d]) for d in 0:desc.order)
        end
    end
end

@testset "Aliased math keeps pool capacity and zero allocations" begin
    desc = set_descriptor!(1, 4)
    pool = desc._pools[Threads.threadid()]
    capacity = pool.sp
    input = CTPS(0.5, 1)
    p = CTPS(input)
    for f! in (exp!, sin!, cos!, log!, sqrt!, sinh!, cosh!)
        f!(p, p) # Compile before measuring.
        PolySeries.copy!(p, input)
        @test (@allocated f!(p, p)) == 0
        @test pool.sp == capacity
        PolySeries.copy!(p, input)
    end
    for exponent in (2, 3)
        pow!(p, p, exponent)
        PolySeries.copy!(p, input)
        @test (@allocated pow!(p, p, exponent)) == 0
        @test pool.sp == capacity
    end
    bad = CTPS(-1.0, 1)
    before = copy(bad.c)
    mask = bad.degree_mask[]
    @test_throws DomainError log!(bad, bad)
    @test bad.c == before
    @test bad.degree_mask[] == mask
    @test pool.sp == capacity
end

@testset "Aliased constant and zero polynomials" begin
    for (f, f!) in ((exp, exp!), (sin, sin!), (cos, cos!), (log, log!),
                    (sqrt, sqrt!), (sinh, sinh!), (cosh, cosh!), (asin, asin!), (acos, acos!))
        p = CTPS(0.5, PSDesc(1, 0))
        f!(p, p)
        @test cst(p) ≈ f(0.5)
    end
    for (f, f!) in ((exp, exp!), (sin, sin!), (cos, cos!), (sinh, sinh!), (cosh, cosh!))
        p = CTPS(Float64, PSDesc(1, 4))
        f!(p, p)
        @test cst(p) == f(0.0)
        @test all(element(p, [d]) == 0 for d in 1:4)
    end
end

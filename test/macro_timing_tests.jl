using Test, PolySeries

timing_sum!(ws, out, p, later) = (@tpsa ws out = p + 1 + later())
timing_product!(ws, out, p, later) = (@tpsa ws out = p * 2 * later())
timing_binary_sum!(ws, out, p, later) = (@tpsa ws out = (p + 1) + later())
timing_binary_product!(ws, out, p, later) = (@tpsa ws out = (p * 2) * later())
timing_long_sum!(ws, out, p, later) = (@tpsa ws out = p + 1 + later() + p)
timing_long_product!(ws, out, p, later) = (@tpsa ws out = p * 2 * later() * p)
timing_nested_sum!(ws, out, p, later) = (@tpsa ws out = (p + 1 + later()) * p)
timing_owned_sum!(ws, out, p, later) = (@tpsa ws out = (p + p) + p + later())
timing_owned_product!(ws, out, p, later) = (@tpsa ws out = (p * p) * p * later())

@testset "Macro n-ary operand evaluation timing" begin
    d = PSDesc(1, 3)
    @testset "Mutating later operands and explicit parentheses" begin
        cases = (
            (timing_sum!, (p, later) -> p + 1 + later()),
            (timing_product!, (p, later) -> p * 2 * later()),
            (timing_binary_sum!, (p, later) -> (p + 1) + later()),
            (timing_binary_product!, (p, later) -> (p * 2) * later()),
            (timing_long_sum!, (p, later) -> p + 1 + later() + p),
            (timing_long_product!, (p, later) -> p * 2 * later() * p),
            (timing_nested_sum!, (p, later) -> (p + 1 + later()) * p),
            (timing_owned_sum!, (p, later) -> (p + p) + p + later()),
            (timing_owned_product!, (p, later) -> (p * p) * p * later()),
        )
        for T in (Float32, Float64, ComplexF64), (kernel, ordinary) in cases
            ws = PSWorkspace(d, 4, T)
            out = CTPS(T, d)
            p = CTPS(one(T), 1, d)
            calls = Ref(0)
            later() = (calls[] += 1; p.c[1] = T(5); p)
            expected = ordinary(p, later)
            p.c[1] = one(T)
            calls[] = 0
            kernel(ws, out, p, later)
            @test all(element(out, [n]) == element(expected, [n]) for n in 0:3)
            @test calls[] == 1
            @test ws.sp == 4 && !any(ws.inuse)
        end
    end
    @testset "Evaluate all arguments before an arithmetic error" begin
        ws = PSWorkspace(d, 4)
        p, out = CTPS(1.0, d), CTPS(7.0, d)
        q = CTPS(2.0, PSDesc(2, 3))
        calls = Ref(0)
        later() = (calls[] += 1; p)
        @test_throws DimensionMismatch p + q + later()
        @test calls[] == 1
        calls[] = 0
        @test_throws DimensionMismatch @tpsa ws out = p + q + later()
        @test calls[] == 1
        @test cst(out) == 7.0
        @test ws.sp == 4 && !any(ws.inuse)
    end
    @testset "Hold argument temporaries until the call and release on failure" begin
        ws = PSWorkspace(d, 4)
        p, out = CTPS(1.0, 1, d), CTPS(7.0, d)
        available_at_failure = Ref(0)
        fail() = (available_at_failure[] = ws.sp; error("last operand failed"))
        @test_throws ErrorException @tpsa ws out = (p*p) + (p*p) + fail()
        @test available_at_failure[] == 2
        @test cst(out) == 7.0
        @test ws.sp == 4 && !any(ws.inuse)
        small = PSWorkspace(d, 1)
        @test_throws ErrorException @tpsa small out = (p*p) + (p*p) + p
        @test small.sp == 1 && !any(small.inuse)
    end
    @testset "Successful calls retain and release argument temporaries" begin
        ws = PSWorkspace(d, 5)
        p, out = CTPS(1.0, 1, d), CTPS(Float64, d)
        @tpsa ws out = p^2 + p^2 + p^2 + p^2
        expected = 4*p^2
        @test all(element(out, [n]) == element(expected, [n]) for n in 0:3)
        @test ws.sp == 5 && !any(ws.inuse)
        @test all(iszero(buf.degree_mask[]) for buf in ws.bufs)
        trace = Int[]
        operand(i) = (push!(trace, i); p)
        @tpsa ws out = operand(1) + operand(2) + operand(3) + operand(4)
        @test trace == [1, 2, 3, 4]
        @test cst(out) == 4.0 && element(out, [1]) == 4.0 && ws.sp == 5
    end
end

timing_alloc!(ws, out, p, q, r, s) = (@tpsa ws out = p + q + r + s)
@testset "N-ary timing preserves zero allocations" begin
    d = PSDesc(1, 3)
    for T in (Float32, Float64, ComplexF64)
        ws = PSWorkspace(d, 4, T)
        p, out = CTPS(one(T), 1, d), CTPS(T, d)
        timing_alloc!(ws, out, p, p, p, p)
        @test cst(out) == T(4) && element(out, [1]) == T(4)
        @test (@allocated timing_alloc!(ws, out, p, p, p, p)) == 0
    end
end

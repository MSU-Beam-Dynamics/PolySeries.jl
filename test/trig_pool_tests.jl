using Test, PolySeries

@testset "Trigonometric domain errors preserve pool capacity" begin
    desc = PSDesc(1, 4)
    pool = desc._pools[Threads.threadid()]
    capacity = pool.sp
    out = CTPS(Float64, desc)
    fill!(out.c, 7.0)
    out.degree_mask[] = UInt64(0x1f)
    before = copy(out.c)
    ws = PSWorkspace(desc, 8)
    for center in (Inf, -Inf)
        p = CTPS(center, 1, desc)
        # Repetition beyond the pool capacity also checks the exhaustion path.
        for _ in 1:capacity+1
            for f! in (sin!, cos!, tan!)
                @test_throws DomainError f!(out, p)
                @test pool.sp == capacity
                @test out.c == before
                @test out.degree_mask[] == UInt64(0x1f)
            end
            for f in (sin, cos, tan)
                @test_throws DomainError f(p)
                @test pool.sp == capacity
            end
            @test_throws DomainError @tpsa ws out = sin(p) * p
            @test pool.sp == capacity
            @test ws.sp == length(ws.bufs)
        end
        for f! in (sin!, cos!, tan!)
            p = CTPS(center, 1, desc)
            snapshot = copy(p.c)
            mask = p.degree_mask[]
            @test_throws DomainError f!(p, p)
            @test pool.sp == capacity
            @test isequal(p.c, snapshot)
            @test p.degree_mask[] == mask
        end
        companion = CTPS(2.0, 1, desc)
        for alias in (:neither, :sin, :cos)
            s = alias == :sin ? p : out
            c = alias == :cos ? p : companion
            sbefore, cbefore = copy(s.c), copy(c.c)
            smask, cmask = s.degree_mask[], c.degree_mask[]
            @test_throws DomainError sincos!(s, c, p)
            @test pool.sp == capacity
            @test isequal(s.c, sbefore) && isequal(c.c, cbefore)
            @test s.degree_mask[] == smask && c.degree_mask[] == cmask
        end
    end
    @test length(unique(pool.avail[1:pool.sp])) == capacity
end

@testset "Trigonometric calls recover without allocations" begin
    desc = PSDesc(1, 4)
    pool = desc._pools[Threads.threadid()]
    capacity = pool.sp
    idx, held = PolySeries._ctps_pooled(Float64, desc)
    try
        PolySeries.copy!(held, CTPS(3.0, 1, desc))
        p = CTPS(0.3, 1, desc)
        out = CTPS(Float64, desc)
        for (f!, f) in ((sin!, sin), (cos!, cos), (tan!, tan), (sinh!, sinh), (cosh!, cosh))
            if f! in (sin!, cos!, tan!)
                @test_throws DomainError f!(out, CTPS(Inf, 1, desc))
            end
            expected = f(p)
            f!(out, p)
            @test (@allocated f!(out, p)) == 0
            @test all(element(out, [k]) ≈ element(expected, [k]) for k in 0:4)
            PolySeries.copy!(out, p)
            f!(out, out) # warm the alias path before measuring
            PolySeries.copy!(out, p)
            @test (@allocated f!(out, out)) == 0
            @test all(element(out, [k]) ≈ element(expected, [k]) for k in 0:4)
            @test pool.sp == capacity - 1
            @test cst(held) == 3.0 && element(held, [1]) == 1.0
        end
    finally
        PolySeries._pool_release!(idx, held, desc)
    end
    @test pool.sp == capacity
end

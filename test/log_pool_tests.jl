using Test, PolySeries

@testset "log! domain errors preserve pool capacity" begin
    desc = set_descriptor!(1, 5)
    pool = desc._pools[Threads.threadid()]
    capacity = pool.sp
    @test capacity > 0
    out = CTPS(7.0, 1)
    before = copy(out.c)
    before_mask = out.degree_mask[]
    ws = PSWorkspace(desc, 8)
    p = CTPS(-1.0, 1)
    # More failures than needed to exhaust the old four-slots-per-call leak.
    for _ in 1:10
        @test_throws DomainError log!(out, p)
        @test pool.sp == capacity
        @test out.c == before
        @test out.degree_mask[] == before_mask
        @test_throws DomainError @tpsa ws out = log(p)*p
        @test pool.sp == capacity
        @test ws.sp == 8
        @test out.c == before
    end
    @test length(unique(pool.avail[1:pool.sp])) == capacity
    # A valid call after failures still uses the pool and computes correctly.
    valid = CTPS(2.0, 1)
    log!(out, valid)
    @test (@allocated log!(out, valid)) == 0
    @test pool.sp == capacity
    for degree in 0:desc.order
        expected = degree == 0 ? log(2.0) : (-1.0)^(degree+1)/(degree*2.0^degree)
        @test element(out, [degree]) ≈ expected
    end
end

@testset "log! validates before touching caller-held pool slots" begin
    desc = set_descriptor!(2, 3)
    pool = desc._pools[Threads.threadid()]
    capacity = pool.sp
    idx, held = PolySeries._ctps_pooled(Float64, desc)
    @test idx != 0
    try
        PolySeries.copy!(held, CTPS(3.0, 1))
        out = CTPS(7.0)
        @test_throws DomainError log!(out, CTPS(-2.0))
        @test pool.sp == capacity - 1
        @test cst(held) == 3.0
        @test element(held, [1, 0]) == 1.0
        @test cst(out) == 7.0
        @test_throws ErrorException log!(out, CTPS(0.0))
        @test pool.sp == capacity - 1
    finally
        PolySeries._pool_release!(idx, held, desc)
    end
    @test pool.sp == capacity
end

@testset "log! preserves non-Float64 behavior" begin
    desc = PSDesc(1, 3)
    for T in (Float32, BigFloat)
        out = CTPS(T(7), desc)
        @test_throws DomainError log!(out, CTPS(-one(T), 1, desc))
        @test cst(out) == T(7)
    end
    # Negative real parts are valid when the coefficient type is complex.
    p = CTPS(ComplexF64(-1), 1, desc)
    out = CTPS(ComplexF64, desc)
    log!(out, p)
    expected = log(p)
    @test all(element(out, [d]) ≈ element(expected, [d]) for d in 0:3)
end

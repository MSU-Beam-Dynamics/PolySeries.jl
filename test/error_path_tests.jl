using Test, PolySeries

# Guard rails on the public API. Each `@test_broken` below is a Phase 1 target
# from the pre-release review: today it either throws the wrong exception type
# or does not throw at all. When the fix lands, Test reports "Unexpected Pass"
# for that line — replace it with the strict assertion written above it.

throws(::Type{E}, f) where E = try
    f()
    false
catch err
    err isa E
end

@testset "Constructor guards" begin
    d = PSDesc(2, 3)
    @test_throws ArgumentError CTPS(0.0, 0, d)
    @test_throws ArgumentError CTPS(0.0, 3, d)
    @test_throws ArgumentError CTPS(0.0, 1, PSDesc(1, 0))
    # C2: legacy form has no order check (BoundsError today) and throws
    # ErrorException for a bad variable index. Target: ArgumentError for both.
    # @test_throws ArgumentError CTPS(0.0, 1, 1, 0)
    @test_broken throws(ArgumentError, () -> CTPS(0.0, 1, 1, 0))
    @test throws(Exception, () -> CTPS(0.0, 1, 1, 0))
    # @test_throws ArgumentError CTPS(0.0, 3, 2, 3)
    @test_broken throws(ArgumentError, () -> CTPS(0.0, 3, 2, 3))
    @test throws(Exception, () -> CTPS(0.0, 3, 2, 3))
end

@testset "Accessor and evaluation guards" begin
    d = PSDesc(2, 3)
    p = CTPS(0.0, 1, d)
    @test_throws ArgumentError findindex(p, [1, 0, 0, 0])
    @test_throws ArgumentError findindex(p, [4, 0])
    @test_throws ArgumentError findindex(p, [-1, 0])
    @test_throws ArgumentError element(p, [2, 1, 0])      # degree prefix mismatch
    # Arity mismatch. Target: ArgumentError (ErrorException today).
    # @test_throws ArgumentError p(0.1)
    @test_broken throws(ArgumentError, () -> p(0.1))
    @test throws(Exception, () -> p(0.1))
    @test_throws DimensionMismatch p + CTPS(0.0, 1, PSDesc(2, 4))
    @test_throws DimensionMismatch mul!(CTPS(Float64, PSDesc(3, 3)), p, p)
end

@testset "Domain guards" begin
    d = PSDesc(2, 3)
    p = CTPS(0.0, 1, d)            # zero constant term
    q = CTPS(Float64, d)
    # Division by a series with zero constant term. Target: DomainError.
    # @test_throws DomainError inv(p)
    @test_broken throws(DomainError, () -> inv(p))
    @test throws(Exception, () -> inv(p))
    # @test_throws DomainError 1.0 / p
    @test_broken throws(DomainError, () -> 1.0 / p)
    @test throws(Exception, () -> 1.0 / p)
    # @test_throws DomainError CTPS(1.0, d) / p
    @test_broken throws(DomainError, () -> CTPS(1.0, d) / p)
    @test throws(Exception, () -> CTPS(1.0, d) / p)
    # @test_throws DomainError p / 0.0
    @test_broken throws(DomainError, () -> p / 0.0)
    @test throws(Exception, () -> p / 0.0)
    # Logarithm / square root at an invalid centre.
    # @test_throws DomainError log(p)
    @test_broken throws(DomainError, () -> log(p))
    @test throws(Exception, () -> log(p))
    @test_throws DomainError sqrt(p)                       # already DomainError
    @test_throws DomainError sqrt!(q, p)
    # @test_throws DomainError sqrt(CTPS(-1.0, 1, d))
    @test_broken throws(DomainError, () -> sqrt(CTPS(-1.0, 1, d)))
    @test throws(Exception, () -> sqrt(CTPS(-1.0, 1, d)))
    # Inverse trigonometric functions outside (-1, 1).
    # @test_throws DomainError asin(CTPS(1.0, 1, d))
    @test_broken throws(DomainError, () -> asin(CTPS(1.0, 1, d)))
    @test throws(Exception, () -> asin(CTPS(1.0, 1, d)))
    @test_broken throws(DomainError, () -> acos!(q, CTPS(-1.5, 1, d)))
    @test throws(Exception, () -> acos!(q, CTPS(-1.5, 1, d)))
    # Negative in-place power. Target: ArgumentError.
    # @test_throws ArgumentError pow!(q, p, -1)
    @test_broken throws(ArgumentError, () -> pow!(q, p, -1))
    @test throws(Exception, () -> pow!(q, p, -1))
    # Failed domain checks must not leak pool capacity (log_pool_tests covers log!).
    y = 1.0 + p
    out = CTPS(Float64, d)
    for _ in 1:3 * PolySeries.CTPS_POOL_SIZE
        try; sqrt!(out, p); catch; end
        try; asin!(out, CTPS(2.0, 1, d)); catch; end
    end
    sqrt!(out, y)
    @test element(out, [1, 0]) ≈ 0.5
end

@testset "Workspace double release" begin
    d = PSDesc(1, 2)
    ws = PSWorkspace(d, 2)
    t1 = borrow!(ws)
    t2 = borrow!(ws)
    release!(ws, t1)
    # C3: releasing the same slot twice must be rejected. Today it silently
    # pushes the slot index a second time, so the workspace believes both
    # slots are free while t2 is still live, and two borrows share a buffer.
    # @test_throws ArgumentError release!(ws, t1)
    @test_broken throws(ArgumentError, () -> release!(ws, t1))
    if ws.sp == length(ws.bufs)
        u1 = borrow!(ws)
        u2 = borrow!(ws)
        @test_broken u1.c !== u2.c
    end
    # A fresh workspace behaves.
    ws = PSWorkspace(d, 2)
    a = borrow!(ws); b = borrow!(ws)
    @test a.c !== b.c
    @test_throws ErrorException borrow!(ws)               # exhausted
    release!(ws, b); release!(ws, a)
    @test ws.sp == 2
    @test_throws ArgumentError release!(ws, CTPS(Float64, d))   # foreign CTPS
    @test_throws DimensionMismatch release!(ws, CTPS(Float64, PSDesc(1, 3)))
end

@testset "Composition aliasing and arity" begin
    d = PSDesc(2, 3)
    x = CTPS(0.0, 1, d); y = CTPS(0.0, 2, d)
    f = x + x^2
    # C4: result must not alias f or a member of g; today compose! zeroes the
    # shared buffer before reading it and returns silently.
    # @test_throws ArgumentError compose!(f, f, [x, y])
    @test_broken throws(ArgumentError, () -> (h = CTPS(f); compose!(h, h, [x, y])))
    # @test_throws ArgumentError compose!(g1, f, [g1, y])
    @test_broken throws(ArgumentError, () -> (g1 = CTPS(x); compose!(g1, f, [g1, y])))
    # Wrong number of substitution polynomials. Target: DimensionMismatch.
    # @test_throws DimensionMismatch compose(f, [x])
    @test_broken throws(DimensionMismatch, () -> compose(f, [x]))
    @test throws(Exception, () -> compose(f, [x]))
    @test_throws DimensionMismatch compose(f, [x, CTPS(0.0, 1, PSDesc(2, 4))])
end

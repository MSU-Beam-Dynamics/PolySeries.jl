using Test, PolySeries

# Guard rails on the public API: every failure mode a user can reach must throw
# a specific, documented exception type (see the Errors section of the API
# reference). Error types are part of the public contract.

@testset "Constructor guards" begin
    d = PSDesc(2, 3)
    @test_throws ArgumentError CTPS(0.0, 0, d)
    @test_throws ArgumentError CTPS(0.0, 3, d)
    @test_throws ArgumentError CTPS(0.0, 1, PSDesc(1, 0))
    # Legacy forms validate through the explicit-descriptor constructors.
    @test_throws ArgumentError CTPS(0.0, 1, 1, 0)
    @test_throws ArgumentError CTPS(0.0, 3, 2, 3)
    @test_throws ArgumentError CTPS(0.0, 0, 2, 3)
    legacy = CTPS(1.5, 2, 2, 3)
    @test cst(legacy) == 1.5 && element(legacy, [0, 1]) == 1.0
    @test get_descriptor() === legacy.desc          # documented side effect
end

@testset "Accessor and evaluation guards" begin
    d = PSDesc(2, 3)
    p = CTPS(0.0, 1, d)
    @test_throws ArgumentError findindex(p, [1, 0, 0, 0])
    @test_throws ArgumentError findindex(p, [4, 0])
    @test_throws ArgumentError findindex(p, [-1, 0])
    @test_throws ArgumentError element(p, [2, 1, 0])      # degree prefix mismatch
    @test_throws ArgumentError p(0.1)                     # arity mismatch
    @test_throws ArgumentError p(0.1, 0.2, 0.3)
    @test_throws DimensionMismatch p + CTPS(0.0, 1, PSDesc(2, 4))
    @test_throws DimensionMismatch mul!(CTPS(Float64, PSDesc(3, 3)), p, p)
    # mul! follows the LinearAlgebra convention and is the same function.
    out = CTPS(Float64, d)
    @test mul!(out, p, p) === out
    @test element(out, [2, 0]) == 1.0
end

@testset "Domain guards" begin
    d = PSDesc(2, 3)
    p = CTPS(0.0, 1, d)            # zero constant term
    q = CTPS(Float64, d)
    # Division by a series with zero constant term, or by a zero scalar.
    @test_throws DomainError inv(p)
    @test_throws DomainError 1.0 / p
    @test_throws DomainError CTPS(1.0, d) / p
    @test_throws DomainError p / 0.0
    @test_throws DomainError p / 0
    # Logarithm / square root at an invalid centre.
    @test_throws DomainError log(p)
    @test_throws DomainError log!(q, p)
    @test_throws DomainError sqrt(p)
    @test_throws DomainError sqrt!(q, p)
    @test_throws DomainError sqrt(CTPS(-1.0, 1, d))
    @test_throws DomainError sqrt!(q, CTPS(-1.0, 1, d))
    # Inverse trigonometric functions outside (-1, 1).
    @test_throws DomainError asin(CTPS(1.0, 1, d))
    @test_throws DomainError asin!(q, CTPS(1.0, 1, d))
    @test_throws DomainError acos(CTPS(-1.5, 1, d))
    @test_throws DomainError acos!(q, CTPS(-1.5, 1, d))
    # Negative in-place power is an argument error, not a domain error.
    @test_throws ArgumentError pow!(q, p, -1)
    @test_throws ArgumentError pow!(q, 1.0 + p, -2)
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
    # Releasing the same slot twice is rejected and leaves the stack intact,
    # so t2 stays live and cannot be handed out a second time.
    @test_throws ArgumentError release!(ws, t1)
    @test ws.sp == 1
    u1 = borrow!(ws)
    @test u1.c !== t2.c
    @test_throws ErrorException borrow!(ws)
    release!(ws, u1); release!(ws, t2)
    @test ws.sp == 2
    @test_throws ArgumentError release!(ws, t2)
    @test ws.sp == 2
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
    # The result must not alias f or a member of g; the traversal clears the
    # result before reading its inputs, so the buffers must be distinct.
    h = CTPS(f)
    @test_throws ArgumentError compose!(h, h, [x, y])
    @test element(h, [1, 0]) == 1.0 && element(h, [2, 0]) == 1.0   # untouched by the check
    g1 = CTPS(x)
    @test_throws ArgumentError compose!(g1, f, [g1, y])
    ws = CompositionWorkspace(d)
    @test_throws ArgumentError compose!(h, h, [x, y], ws)
    # Wrong number of substitution polynomials.
    @test_throws DimensionMismatch compose(f, [x])
    @test_throws DimensionMismatch compose!(CTPS(Float64, d), f, [x, y, x], ws)
    @test_throws DimensionMismatch compose(f, [x, CTPS(0.0, 1, PSDesc(2, 4))])
end

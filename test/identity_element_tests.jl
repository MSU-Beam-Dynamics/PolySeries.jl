using Test, PolySeries

@testset "zero, one and iszero" begin
    for (nv, order) in ((1, 0), (1, 4), (2, 3), (3, 2))
        desc = PSDesc(nv, order)
        p = order == 0 ? CTPS(0.7, desc) : 0.7 + 2.0 * CTPS(0.0, 1, desc)
        allc(q) = [element(q, [Int(desc.polymap.map[i, v]) for v in 2:nv + 1])
                   for i in 1:desc.N]

        z = zero(p)
        o = one(p)
        @test z isa CTPS{Float64} && o isa CTPS{Float64}
        @test z.desc === desc && o.desc === desc
        @test allc(z) == zeros(desc.N)
        @test cst(o) == 1.0 && allc(o) == [i == 1 ? 1.0 : 0.0 for i in 1:desc.N]
        # The source is neither read nor modified, and storage is not shared.
        @test allc(p) == allc(order == 0 ? CTPS(0.7, desc) : 0.7 + 2.0 * CTPS(0.0, 1, desc))
        @test z.c !== p.c && o.c !== p.c && z.c !== o.c

        # Algebraic identities.
        @test allc(p + z) == allc(p)
        @test allc(p * o) == allc(p)
        @test allc(p * z) == zeros(desc.N)
        @test allc(z + z) == zeros(desc.N)
        @test allc(o * o) == allc(o)

        @test iszero(z) && !iszero(o) && !iszero(p)
        @test iszero(p - p)
        @test iszero(zero(p) * p)
        # A set mask that cancels to zero must still report zero: the mask is
        # conservative, so the coefficients decide.
        cancelled = p - p
        @test cancelled.degree_mask[] != 0 || iszero(cancelled)
        @test iszero(cancelled)
        # iszero never reads inactive storage: poison every uninitialized slot.
        sparse = CTPS(Float64, desc)
        fill!(sparse.c, NaN)
        sparse.degree_mask[] = UInt64(0)
        @test iszero(sparse)
        if order >= 1
            lazy = CTPS(Float64, desc)
            fill!(lazy.c, NaN)
            lazy.c[1] = 0.0
            lazy.degree_mask[] = UInt64(1)      # only degree 0 is readable
            @test iszero(lazy)
            lazy.c[1] = 1e-30
            @test !iszero(lazy)
        end
    end

    # Coefficient type is preserved, including complex and extended precision.
    for T in (Float32, ComplexF64, BigFloat)
        desc = PSDesc(2, 2)
        q = CTPS(one(T), 1, desc)
        @test zero(q) isa CTPS{T} && one(q) isa CTPS{T}
        @test cst(one(q)) == one(T) && typeof(cst(one(q))) === T
        @test iszero(zero(q))
    end

    # The type-level forms are deliberately absent: CTPS{T} carries no descriptor.
    @test_throws MethodError zero(CTPS{Float64})
    @test_throws MethodError one(CTPS{Float64})

    # zero! is the in-place sibling and is unaffected.
    desc = PSDesc(2, 3)
    r = 1.0 + CTPS(0.0, 1, desc)
    @test !iszero(r)
    zero!(r)
    @test iszero(r) && r.degree_mask[] == UInt64(0)
end

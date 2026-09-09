using Test, PolySeries

@testset "Square root rejects singular expansion centers" begin
    desc = PSDesc(1, 3)
    for T in (Float64, ComplexF64)
        p = CTPS(zero(T), 1, desc)
        out = CTPS(T(7), desc)
        @test_throws DomainError sqrt(p)
        @test_throws DomainError sqrt!(out, p)
        @test cst(out) == T(7)
        @test_throws DomainError sqrt!(p, p)
        @test element(p, [1]) == one(T)
    end
end

@testset "Negative powers avoid intermediate overflow" begin
    desc = PSDesc(1, 3)
    @test cst(pow(CTPS(2.0, desc), -1024)) == 2.0^-1024
    @test cst(pow(CTPS(1.0, desc), typemin(Int))) == 1.0
    @test cst(pow(CTPS(-1.0, desc), typemin(Int))) == 1.0
    p = CTPS(2.0, 1, desc)
    q = pow(p, -3)
    @test [element(q, [k]) for k in 0:3] ≈ [1/8, -3/16, 3/16, -5/32]
end

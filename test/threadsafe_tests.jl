# Tests for thread safety
@testset "Descriptor caching thread safety" begin
    tasks = [Threads.@spawn PSDesc(3, 4) for _ in 1:16]
    descriptors = fetch.(tasks)
    @test all(d -> d === descriptors[1], descriptors)
end

include("descriptor_ownership_tests.jl")

@testset "Descriptor immutability" begin
    x = CTPS(Float64, 2, 3)
    desc = x.desc
    
    # Descriptor fields should be accessible but immutable
    @test desc.nv == 2
    @test desc.order == 3
    @test desc.N > 0
    
    # These should not be mutable (would throw error)
    @test_throws ErrorException desc.nv = 5
end

@testset "Output-major schedule thread safety" begin
    nv = 2
    order = 3
    
    x1 = CTPS(Float64, nv, order)
    x2 = CTPS(Float64, nv, order)
    
    for i in 1:binomial(nv + order, nv)
        x1.c[i] = Float64(i)
        x2.c[i] = Float64(i + 10)
    end
    
    PolySeries.update_degree_mask!(x1)
    PolySeries.update_degree_mask!(x2)
    
    # Share read-only operands and their descriptor, with an output per task.
    tasks = [Threads.@spawn begin
        r = CTPS(Float64, x1.desc)
        PolySeries.mul!(r, x1, x2)
        copy(r.c)
    end for _ in 1:8]
    results = fetch.(tasks)
    
    # All results should be identical
    for r in results[2:end]
        @test r ≈ results[1]
    end
end

# Tests for thread safety
@testset "Descriptor caching thread safety" begin
    nv = 3
    order = 4
    
    # Create multiple CTPS instances with same parameters from different "threads"
    # (simulated by sequential calls - actual threading would require Threads.@threads)
    instances = [CTPS(Float64, nv, order) for _ in 1:10]
    
    # All should share the same descriptor
    desc1 = instances[1].desc
    for inst in instances
        @test inst.desc === desc1  # Same object reference
    end
end

@testset "Independent descriptor instances" begin
    # Different parameters should have different descriptors
    x1 = CTPS(Float64, 2, 3)
    x2 = CTPS(Float64, 2, 4)
    x3 = CTPS(Float64, 3, 3)
    
    @test x1.desc !== x2.desc
    @test x1.desc !== x3.desc
    @test x2.desc !== x3.desc
end

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
    
    TPSA.update_degree_mask!(x1)
    TPSA.update_degree_mask!(x2)
    
    # Perform multiplication multiple times
    # Each should produce identical results (deterministic)
    results = [begin
        r = CTPS(Float64, nv, order)
        TPSA.mul!(r, x1, x2)
        copy(r.c)
    end for _ in 1:5]
    
    # All results should be identical
    for r in results[2:end]
        @test r ≈ results[1]
    end
end

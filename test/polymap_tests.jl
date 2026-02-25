# Tests for PolyMap functionality
@testset "decomposite function" begin
    # Test that decomposite returns Vector{Int}
    result = TPSA.decomposite(5, 3)
    @test eltype(result) == Int
    @test length(result) == 4  # dim + 1
    
    # Test basic cases
    @test TPSA.decomposite(0, 2) == [0, 0, 0]
    @test TPSA.decomposite(1, 2) == [1, 1, 0]
end

@testset "PolyMap construction" begin
    pm = TPSA.PolyMap(4, 3)
    @test pm.dim == 4
    @test pm.max_order == 3
    @test size(pm.map, 1) > 0
    @test size(pm.map, 2) == pm.dim + 1
end

@testset "getindexmap bounds checking" begin
    pm = TPSA.PolyMap(4, 3)
    n_rows = size(pm.map, 1)
    
    # Valid index
    @test length(TPSA.getindexmap(pm, 1)) == pm.dim + 1
    @test length(TPSA.getindexmap(pm, n_rows)) == pm.dim + 1
    
    # Invalid indices
    @test_throws ErrorException TPSA.getindexmap(pm, 0)
    @test_throws ErrorException TPSA.getindexmap(pm, n_rows + 1)
end

@testset "PolyMap view allocation" begin
    pm = TPSA.PolyMap(3, 4)
    result = TPSA.getindexmap(pm, 5)
    
    # Should return a view (SubArray)
    @test result isa SubArray
end

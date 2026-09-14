# Tests for PolyMap functionality
@testset "decomposite input validation" begin
    for dim in (1, 2, 6), n in (-1, -2, typemin(Int), typemax(Int))
        @test_throws ArgumentError decomposite(n, dim)
    end
    for dim in (-1, 0, typemin(Int), typemax(Int)), n in (0, 1)
        @test_throws ArgumentError decomposite(n, dim)
    end
end

@testset "decomposite valid-index round trips and boundaries" begin
    for dim in (1, 2, 3, 6, 16, 128), n in 0:63
        exponents = decomposite(n, dim)
        @test length(exponents) == dim + 1 && all(>=(0), exponents)
        @test exponents[1] == sum(exponents[2:end])
        # Independent BigInt ranking formula, rather than a second search.
        remaining = big(exponents[1])
        rank = big(0)
        for j in 1:dim
            i = dim - j + 1
            rank += binomial(remaining + i - 1, i)
            remaining -= exponents[j + 1]
        end
        @test rank == n
    end
    for n in (typemax(Int)-2, typemax(Int)-1)
        @test decomposite(n, 1) == [n, n]
    end
    # Oversized multivariate intermediates must throw, never wrap to a
    # negative or unrelated exponent vector.
    @test_throws OverflowError decomposite(typemax(Int)-1, 64)
end

@testset "decomposite function" begin
    # Test that decomposite returns Vector{Int}
    result = PolySeries.decomposite(5, 3)
    @test eltype(result) == Int
    @test length(result) == 4  # dim + 1
    
    # Test basic cases
    @test PolySeries.decomposite(0, 2) == [0, 0, 0]
    @test PolySeries.decomposite(1, 2) == [1, 1, 0]
end

@testset "PolyMap construction" begin
    pm = PolySeries.PolyMap(4, 3)
    @test pm.dim == 4
    @test pm.max_order == 3
    @test size(pm.map, 1) > 0
    @test size(pm.map, 2) == pm.dim + 1
end

@testset "getindexmap bounds checking" begin
    pm = PolySeries.PolyMap(4, 3)
    n_rows = size(pm.map, 1)
    
    # Valid index
    @test length(PolySeries.getindexmap(pm, 1)) == pm.dim + 1
    @test length(PolySeries.getindexmap(pm, n_rows)) == pm.dim + 1
    
    # Invalid indices
    @test_throws ErrorException PolySeries.getindexmap(pm, 0)
    @test_throws ErrorException PolySeries.getindexmap(pm, n_rows + 1)
end

@testset "PolyMap view allocation" begin
    pm = PolySeries.PolyMap(3, 4)
    result = PolySeries.getindexmap(pm, 5)
    
    # Should return a view (SubArray)
    @test result isa SubArray
end

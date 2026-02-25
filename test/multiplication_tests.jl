# Tests for multiplication correctness
@testset "Basic multiplication" begin
    # Test (1 + 2x + 3y) * (4 + 5x + 6y)
    # = 4 + 13x + 18y + 10x^2 + 27xy + 18y^2
    nv = 2
    order = 2

    x1 = CTPS(Float64, nv, order)
    x1.c[1] = 1.0  # constant
    x1.c[2] = 2.0  # x coefficient
    x1.c[3] = 3.0  # y coefficient

    x2 = CTPS(Float64, nv, order)
    x2.c[1] = 4.0  # constant
    x2.c[2] = 5.0  # x coefficient
    x2.c[3] = 6.0  # y coefficient

    TPSA.update_degree_mask!(x1)
    TPSA.update_degree_mask!(x2)

    result = CTPS(Float64, nv, order)
    TPSA.mul!(result, x1, x2)

    @test result.c[1] ≈ 4.0   # constant
    @test result.c[2] ≈ 13.0  # x
    @test result.c[3] ≈ 18.0  # y
    
    # Find second-order terms
    desc = result.desc
    for i in 1:desc.N
        exp_vec = TPSA.getindexmap(desc.polymap, i)
        if exp_vec[1] == 2
            if exp_vec[2] == 2 && exp_vec[3] == 0  # x^2
                @test result.c[i] ≈ 10.0
            elseif exp_vec[2] == 1 && exp_vec[3] == 1  # xy
                @test result.c[i] ≈ 27.0
            elseif exp_vec[2] == 0 && exp_vec[3] == 2  # y^2
                @test result.c[i] ≈ 18.0
            end
        end
    end
end

@testset "Sparse multiplication" begin
    nv = 3
    order = 5

    x1 = CTPS(Float64, nv, order)
    x2 = CTPS(Float64, nv, order)
    x1.c[1] = 1.0
    x1.c[2] = 2.0
    x2.c[1] = 3.0
    x2.c[3] = 4.0
    
    TPSA.update_degree_mask!(x1)
    TPSA.update_degree_mask!(x2)

    out = CTPS(Float64, nv, order)
    TPSA.mul!(out, x1, x2)

    # Expected: (1 + 2x)(3 + 4y) = 3 + 6x + 4y + 8xy
    @test out.c[1] ≈ 3.0  # constant
    @test out.c[2] ≈ 6.0  # x coefficient
    @test out.c[3] ≈ 4.0  # y coefficient
end

@testset "Dense multiplication" begin
    nv = 3
    order = 5
    N = binomial(nv + order, nv)

    x1 = CTPS(Float64, nv, order)
    x2 = CTPS(Float64, nv, order)
    
    # Fill with sequential values
    for i in 1:N
        x1.c[i] = Float64(i)
        x2.c[i] = Float64(100 + i)
    end
    
    TPSA.update_degree_mask!(x1)
    TPSA.update_degree_mask!(x2)

    result = CTPS(Float64, nv, order)
    TPSA.mul!(result, x1, x2)

    # Result should be non-zero and within reasonable range
    @test !all(iszero, result.c)
    @test all(isfinite, result.c)
end

@testset "Complex multiplication" begin
    nv = 2
    order = 3
    
    # Create complex TPSA
    x1 = CTPS(ComplexF64, nv, order)
    x2 = CTPS(ComplexF64, nv, order)
    
    x1.c[1] = 1.0 + 2.0im
    x1.c[2] = 3.0 + 4.0im
    x2.c[1] = 5.0 + 6.0im
    x2.c[2] = 7.0 + 8.0im
    
    TPSA.update_degree_mask!(x1)
    TPSA.update_degree_mask!(x2)
    
    result = CTPS(ComplexF64, nv, order)
    TPSA.mul!(result, x1, x2)
    
    # (1+2i)(5+6i) = 5+6i+10i+12i² = 5+16i-12 = -7+16i
    @test real(result.c[1]) ≈ -7.0
    @test imag(result.c[1]) ≈ 16.0
end

@testset "Multiplication operator" begin
    nv = 2
    order = 2
    
    x = CTPS(0.0, 1, nv, order)
    y = CTPS(0.0, 2, nv, order)
    
    # Test * operator
    result = x * y
    
    # x * y should give coefficient at xy term
    desc = result.desc
    for i in 1:desc.N
        exp_vec = TPSA.getindexmap(desc.polymap, i)
        if exp_vec[1] == 2 && exp_vec[2] == 1 && exp_vec[3] == 1  # xy term
            @test result.c[i] ≈ 1.0
        end
    end
end

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

    PolySeries.update_degree_mask!(x1)
    PolySeries.update_degree_mask!(x2)

    result = CTPS(Float64, nv, order)
    PolySeries.mul!(result, x1, x2)

    @test result.c[1] ≈ 4.0   # constant
    @test result.c[2] ≈ 13.0  # x
    @test result.c[3] ≈ 18.0  # y
    
    # Find second-order terms
    desc = result.desc
    for i in 1:desc.N
        exp_vec = PolySeries.getindexmap(desc.polymap, i)
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
    
    PolySeries.update_degree_mask!(x1)
    PolySeries.update_degree_mask!(x2)

    out = CTPS(Float64, nv, order)
    PolySeries.mul!(out, x1, x2)

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
    
    PolySeries.update_degree_mask!(x1)
    PolySeries.update_degree_mask!(x2)

    result = CTPS(Float64, nv, order)
    PolySeries.mul!(result, x1, x2)

    # Result should be finite throughout
    @test all(isfinite, result.c)

    # Verify several exact coefficients:
    #   constant = x1.c[1] * x2.c[1] = 1 * 101 = 101
    #   x1 coeff (idx 2): x1.c[1]*x2.c[2] + x1.c[2]*x2.c[1] = 1*102 + 2*101 = 304
    #   x2 coeff (idx 3): x1.c[1]*x2.c[3] + x1.c[3]*x2.c[1] = 1*103 + 3*101 = 406
    #   x3 coeff (idx 4): x1.c[1]*x2.c[4] + x1.c[4]*x2.c[1] = 1*104 + 4*101 = 508
    @test result.c[1] ≈ 101.0
    @test result.c[2] ≈ 304.0
    @test result.c[3] ≈ 406.0
    @test result.c[4] ≈ 508.0
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
    
    PolySeries.update_degree_mask!(x1)
    PolySeries.update_degree_mask!(x2)
    
    result = CTPS(ComplexF64, nv, order)
    PolySeries.mul!(result, x1, x2)
    
    # (1+2i)(5+6i) = 5+6i+10i+12i² = 5+16i-12 = -7+16i
    @test real(result.c[1]) ≈ -7.0
    @test imag(result.c[1]) ≈ 16.0
end

@testset "Multiplication operator" begin
    nv = 2
    order = 2
    
    x = CTPS(0.0, 1, nv, order)
    y = CTPS(0.0, 2, nv, order)
    
    # x * y = xy;  all other terms must be zero
    result = x * y
    
    # Read through the mask-aware accessor: coefficient blocks whose degree
    # bit is clear are mathematically zero but hold uninitialized memory.
    desc = result.desc
    for i in 1:desc.N
        exp_vec = PolySeries.getindexmap(desc.polymap, i)
        coeff = element(result, collect(Int, exp_vec))
        degree = exp_vec[1]
        e1, e2 = exp_vec[2], exp_vec[3]
        if degree == 2 && e1 == 1 && e2 == 1          # xy term
            @test coeff ≈ 1.0
        elseif degree == 0                              # constant
            @test coeff ≈ 0.0  atol=1e-15
        elseif degree == 1                              # linear terms
            @test coeff ≈ 0.0  atol=1e-15
        elseif degree == 2 && (e1 == 2 || e2 == 2)    # pure-square terms
            @test coeff ≈ 0.0  atol=1e-15
        end
    end
end

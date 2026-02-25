# Tests for index correctness in multiplication
@testset "Simple 2-variable multiplication" begin
    # Test (1 + x + y) * (1 + x + y)
    nv = 2
    order = 3
    x = CTPS(0.0, 1, nv, order)  # x variable
    y = CTPS(0.0, 2, nv, order)  # y variable
    one = CTPS(1.0, nv, order)   # constant 1

    a = one + x + y
    b = one + x + y
    c = a * b

    # Should be: 1 + 2x + 2y + x^2 + 2xy + y^2
    @test c.c[1] ≈ 1.0  # constant
    @test c.c[2] ≈ 2.0  # x coefficient
    @test c.c[3] ≈ 2.0  # y coefficient

    # Check second-order terms
    desc = c.desc
    for i in 1:desc.N
        exp_vec = TPSA.getindexmap(desc.polymap, i)
        degree = exp_vec[1]
        if degree == 2
            if exp_vec[2] == 2 && exp_vec[3] == 0  # x^2
                @test c.c[i] ≈ 1.0
            elseif exp_vec[2] == 1 && exp_vec[3] == 1  # xy
                @test c.c[i] ≈ 2.0
            elseif exp_vec[2] == 0 && exp_vec[3] == 2  # y^2
                @test c.c[i] ≈ 1.0
            end
        end
    end
end

@testset "Product of different terms" begin
    # Test (2 + 3x) * (4 + 5y)
    nv = 2
    order = 3
    x = CTPS(0.0, 1, nv, order)
    y = CTPS(0.0, 2, nv, order)
    one = CTPS(1.0, nv, order)

    a = 2*one + 3*x
    b = 4*one + 5*y
    c = a * b

    # Should be: 8 + 12x + 10y + 15xy
    @test c.c[1] ≈ 8.0   # constant
    @test c.c[2] ≈ 12.0  # x coefficient
    @test c.c[3] ≈ 10.0  # y coefficient

    desc = c.desc
    for i in 1:desc.N
        exp_vec = TPSA.getindexmap(desc.polymap, i)
        degree = exp_vec[1]
        if degree == 2 && exp_vec[2] == 1 && exp_vec[3] == 1  # xy
            @test c.c[i] ≈ 15.0
        end
    end
end

@testset "Three-variable multiplication" begin
    # Test (1 + x + y + z)^2
    nv = 3
    order = 4
    x = CTPS(0.0, 1, nv, order)
    y = CTPS(0.0, 2, nv, order)
    z = CTPS(0.0, 3, nv, order)
    one = CTPS(1.0, nv, order)

    a = one + x + y + z
    c = a * a

    # Should be: 1 + 2x + 2y + 2z + x^2 + y^2 + z^2 + 2xy + 2xz + 2yz
    @test c.c[1] ≈ 1.0  # constant
    @test c.c[2] ≈ 2.0  # x coefficient
    @test c.c[3] ≈ 2.0  # y coefficient
    @test c.c[4] ≈ 2.0  # z coefficient

    desc = c.desc
    for i in 1:desc.N
        exp_vec = TPSA.getindexmap(desc.polymap, i)
        degree = exp_vec[1]
        if degree == 2
            if exp_vec[2] == 2  # x^2
                @test c.c[i] ≈ 1.0
            elseif exp_vec[3] == 2  # y^2
                @test c.c[i] ≈ 1.0
            elseif exp_vec[4] == 2  # z^2
                @test c.c[i] ≈ 1.0
            elseif exp_vec[2] == 1 && exp_vec[3] == 1  # xy
                @test c.c[i] ≈ 2.0
            elseif exp_vec[2] == 1 && exp_vec[4] == 1  # xz
                @test c.c[i] ≈ 2.0
            elseif exp_vec[3] == 1 && exp_vec[4] == 1  # yz
                @test c.c[i] ≈ 2.0
            end
        end
    end
end

# Tests for type stability
@testset "TPSADesc type stability" begin
    x = CTPS(1.0, 1, 2, 3)
    desc = x.desc

    # Check that exp_to_idx is a concrete type
    @test isconcretetype(typeof(desc.exp_to_idx))
    
    # Check Dict type parameters
    @test keytype(desc.exp_to_idx) != Any
    # * Changed to Int32 based on ctps.jl line 136 - Kelly
    @test valtype(desc.exp_to_idx) == Int32 # Int
end

@testset "CTPS type stability" begin
    nv = 3
    order = 4
    
    # Float64 CTPS
    x_float = CTPS(Float64, nv, order)
    @test eltype(x_float.c) == Float64
    @test isconcretetype(typeof(x_float))
    
    # ComplexF64 CTPS
    x_complex = CTPS(ComplexF64, nv, order)
    @test eltype(x_complex.c) == ComplexF64
    @test isconcretetype(typeof(x_complex))
end

@testset "Multiplication type stability" begin
    nv = 2
    order = 3
    
    x1 = CTPS(Float64, nv, order)
    x2 = CTPS(Float64, nv, order)
    
    x1.c[1] = 1.0
    x2.c[1] = 2.0
    
    TPSA.update_degree_mask!(x1)
    TPSA.update_degree_mask!(x2)
    
    # Test type stability of mul! operation
    result = CTPS(Float64, nv, order)
    @inferred TPSA.mul!(result, x1, x2)
end

@testset "Schedule type stability" begin
    x = CTPS(1.0, 2, 3)
    desc = x.desc
    
    # Check schedule types
    @test eltype(desc.mul[1].ivalues) == Int32
    @test eltype(desc.mul[1].jidx) == Int32
    @test eltype(desc.mul[1].kidx) == Int32
    
    @test eltype(desc.mul_output[1].kvalues) == Int32
    @test eltype(desc.mul_output[1].iidx) == Int32
    @test eltype(desc.mul_output[1].jidx) == Int32
end

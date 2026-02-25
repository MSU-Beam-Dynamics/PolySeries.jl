using Test
using TPSA

@testset "TPSA.jl" begin
    @testset "PolyMap" begin
        include("polymap_tests.jl")
    end

    @testset "Index Mapping" begin
        include("index_tests.jl")
    end

    @testset "Multiplication" begin
        include("multiplication_tests.jl")
    end

    @testset "Type Stability" begin
        include("type_stability_tests.jl")
    end

    @testset "Thread Safety" begin
        include("threadsafe_tests.jl")
    end
end

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

    @testset "Math Functions" begin
        include("mathfunc_tests.jl")
    end

    @testset "Arithmetic Accuracy" begin
        include("arithmetic_tests.jl")
    end

    @testset "Extension: TPSAEnzymeExt" begin
        try
            include("ext_enzyme_test.jl")
        catch e
            @warn "Skipping TPSAEnzymeExt tests (Enzyme unavailable): $e"
        end
    end
end

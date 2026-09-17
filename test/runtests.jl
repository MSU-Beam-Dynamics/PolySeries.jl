using Test
using PolySeries

# verbose: print every top-level testset with its own timing, so a slow section
# (Enzyme compilation, the example scripts) is visible without a failure.
@testset verbose = true "PolySeries.jl" begin
    @testset "PolyMap" begin
        include("polymap_tests.jl")
    end

    @testset "Index Mapping" begin
        include("index_tests.jl")
    end

    @testset "Multiplication" begin
        include("multiplication_tests.jl")
        include("active_mul_schedule_tests.jl")
    end

    @testset "Multiplication Reference" begin
        include("mul_reference_tests.jl")
        include("square_kernel_tests.jl")
        include("output_product_tests.jl")
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

    @testset "Complex Branches" begin
        include("complex_branch_tests.jl")
    end

    @testset "Real Inverse Trig Accuracy" begin
        include("real_inverse_trig_tests.jl")
    end

    @testset "Math Aliasing" begin
        include("math_alias_tests.jl")
    end

    @testset "Logarithm Pool Safety" begin
        include("log_pool_tests.jl")
    end

    @testset "Trigonometric Pool Safety" begin
        include("trig_pool_tests.jl")
    end

    @testset "Order Limits" begin
        include("order_limits_tests.jl")
    end

    @testset "Release edge cases" begin
        include("release_edge_tests.jl")
    end

    @testset "Error Paths" begin
        include("error_path_tests.jl")
    end

    @testset "Arithmetic Accuracy" begin
        include("arithmetic_tests.jl")
        include("identity_element_tests.jl")
    end

    @testset "Degree Mask Regression" begin
        include("degree_mask_tests.jl")
    end

    @testset "TPSA Macro" begin
        include("macro_tests.jl")
    end

    @testset "Composition" begin
        include("composition_tests.jl")
        include("performance_strategy_tests.jl")
        include("translation_reference_tests.jl")
    end

    @testset "Extension: PolySeriesEnzymeExt" begin
        # Enzyme is a required test extra. Import/compiler failures must fail
        # the suite rather than silently skip differentiation regressions.
        include("ext_enzyme_test.jl")
    end

    # Last: the Enzyme example reuses the rules compiled by the extension tests.
    @testset "Documentation Examples" begin
        include("documentation_examples_tests.jl")
    end
end

# Tests for the PolySeriesEnzymeExt package extension.
# Enzyme is a required dependency; failures must not be skipped by the runner.

using Enzyme

include("enzyme_zero_tests.jl")
include("enzyme_prebuilt_tests.jl")
include("enzyme_descriptor_switch_tests.jl")

@testset "Enzyme with an explicitly owned descriptor" begin
    desc = PSDesc(1, 4)
    f = a -> cst(exp(CTPS(a, 1, desc)))
    set_descriptor!(2, 3)
    for a in (0.0, 0.5)
        @test Enzyme.gradient(Reverse, f, a)[1] ≈ exp(a)
    end
    clear_descriptor!()
    @test Enzyme.gradient(Reverse, f, 0.0)[1] ≈ 1.0
    @test Enzyme.autodiff(Forward, f, Duplicated(0.0, 1.0))[1] ≈ 1.0
end

@testset "PolySeriesEnzymeExt — extension loads" begin
    ext = Base.get_extension(PolySeries, :PolySeriesEnzymeExt)
    @test ext !== nothing
end

@testset "PolySeriesEnzymeExt — inactive_type rules" begin
    @test Enzyme.EnzymeRules.inactive_type(PolySeries.PSDesc)     == true
    @test Enzyme.EnzymeRules.inactive_type(PolySeries.DescPool)     == true
    @test Enzyme.EnzymeRules.inactive_type(PolySeries.PolyMap)      == true
    @test Enzyme.EnzymeRules.inactive_type(PolySeries.MulSchedule2D) == true
    @test Enzyme.EnzymeRules.inactive_type(PolySeries.CompPlan)     == true
    @test Enzyme.EnzymeRules.inactive_type(PolySeries.DescriptorRegistry) == true
end

@testset "PolySeriesEnzymeExt — Enzyme.gradient through CTPS (single variable)" begin
    set_descriptor!(1, 4)

    # d/dx₀ exp(x₀) = exp(x₀)
    g = Enzyme.gradient(Reverse, x0 -> cst(exp(CTPS(x0, 1))), 1.0)
    @test abs(g[1] - exp(1.0)) < 1e-12

    # d/dx₀ sin(x₀) = cos(x₀)
    g = Enzyme.gradient(Reverse, x0 -> cst(sin(CTPS(x0, 1))), 0.7)
    @test abs(g[1] - cos(0.7)) < 1e-12

    # d/dx₀ element([2]) of exp(x₀ + δ) = exp(x₀)/1! = exp(x₀)
    g = Enzyme.gradient(Reverse, x0 -> element(exp(CTPS(x0, 1)), [1]), 0.5)
    @test abs(g[1] - exp(0.5)) < 1e-12
end

@testset "PolySeriesEnzymeExt — Enzyme.gradient through CTPS (multi-variable)" begin
    set_descriptor!(2, 3)

    # ∂/∂x₀ [sin(x)*cos(y) + exp(x)] at (x₀,y₀) = cos(x₀)cos(y₀) + exp(x₀)
    x0, y0 = 0.7, 0.5
    function f_mv(x0::Float64)
        x = CTPS(x0, 1)
        y = CTPS(y0, 2)
        return cst(sin(x) * cos(y) + exp(x))
    end
    g = Enzyme.gradient(Reverse, f_mv, x0)
    exact = cos(x0) * cos(y0) + exp(x0)
    @test abs(g[1] - exact) < 1e-10
end

@testset "Composition retains derivative paths at zero" begin
    set_descriptor!(1, 3)
    function composition_coefficient(a)
        x = CTPS(0.0, 1)
        f = a*x^2 + x
        return cst(compose(f, [CTPS(2.0, 1)]))
    end
    function composition_center(a)
        x = CTPS(0.0, 1)
        return cst(compose(x + x^2, [CTPS(a, 1)]))
    end
    for a in (0.0, 0.5)
        @test Enzyme.gradient(Reverse, composition_coefficient, a)[1] ≈ 4.0
        @test Enzyme.gradient(Reverse, composition_center, a)[1] ≈ 1 + 2a
    end
end

@testset "Explicit composition workspace is bypassed during AD" begin
    set_descriptor!(1, 3)
    ws = CompositionWorkspace(get_descriptor())
    function composition_workspace_derivative(a)
        x = CTPS(0.0, 1)
        out = CTPS(Float64)
        compose!(out, a*x^2 + x, [CTPS(2.0, 1)], ws)
        return cst(out)
    end
    @test Enzyme.gradient(Reverse, composition_workspace_derivative, 0.0)[1] ≈ 4.0
    @test !any(ws.needed)
end

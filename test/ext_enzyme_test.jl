# Tests for the TPSAEnzymeExt package extension.
# This file is only included when Enzyme is available (see runtests.jl).

using Enzyme

@testset "TPSAEnzymeExt — extension loads" begin
    ext = Base.get_extension(TPSA, :TPSAEnzymeExt)
    @test ext !== nothing
end

@testset "TPSAEnzymeExt — inactive_type rules" begin
    @test Enzyme.EnzymeRules.inactive_type(TPSA.TPSADesc)     == true
    @test Enzyme.EnzymeRules.inactive_type(TPSA.DescPool)     == true
    @test Enzyme.EnzymeRules.inactive_type(TPSA.PolyMap)      == true
    @test Enzyme.EnzymeRules.inactive_type(TPSA.MulSchedule2D) == true
    @test Enzyme.EnzymeRules.inactive_type(TPSA.CompPlan)     == true
end

@testset "TPSAEnzymeExt — Enzyme.gradient through CTPS (single variable)" begin
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

@testset "TPSAEnzymeExt — Enzyme.gradient through CTPS (multi-variable)" begin
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

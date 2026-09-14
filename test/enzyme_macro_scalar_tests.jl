using Test, PolySeries, Enzyme

@testset "Enzyme scalar macro semantics" begin
    d = PSDesc(1, 3)
    function root_power(a)
        ws = PSWorkspace(d, 4)
        out = CTPS(Float64, d)
        @tpsa ws out = a^-1
        cst(out)
    end
    function nested_power(a)
        ws = PSWorkspace(d, 4)
        out = CTPS(Float64, d)
        x = CTPS(0.0, 1, d)
        @tpsa ws out = (a^-1)*x
        element(out, [1])
    end
    function root_difference(a)
        ws = PSWorkspace(d, 4)
        out = CTPS(Float64, d)
        @tpsa ws out = a - 0.5
        cst(out)
    end
    for a in (0.5, 2.0), f in (root_power, nested_power, root_difference)
        expected = f === root_difference ? 1.0 : -inv(a*a)
        @test Enzyme.autodiff(Forward, f, Duplicated(a, 1.0))[1] ≈ expected
        @test Enzyme.gradient(Reverse, f, a)[1] ≈ expected
    end
end

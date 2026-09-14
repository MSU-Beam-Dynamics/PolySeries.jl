using Test, PolySeries, Enzyme

@testset "Enzyme macro operand timing" begin
    d = PSDesc(1, 3)
    function nary(a)
        ws = PSWorkspace(d, 4)
        p, out = CTPS(a, 1, d), CTPS(Float64, d)
        later() = (p.c[1] = 3a; p)
        @tpsa ws out = p + 1 + later()
        cst(out)
    end
    function parenthesized(a)
        ws = PSWorkspace(d, 4)
        p, out = CTPS(a, 1, d), CTPS(Float64, d)
        later() = (p.c[1] = 3a; p)
        @tpsa ws out = (p + 1) + later()
        cst(out)
    end
    for (f, slope) in ((nary, 6.0), (parenthesized, 4.0)), a in (0.0, 0.5)
        @test f(a) == 1 + slope*a
        # Workspace metadata is constant while its coefficient buffers are
        # active. Runtime activity handles that mixture in reverse mode.
        # Forward-mode compiler failures are preserved separately in audit/.
        @test Enzyme.gradient(Enzyme.set_runtime_activity(Reverse), f, a)[1] == slope
    end
end

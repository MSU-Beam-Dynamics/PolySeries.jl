using Test, PolySeries

@testset "Documented macro expressions" begin
    root = dirname(@__DIR__)
    for path in ("README.md", "docs/src/index.md", "docs/src/tutorial.md", "src/macro.jl")
        source = read(joinpath(root, path), String)
        sandbox = Module(gensym(:MacroExample))
        setup = """
            using PolySeries
            desc = set_descriptor!(4, 6)
            ws = PSWorkspace(desc, 20)
            x1 = x = CTPS(0.0, 1); x2 = y = CTPS(0.0, 2)
            x3 = z = CTPS(0.0, 3)
            nx = CTPS(Float64, desc)
            θ = 2π * 0.205
            """
        Base.include_string(sandbox, setup)
        # Run complete tutorial/docstring blocks; the other snippets use
        # the context above. Read source so documentation changes are tested.
        code = if endswith(path, "tutorial.md")
            section = split(source, "## 8. The `@tpsa` Macro")[2]
            match(r"```julia\n(.*?)```"s, section).captures[1]
        elseif endswith(path, ".jl")
            match(r"```julia\n(.*?)```"s, source).captures[1]
        else
            match(r"^@tpsa .*"m, source).match
        end
        @test begin
            Base.include_string(sandbox, code, path)
            true
        end
        result = Base.invokelatest(getfield, sandbox, (endswith(path, "tutorial.md") || endswith(path, ".jl")) ? :nx1 : :nx)
        θ = 2π * 0.205
        for (ind, expected) in (([1,0,0,0], cos(θ)), ([0,1,0,0], sin(θ)),
                                ([2,0,0,0], sin(θ)), ([0,0,2,0], -sin(θ)))
            @test element(result, ind[1:result.desc.nv]) ≈ expected
        end
        ws = Base.invokelatest(getfield, sandbox, :ws)
        @test ws.sp == length(ws.bufs)
    end
end

@testset "Scalar unary dispatch" begin
    desc = set_descriptor!(1, 4)
    ws = PSWorkspace(desc, 8)
    x = CTPS(0.0, 1)
    out = CTPS(Float64, desc)
    @test begin
        @tpsa ws out = cos(0.2)*x
        element(out, [1]) ≈ cos(0.2)
    end
    @test ws.sp == 8
    for f in (:sin, :cos, :exp, :log, :sqrt, :sinh, :cosh)
        # Expand each supported function by name, just as user code does.
        kernel = Core.eval(@__MODULE__, :( (ws, out, x, a) -> (@tpsa ws out = $f(a)*x) ))
        Base.invokelatest(kernel, ws, out, x, 0.2)
        @test element(out, [1]) ≈ getfield(Base, f)(0.2)
        @test ws.sp == 8
        # The same lowering must still dispatch polynomial arguments in place.
        Base.invokelatest(kernel, ws, out, x, CTPS(0.2, 1, desc))
        expected = getfield(Base, f)(CTPS(0.2, 1, desc))*x
        @test all(element(out, [n]) ≈ element(expected, [n]) for n in 0:4)
        @test ws.sp == 8
    end
end

@testset "Macro releases temporaries after errors" begin
    desc = set_descriptor!(1, 4)
    ws = PSWorkspace(desc, 8)
    x = CTPS(0.0, 1)
    out = CTPS(Float64, desc)
    held = borrow!(ws)
    PolySeries.copy!(held, x)
    for _ in 1:3
        @test_throws DomainError @tpsa ws out = (x*x + x)*log(-1.0)
        @test ws.sp == 7
        @test element(held, [1]) == 1.0
        @test length(unique(ws.avail[1:ws.sp])) == ws.sp
    end
    release!(ws, held)
    @test ws.sp == 8

    tiny = PSWorkspace(desc, 2)
    @test_throws ErrorException @tpsa tiny out = (x*x)*(x*x) + (x*x)*(x*x)
    @test tiny.sp == 2
    @test length(unique(tiny.avail)) == 2
end

macro_rotation!(ws, out, x, y, z, θ) =
    @tpsa ws out = cos(θ)*x + sin(θ)*(y + x^2 - z^2)

@testset "Workspace reuse and allocation" begin
    desc = set_descriptor!(3, 4)
    ws = PSWorkspace(desc, 8)
    x = CTPS(0.0, 1); y = CTPS(0.0, 2); z = CTPS(0.0, 3)
    out = CTPS(Float64, desc)
    macro_rotation!(ws, out, x, y, z, 0.2)
    @test (@allocated macro_rotation!(ws, out, x, y, z, 0.2)) == 0
    @test ws.sp == 8
    calls = Ref(0)
    workspace() = (calls[] += 1; ws)
    @tpsa workspace() out = x*x + y*y
    @test calls[] == 1
    @test ws.sp == 8
    @test_throws ErrorException @tpsa ws out = (x*x)*log(x)
    @test ws.sp == 8
    wrong = CTPS(0.0, 1, PSDesc(1, 4))
    @test_throws DimensionMismatch @tpsa ws out = x*x + wrong
    @test ws.sp == 8
    @test all(p.degree_mask[] == 0 for p in ws.bufs)
    # Borrow every slot simultaneously: detect duplicates as well as leaks.
    slots = [borrow!(ws) for _ in 1:8]
    @test length(unique(objectid(p.c) for p in slots)) == 8
    foreach(p -> release!(ws, p), slots)
    @test ws.sp == 8
end

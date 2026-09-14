using Test, PolySeries

include("macro_scalar_tests.jl")
include("macro_timing_tests.jl")

@testset "Macro scalar division follows ordinary domain checks" begin
    desc = PSDesc(1, 3)
    for T in (Float64, Float32, ComplexF64, BigFloat)
        ws = PSWorkspace(desc, 4, T)
        x = CTPS(zero(T), 1, desc)
        out = CTPS(T, desc)
        fill!(out.c, T(7))
        out.degree_mask[] = UInt64(0xf)
        before = copy(out.c)
        for divisor in (0, 0.0, -0.0, zero(T))
            @test_throws DomainError x / divisor
            @test_throws DomainError @tpsa ws out = x / divisor
            @test out.c == before && out.degree_mask[] == UInt64(0xf)
            @test_throws DomainError @tpsa ws out = (x*x) / divisor + x
            @test out.c == before && out.degree_mask[] == UInt64(0xf)
            @test ws.sp == 4 && !any(ws.inuse)
        end
        @tpsa ws out = x / 2
        @test element(out, [1]) == T(0.5)
        @test ws.sp == 4
    end
    # Validation applies after conversion to the polynomial coefficient type.
    ws = PSWorkspace(desc, 4, Float32)
    x = CTPS(0.0f0, 1, desc)
    out = CTPS(Float32, desc)
    @test_throws DomainError x / 1e-100
    @test_throws DomainError @tpsa ws out = x / 1e-100
    @test ws.sp == 4
    # A purely scalar expression retains Julia's scalar division semantics.
    @tpsa ws out = 1.0 / 0.0
    @test cst(out) == Inf32
end

@testset "Macro operand evaluation and typed storage" begin
    @testset "Evaluate once, in source order" begin
        desc = PSDesc(1, 3)
        ws = PSWorkspace(desc, 8)
        x = CTPS(0.0, 1, desc)
        out = CTPS(Float64, desc)
        calls = Ref(0)
        scalar() = (calls[] += 1; 0.1calls[])
        series() = (calls[] += 1; calls[] * x)
        @tpsa ws out = sin(scalar()) * x
        @test calls[] == 1
        @test element(out, [1]) ≈ sin(0.1)
        calls[] = 0
        @tpsa ws out = (series() + x) * x
        @test calls[] == 1
        @test element(out, [2]) == 2.0
        trace = Int[]
        value(i) = (push!(trace, i); Float64(i))
        @tpsa ws out = value(1) + (value(2) + x) * value(3)
        @test trace == [1, 2, 3]
        @test cst(out) == 7.0 && element(out, [1]) == 3.0
        empty!(trace)
        @tpsa ws out = (x ^ Int(value(2))) * x
        @test trace == [2]
        @test element(out, [3]) == 1.0
        empty!(trace)
        @tpsa ws out = ([x][Int(value(1))] + x) * x
        @test trace == [1]
        @test element(out, [2]) == 2.0
        # A later expression may change a variable, but the earlier operand
        # must retain the value it had when Julia evaluated it.
        a = 2.0
        @tpsa ws out = a + ((a = 3.0) * x)
        @test cst(out) == 2.0 && element(out, [1]) == 3.0
        fail() = error("operand failure")
        @test_throws ErrorException @tpsa ws out = (x * x) + fail()
        @test ws.sp == 8 && !any(ws.inuse)
    end
    @testset "Typed workspace arithmetic and ownership" begin
        desc = PSDesc(1, 3)
        for T in (Float64, Float32, ComplexF64, BigFloat)
            ws = PSWorkspace(desc, 4, T)
            @test ws isa PSWorkspace{T}
            @test PSWorkspace{T}(desc, 1) isa PSWorkspace{T}
            x = CTPS(zero(T), 1, desc)
            out = CTPS(T, desc)
            factor = T <: Complex ? T(2im) : T(2)
            offset = T <: Complex ? T(0.5 + 0.5im) : T(0.5)
            @tpsa ws out = x / factor - offset
            @test element(out, [1]) == inv(factor) && cst(out) == -offset
            @test ws.sp == 4 && !any(ws.inuse)
            held = borrow!(ws)
            PolySeries.copy!(held, x)
            @test_throws DomainError @tpsa ws out = (x*x) / x
            @test ws.sp == 3
            @test element(held, [1]) == one(T)
            release!(ws, held)
            @test all(p.degree_mask[] == 0 for p in ws.bufs)
            @test_throws ArgumentError release!(ws, held)
            other = PSWorkspace(desc, 1, T)
            foreign = borrow!(other)
            @test_throws ArgumentError release!(ws, foreign)
            release!(other, foreign)
            @test ws.sp == 4
            if T != Float64
                wrong = PSWorkspace(desc, 4)
                @test_throws ArgumentError @tpsa wrong out = x / factor - offset
                @test wrong.sp == 4 && !any(wrong.inuse)
            end
        end
    end
end

typed_macro!(ws, out, x, a, b) = (@tpsa ws out = x/a - b)
@testset "Typed arithmetic temporaries allocate nothing" begin
    desc = PSDesc(1, 3)
    for T in (Float64, Float32, ComplexF64)
        ws = PSWorkspace(desc, 4, T)
        x, out = CTPS(zero(T), 1, desc), CTPS(T, desc)
        typed_macro!(ws, out, x, T(2), T(0.5))
        @test (@allocated typed_macro!(ws, out, x, T(2), T(0.5))) == 0
    end
end

@testset "Documented macro expressions" begin
    root = dirname(@__DIR__)
    for path in ("docs/src/index.md", "docs/src/tutorial.md", "src/macro.jl")
        source = read(joinpath(root, path), String)
        sandbox = Module(gensym(:MacroExample))
        setup = """
            using PolySeries
            desc = set_descriptor!(4, 6)
            ws = PSWorkspace(desc, 20)
            x1 = x = CTPS(0.0, 1); x2 = y = CTPS(0.0, 2)
            x3 = z = CTPS(0.0, 3)
            nx = CTPS(Float64, desc)
            nx1 = nx
            θ = 2π * 0.205
            μ = θ
            """
        Base.include_string(sandbox, setup)
        # Execute the documented assignment directly. README snippets have a
        # separate extraction test because its quick-reference setup is
        # intentionally self-contained and uses a two-variable descriptor.
        assignment = match(r"^@tpsa\s+\S+\s+(\w+)\s*=.*$"m, source)
        @test assignment !== nothing
        code = assignment.match
        @test begin
            Base.include_string(sandbox, code, path)
            true
        end
        result = Base.invokelatest(getfield, sandbox, Symbol(assignment.captures[1]))
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
    @test_throws DomainError @tpsa ws out = (x*x)*log(x)
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

using Test, PolySeries

@testset "Macro scalar semantics" begin
    desc = PSDesc(1, 3)
    @testset "Convert results after scalar arithmetic" begin
        for op in (:+, :-, :*, :/)
            root = Core.eval(@__MODULE__, :((ws, out, a, b) -> (@tpsa ws out = $op(a, b))))
            nested = Core.eval(@__MODULE__, :((ws, out, a, b) -> (@tpsa ws out = $op(a, b) * 1)))
            for T in (Float32, Float64, ComplexF64)
                ws = PSWorkspace(desc, 4, T)
                out = CTPS(T, desc)
                cases = op === :+ ? ((16777217.0, -16777216.0), (1+im, -im)) :
                        op === :- ? ((16777217.0, 16777216.0), (1+im, im)) :
                        op === :* ? ((1e-100, 1e100), (im, -im)) :
                                    ((1e100, 1e100), (im, im))
                for (a, b) in cases
                    expected = T(getfield(Base, op)(a, b))
                    @test begin
                        Base.invokelatest(root, ws, out, a, b)
                        isequal(cst(out), expected)
                    end
                    @test begin
                        Base.invokelatest(nested, ws, out, a, b)
                        isequal(cst(out), expected)
                    end
                    @test ws.sp == 4 && !any(ws.inuse)
                end
            end
        end
        ws = PSWorkspace(desc, 4)
        out = CTPS(Float64, desc)
        a = UInt8(1)
        @tpsa ws out = -a
        @test cst(out) == Float64(-a)
        @tpsa ws out = (-a) * 1
        @test cst(out) == Float64(-a)
    end

    @testset "Literal and dynamic scalar powers" begin
        ws = PSWorkspace(desc, 4)
        out = CTPS(Float64, desc)
        x = CTPS(0.0, 1, desc)
        @test begin
            @tpsa ws out = 2^-1
            cst(out) == 2^-1
        end
        @test begin
            @tpsa ws out = (2^-1) * x
            element(out, [1]) == 2^-1
        end
        a = Int8(100)
        @tpsa ws out = a^2
        @test cst(out) == Float64(a^2)
        n = 2
        @tpsa ws out = a^n
        @test cst(out) == Float64(a^n)
        @tpsa ws out = (a^n) * x
        @test element(out, [1]) == Float64(a^n)
        base, n = 2, -1
        @test_throws DomainError base^n
        @test_throws DomainError @tpsa ws out = base^n
        @test_throws DomainError @tpsa ws out = (x*x) + base^n
        @test ws.sp == 4 && !any(ws.inuse)
        calls = Ref(0)
        value() = (calls[] += 1; 2)
        @test begin
            @tpsa ws out = value()^-1 * x
            element(out, [1]) == 0.5 && calls[] == 1
        end
        @tpsa ws out = x^2
        @test element(out, [2]) == 1.0
        @tpsa ws out = (x^2) * x
        @test element(out, [3]) == 1.0
        @test ws.sp == 4 && !any(ws.inuse)
        for exponent in (Int32(2), UInt(2), 0.5, 0.5+0im)
            @tpsa ws out = 4^exponent
            @test cst(out) == Float64(4^exponent)
            @tpsa ws out = (4^exponent) * x
            @test element(out, [1]) == Float64(4^exponent)
        end
        @test_throws ArgumentError @tpsa ws out = x^-1
        @test_throws ArgumentError @tpsa ws out = (x^2) * (x^-1)
        @test ws.sp == 4 && !any(ws.inuse)
    end
end

scalar_macro_alloc!(ws, out, x, a, b) = (@tpsa ws out = ((a-b) * (2^-1)) * x + x^2)
@testset "Scalar macro changes preserve zero allocations" begin
    d = PSDesc(1, 3)
    for T in (Float32, Float64, ComplexF64)
        ws = PSWorkspace(d, 4, T)
        x, out = CTPS(zero(T), 1, d), CTPS(T, d)
        scalar_macro_alloc!(ws, out, x, 16777217.0, 16777216.0)
        @test element(out, [1]) == T(0.5) && element(out, [2]) == one(T)
        @test (@allocated scalar_macro_alloc!(ws, out, x, 16777217.0, 16777216.0)) == 0
    end
end

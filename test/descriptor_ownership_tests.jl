using Test, PolySeries

@testset "Polynomials retain their construction descriptor" begin
    d1 = set_descriptor!(1, 2)
    x = CTPS(2.0, 1)
    d2 = set_descriptor!(2, 1) # Same buffer length, different monomial meanings.
    @test x.desc === d1
    @test CTPS(x).desc === d1
    @test (x + 1.0).desc === d1
    @test (x * x).desc === d1
    @test get_descriptor() === d2
    clear_descriptor!()
    @test x.desc === d1
    @test x(0.25) == 2.25
end

@testset "Explicit descriptors, copies, and workspace ownership" begin
    d = PSDesc(1, 4)
    other = set_descriptor!(2, 2)
    x = CTPS(0.25, 1, d)
    @test CTPS(2.0, d).desc === d
    @test CTPS(Float64, d).desc === d
    @test get_descriptor() === other
    ws = PSWorkspace(d, 3)
    out = borrow!(ws)
    @test out.desc === d
    clear_descriptor!()
    @test_throws ErrorException CTPS(1.0)
    for f in (identity, CTPS, p -> p + p, p -> p - 1.0, p -> p * p,
              p -> p^3, exp, sin, p -> compose(p, [p]))
        @test f(x).desc === d
    end
    exp!(out, x)
    @test cst(out) ≈ exp(0.25)
    release!(ws, out)
    @test ws.sp == 3
    foreign = CTPS(7.0, d)
    @test_throws ArgumentError release!(ws, foreign)
    @test cst(foreign) == 7.0
    @test ws.sp == 3
    @test_throws DimensionMismatch CTPS{Float64}(zeros(d.N + 1), d, Ref(UInt64(0)))
    @test_throws DimensionMismatch release!(ws, CTPS(7.0, other))
end

@testset "Descriptor mismatch rejected before output mutation" begin
    d = PSDesc(1, 2)
    x = CTPS(0.25, 1, d)
    for other in (PSDesc(2, 1), PSDesc(2, 4)) # Equal and unequal buffer lengths.
        y = CTPS(0.25, 1, other)
        for op in (+, -, *, /)
            @test_throws DimensionMismatch op(x, y)
        end
        for op in (add!, sub!, mul!)
            out = CTPS(7.0, d)
            before = copy(out.c)
            @test_throws DimensionMismatch op(out, x, y)
            @test out.c == before
            @test out.degree_mask[] == UInt64(1)
            @test_throws DimensionMismatch op(y, x, x)
        end
        for op in (addto!, subfrom!, copy!)
            out = CTPS(7.0, d)
            @test_throws DimensionMismatch op(out, y)
            @test cst(out) == 7.0
        end
        out = CTPS(7.0, d)
        @test_throws DimensionMismatch scale!(out, y, 2.0)
        @test_throws DimensionMismatch scaleadd!(out, 0.0, x, 0.0, y)
        @test_throws DimensionMismatch compose!(out, x, [y])
        @test_throws DimensionMismatch compose!(y, x, [x])
        @test cst(out) == 7.0
        for op in (exp!, log!, sqrt!, sin!, cos!, sinh!, cosh!, asin!, acos!)
            @test_throws DimensionMismatch op(out, y)
            @test cst(out) == 7.0
        end
        @test_throws DimensionMismatch pow!(out, y, 0)
        @test cst(out) == 7.0
    end
end

@testset "Concurrent descriptor creation and polynomial use" begin
    # Run with --threads=4 to exercise parallel cache publication. Explicit
    # yields also force task interleaving when the suite runs with one thread.
    tasks = map(1:12) do i
        Threads.@spawn begin
            d = set_descriptor!(1, i + 5)
            x = CTPS(2.0, 1)
            for _ in 1:5
                yield()
                get_descriptor() === d || error("Task default changed")
                x.desc === d || error("Polynomial descriptor changed")
                cst(x^3) == 8.0 || error("Wrong polynomial value")
            end
            clear_descriptor!()
            (x, d)
        end
    end
    for (x, d) in fetch.(tasks)
        @test x.desc === d
        @test x(0.25) == 2.25
    end
end

@testset "Task defaults do not overwrite one another" begin
    parent = set_descriptor!(1, 2)
    ready = Channel{Nothing}(1)
    resume = Channel{Nothing}(1)
    worker = Threads.@spawn begin
        own = set_descriptor!(2, 3)
        x = CTPS(1.0, 2)
        put!(ready, nothing)
        take!(resume)
        (get_descriptor() === own, x.desc === own)
    end
    take!(ready)
    @test get_descriptor() === parent
    set_descriptor!(3, 2)
    put!(resume, nothing)
    @test fetch(worker) == (true, true)
end

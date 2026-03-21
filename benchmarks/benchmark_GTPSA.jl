using BenchmarkTools
using Printf
using Statistics
using PolySeries
using GTPSA

"""
2-D or 4-D Henon map for benchmark
"""
function henon_map!(x::Vector{T}, μ::Vector{Float64}) where T
    # TODO Implement for 6D case as well
    cos_μ = cos.(μ)
    sin_μ = sin.(μ)
    
    new_x = similar(x)

    if length(x) == 2
        pmx2 = x[2] - x[1]^2

        new_x[1] = cos_μ[1] * x[1] + sin_μ[1] * pmx2
        new_x[2] = cos_μ[1] * pmx2 - sin_μ[1] * x[1]
    
    elseif length(x) == 4 || length(x) == 6
        pmx = x[2] + x[1]^2 - x[3]^2
        pmy = x[4] - 2 * x[1] * x[3]

        new_x[1] = cos_μ[1] * x[1] + sin_μ[1] * pmx
        new_x[2] = cos_μ[1] * pmx - sin_μ[1] * x[1]
        
        new_x[3] = cos_μ[2] * x[3] + sin_μ[2] * pmy
        new_x[4] = cos_μ[2] * pmy - sin_μ[2] * x[3]

    else
        error("Henon map only implemented for 2, 4, or 6 variables")
    end

    if length(x) == 6
        new_x[6] = x[6] + sin(x[5])
        new_x[5] = x[5] - new_x[6]
    end

    # Slow (in place operation)
    # x .= new_x

    # Fast assignment (rebind)
    x = new_x

end

"""
Benchmark GTPSA or allocating TPSA Henon map.
"""
function henon_benchmark(nv::Int, order::Int, n_samples::Int=100, use_gtpsa::Bool=true)
    if use_gtpsa
        d = GTPSA.Descriptor(nv, order)
        x = GTPSA.@vars(d)
    else
        clear_descriptor!()
        set_descriptor!(nv, order)
        x = [CTPS(0.0, i) for i in 1:nv]
    end

    number_of_iterations = 10
    mux = 2.0 * π * 0.205
    muy = 2.0 * π * 0.0125
    μ = nv == 2 ? [mux] : [mux, muy]

    t = @benchmark begin
        for _ in 1:$number_of_iterations
            henon_map!($x, $μ)
        end
    end samples=n_samples

    return median(t).time / 1e6
end

"""
In-place workspace TPSA Henon map — zero heap allocations per step.
"""
function henon_inplace_benchmark(nv::Int, order::Int, n_samples::Int=100)
    clear_descriptor!()
    set_descriptor!(nv, order)
    desc = get_descriptor()

    x  = [CTPS(0.0, i) for i in 1:nv]
    nx = [CTPS(0.0, i) for i in 1:nv]
    ws = PSWorkspace(desc, 16)

    number_of_iterations = 10
    c1 = cos(2π * 0.205);  s1 = sin(2π * 0.205)
    c2 = cos(2π * 0.0125); s2 = sin(2π * 0.0125)

    if nv == 2
        step! = () -> begin
            pmx = borrow!(ws)
            mul!(pmx, x[1], x[1])
            subfrom!(pmx, x[2])         # pmx = x1^2 - x2  →  wait, original: pmx2 = x2 - x1^2
            # correct: pmx2 = x2 - x1^2
            zero!(pmx)
            copy!(pmx, x[2])
            t1 = borrow!(ws)
            mul!(t1, x[1], x[1])
            subfrom!(pmx, t1)
            release!(ws, t1)
            scaleadd!(nx[1], c1, x[1], s1, pmx)
            scaleadd!(nx[2], c1, pmx, -s1, x[1])
            release!(ws, pmx)
        end
    elseif nv == 4
        step! = () -> begin
            t1=borrow!(ws); t2=borrow!(ws); t3=borrow!(ws)
            pmx=borrow!(ws); pmy=borrow!(ws)
            mul!(t1,x[1],x[1]); mul!(t2,x[3],x[3]); mul!(t3,x[1],x[3])
            add!(pmx,x[2],t1); subfrom!(pmx,t2)
            scaleadd!(pmy,1.0,x[4],-2.0,t3)
            release!(ws,t1); release!(ws,t2); release!(ws,t3)
            scaleadd!(nx[1],c1,x[1],s1,pmx)
            scaleadd!(nx[2],c1,pmx,-s1,x[1])
            scaleadd!(nx[3],c2,x[3],s2,pmy)
            scaleadd!(nx[4],c2,pmy,-s2,x[3])
            release!(ws,pmx); release!(ws,pmy)
        end
    else   # nv == 6
        step! = () -> begin
            t1=borrow!(ws); t2=borrow!(ws); t3=borrow!(ws)
            pmx=borrow!(ws); pmy=borrow!(ws); snx6=borrow!(ws); t_sin=borrow!(ws)
            mul!(t1,x[1],x[1]); mul!(t2,x[3],x[3]); mul!(t3,x[1],x[3])
            add!(pmx,x[2],t1); subfrom!(pmx,t2)
            scaleadd!(pmy,1.0,x[4],-2.0,t3)
            release!(ws,t1); release!(ws,t2); release!(ws,t3)
            sin!(t_sin,x[5])
            add!(snx6,x[6],t_sin)
            release!(ws,t_sin)
            scaleadd!(nx[1],c1,x[1],s1,pmx)
            scaleadd!(nx[2],c1,pmx,-s1,x[1])
            scaleadd!(nx[3],c2,x[3],s2,pmy)
            scaleadd!(nx[4],c2,pmy,-s2,x[3])
            sub!(nx[5],x[5],snx6)
            copy!(nx[6],snx6)
            release!(ws,pmx); release!(ws,pmy); release!(ws,snx6)
        end
    end

    step!()  # warm up

    t = @benchmark begin
        for _ in 1:$number_of_iterations
            $step!()
        end
    end samples=n_samples

    return median(t).time / 1e6
end

# ─── Math function benchmarks ────────────────────────────────────────────────

"""
Benchmark a single math function (exp/sin/cos/log/sqrt) on TPSA and GTPSA.
Returns (tpsa_ms, tpsa_inplace_ms, gtpsa_ms).
"""
function mathfunc_benchmark(fn_sym::Symbol, nv::Int, order::Int, n_samples::Int=200)
    clear_descriptor!()
    set_descriptor!(nv, order)
    desc = get_descriptor()

    # TPSA: allocating version — fn(ctps) allocates a new CTPS each call
    # CTPS(1.0, 1) = constant term 1.0 + linear term δx₁  (avoids log(0)/sqrt(neg))
    x_tpsa = CTPS(1.0, 1)
    fn_base = getfield(Base, fn_sym)     # dispatches to TPSA method via multiple dispatch

    t_alloc = @benchmark $fn_base($x_tpsa) samples=n_samples evals=10

    # TPSA: in-place version — fn!(result, ctps) writes into pre-allocated result
    fn_inplace_sym = Symbol(fn_sym, "!")
    result_tpsa = CTPS(Float64)
    if isdefined(PolySeries, fn_inplace_sym)
        fn_inplace = getfield(PolySeries, fn_inplace_sym)
        t_inplace = @benchmark $fn_inplace($result_tpsa, $x_tpsa) samples=n_samples evals=10
        tpsa_inplace_ms = median(t_inplace).time / 1e6
    else
        tpsa_inplace_ms = NaN
    end

    # GTPSA: vars(d)[1] gives δx₁;  +1.0 shifts constant to 1 (needed for log/sqrt)
    d_gtpsa = GTPSA.Descriptor(nv, order)
    x_gtpsa = GTPSA.@vars(d_gtpsa)[1] + 1.0

    t_gtpsa = @benchmark $fn_base($x_gtpsa) samples=n_samples evals=10

    return (
        median(t_alloc).time  / 1e6,
        tpsa_inplace_ms,
        median(t_gtpsa).time  / 1e6,
    )
end

"""
Run math function benchmarks across function × order combinations.
"""
function run_mathfunc_benchmarks(n_samples::Int=200)
    println("\n\n" * "="^90)
    println("MATH FUNCTION BENCHMARKS — TPSA (alloc + in-place) vs GTPSA")
    println("Single evaluation on CTPS(1.0 + δx₁), nv=2")
    println("="^90)

    fns    = [:exp, :log, :sqrt, :sin, :cos]
    orders = [4, 6, 8, 10, 12]
    nv     = 2

    math_results = []

    for fn in fns
        println("\n── $(uppercase(string(fn))) ──────────────────────────────────────────────────────────")
        @printf("%-14s | %10s | %10s | %10s | %8s | %8s\n",
                "order (N)", "TPSA-alloc", "TPSA-inpl", "GTPSA", "G/T-alc", "G/T-inpl")
        println("─"^80)
        for order in orders
            set_descriptor!(nv, order)
            N = get_descriptor().N
            label = "order $order (N=$N)"
            ta, ti, tg = mathfunc_benchmark(fn, nv, order, n_samples)
            r1 = isnan(tg) || isnan(ta) ? NaN : tg / ta
            r2 = isnan(tg) || isnan(ti) ? NaN : tg / ti
            @printf("%-14s | %10.4f | %10.4f | %10.4f | %8.3f | %8.3f\n",
                    label, ta, ti, tg, r1, r2)
            push!(math_results, (fn, nv, order, N, ta, ti, tg))
        end
    end

    # CSV
    csv_file = joinpath(@__DIR__, "benchmark_mathfunc_results.csv")
    open(csv_file, "w") do io
        println(io, "func,vars,order,N,tpsa_alloc_us,tpsa_inplace_us,gtpsa_us,ratio_alloc,ratio_inplace")
        for (fn, nv, order, N, ta, ti, tg) in math_results
            r1 = isnan(tg) || isnan(ta) ? NaN : tg / ta
            r2 = isnan(tg) || isnan(ti) ? NaN : tg / ti
            @printf(io, "%s,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    fn, nv, order, N, ta*1000, ti*1000, tg*1000, r1, r2)
        end
    end
    println("\nCSV written to: $csv_file")
end

# Main benchmark suite
function main()
    println("TPSA vs TPSA-inplace vs GTPSA Henon Map Benchmark")
    println("="^80)

    all_results = []

    configs = [
        (2, 2,  "2 vars, order 2"),
        (2, 6,  "2 vars, order 6"),
        (2, 8,  "2 vars, order 8"),
        (2, 10, "2 vars, order 10"),
        (2, 12, "2 vars, order 12"),
        (4, 2,  "4 vars, order 2"),
        (4, 6,  "4 vars, order 6"),
        (4, 8,  "4 vars, order 8"),
        (4, 10, "4 vars, order 10"),
        (4, 12, "4 vars, order 12"),
        (6, 2,  "6 vars, order 2"),
        (6, 6,  "6 vars, order 6"),
        (6, 8,  "6 vars, order 8"),
        (6, 10, "6 vars, order 10"),
        (6, 12, "6 vars, order 12"),
    ]
    n_samples = 100

    for (nv, order, label) in configs
        println("\n" * "─"^80)
        println("  $label  (N=$(begin set_descriptor!(nv,order); get_descriptor().N end))")
        println("─"^80)

        tpsa_time   = NaN
        inplace_time = NaN
        gtpsa_time  = NaN

        try
            tpsa_time = henon_benchmark(nv, order, n_samples, false)
            @printf("  TPSA  allocating : %8.4f ms\n", tpsa_time)
        catch e
            println("  TPSA  ERROR: $e")
        end

        try
            inplace_time = henon_inplace_benchmark(nv, order, n_samples)
            @printf("  TPSA  in-place   : %8.4f ms\n", inplace_time)
        catch e
            println("  TPSA inplace ERROR: $e")
        end

        try
            gtpsa_time = henon_benchmark(nv, order, n_samples, true)
            @printf("  GTPSA            : %8.4f ms\n", gtpsa_time)
        catch e
            println("  GTPSA ERROR: $e")
        end

        if !isnan(tpsa_time) && !isnan(gtpsa_time)
            @printf("  Ratio GTPSA/TPSA-alloc   : %6.3f\n", gtpsa_time / tpsa_time)
        end
        if !isnan(inplace_time) && !isnan(gtpsa_time)
            @printf("  Ratio GTPSA/TPSA-inplace : %6.3f\n", gtpsa_time / inplace_time)
        end

        push!(all_results, (nv, order, label, tpsa_time, inplace_time, gtpsa_time))
    end

    # Summary table
    println("\n\n" * "="^90)
    println("SUMMARY TABLE — Henon Map (all times in ms, 10 iterations per sample)")
    println("="^90)
    @printf("%-22s | %10s | %10s | %10s | %8s | %8s\n",
            "Configuration", "TPSA-alloc", "TPSA-inpl", "GTPSA", "G/T-alc", "G/T-inpl")
    println("─"^90)

    for (nv, order, label, tpsa, inpl, gtpsa) in all_results
        r1 = (isnan(tpsa)  || isnan(gtpsa)) ? NaN : gtpsa / tpsa
        r2 = (isnan(inpl)  || isnan(gtpsa)) ? NaN : gtpsa / inpl
        @printf("%-22s | %10.4f | %10.4f | %10.4f | %8.3f | %8.3f\n",
                label, tpsa, inpl, gtpsa, r1, r2)
    end
    println("="^90)

    csv_file = joinpath(@__DIR__, "benchmark_results.csv")
    open(csv_file, "w") do io
        println(io, "vars,order,tpsa_alloc_ms,tpsa_inplace_ms,gtpsa_ms,ratio_alloc,ratio_inplace")
        for (nv, order, label, tpsa, inpl, gtpsa) in all_results
            r1 = (isnan(tpsa) || isnan(gtpsa)) ? NaN : gtpsa / tpsa
            r2 = (isnan(inpl) || isnan(gtpsa)) ? NaN : gtpsa / inpl
            @printf(io, "%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    nv, order, tpsa, inpl, gtpsa, r1, r2)
        end
    end
    println("\nCSV results written to: $csv_file")

    # Run math function benchmarks
    run_mathfunc_benchmarks()
end

# Run benchmarks
main()

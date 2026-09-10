#!/usr/bin/env julia
#
# PolySeries vs GTPSA comparison harness.
#
#   julia --project=benchmarks benchmarks/benchmark_GTPSA.jl          # full run
#   julia --project=benchmarks benchmarks/benchmark_GTPSA.jl --quick  # short budget
#
# Four sections, each writing a CSV next to this file:
#   Henon map        benchmark_results.csv
#   Multiplication   benchmark_mul_results.csv
#   Math functions   benchmark_mathfunc_results.csv
#   Composition      benchmark_compose_gtpsa_results.csv
#
# Every CSV starts with `#` provenance lines (date, machine, versions) so a
# stored result can be traced to the code and hardware that produced it.
#
# Measurement notes
#   * The Henon map is genuinely composed: each iteration substitutes the
#     previous result, so the polynomials densify exactly as they do in real
#     map tracking. An earlier version rebound a local variable instead of
#     updating the state, which measured the same first step ten times over
#     and never left the sparse degree-0..2 corner.
#   * Nothing is allocated inside a timed region that a caller would hoist:
#     rotation constants are precomputed and in-place runs reuse a workspace.
#   * Sparse and dense operands are reported separately. Sparse inputs favour
#     an implementation that tracks active degrees; dense inputs are the
#     worst case. Quoting only one of the two is misleading.
#   * Times are the minimum over the sample budget, which is the least noisy
#     estimator for short deterministic kernels.

using Printf
using Statistics
using Dates
using Pkg
using PolySeries
using GTPSA

const QUICK  = "--quick" in ARGS
const BUDGET = QUICK ? 0.4 : 2.0          # seconds of sampling per measurement

# ── provenance ───────────────────────────────────────────────────────────────

function package_versions()
    versions = Dict{String,String}()
    for (_, dep) in Pkg.dependencies()
        dep.name in ("PolySeries", "GTPSA") || continue
        versions[dep.name] = dep.version === nothing ? "dev" : string(dep.version)
    end
    return versions
end

const VERSIONS = package_versions()
const PROVENANCE = [
    "date=$(Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"))",
    "julia=$(VERSION)",
    "machine=$(Sys.MACHINE)",
    "cpu=$(Sys.CPU_NAME)",
    "threads=$(Threads.nthreads())",
    "polyseries=$(get(VERSIONS, "PolySeries", "?"))",
    "gtpsa=$(get(VERSIONS, "GTPSA", "?"))",
    "budget_seconds=$(BUDGET)",
]

function write_csv(name, header, rows)
    path = joinpath(@__DIR__, name)
    open(path, "w") do io
        for line in PROVENANCE
            println(io, "# ", line)
        end
        println(io, header)
        for row in rows
            println(io, join(row, ','))
        end
    end
    println("  wrote $path")
    return path
end

fmt(x) = isnan(x) ? "NaN" : @sprintf("%.6g", x)

# ── timing ───────────────────────────────────────────────────────────────────

"""
    timed(run!; reset!, budget, label) -> seconds

Minimum wall time of one `run!()` call. `reset!` restores mutable state and is
called before every sample; supplying it forces one evaluation per sample so a
mutating kernel never measures an already-advanced state. Without it the number
of evaluations per sample is tuned so a sample lasts about a millisecond.
Returns `NaN` (with a note) when the target throws, so one unsupported
operation cannot abort the run.
"""
function timed(run!; reset!::Union{Nothing,Function}=nothing,
               budget::Float64=BUDGET, label::AbstractString="")
    try
        reset! === nothing || reset!()
        run!()
    catch err
        printstyled("    [skipped] ", label, ": ", sprint(showerror, err), "\n"; color=:yellow)
        return NaN
    end

    evals = 1
    if reset! === nothing
        single = @elapsed run!()
        single <= 0 && (single = 1e-9)
        evals = clamp(round(Int, 1e-3 / single), 1, 100_000)
    end

    best = Inf
    samples = 0
    started = time()
    while samples < 5 || (time() - started) < budget
        reset! === nothing || reset!()
        elapsed = @elapsed for _ in 1:evals
            run!()
        end
        best = min(best, elapsed / evals)
        samples += 1
        samples >= 100_000 && break
    end
    return best
end

ratio(gtpsa, ours) = (isnan(gtpsa) || isnan(ours) || ours == 0) ? NaN : gtpsa / ours

# ── shared construction helpers ──────────────────────────────────────────────

"""Dense polynomial with a nonzero constant term, built from generic arithmetic
so PolySeries and GTPSA get structurally identical inputs."""
dense_operand(vars, order) = (1.0 + 0.1 * foldl(+, vars))^order

"""Sparse polynomial: constant plus one linear term."""
sparse_operand(vars) = 1.0 + vars[1]

polyseries_vars(desc) = [CTPS(0.0, i, desc) for i in 1:desc.nv]
gtpsa_vars(d) = GTPSA.@vars(d)

"""Fraction of degree blocks that are active — evidence that the Henon
iteration really densifies the state."""
function active_fraction(p::CTPS)
    mask = p.degree_mask[]
    return count_ones(mask) / (p.desc.order + 1)
end

# ── section 1: Henon map ─────────────────────────────────────────────────────

"""
One Henon step. Allocating and generic: the same method body runs for
`CTPS` and for GTPSA's `TPS`, so the two libraries execute the same program.
Returns a new state vector; the driver rebinds it, which is what makes the
iteration actually compose.
"""
function henon_step(x::Vector{T}, c::Vector{Float64}, s::Vector{Float64}) where T
    if length(x) == 2
        pmx = x[2] - x[1]^2
        return T[c[1] * x[1] + s[1] * pmx,
                 c[1] * pmx  - s[1] * x[1]]
    elseif length(x) == 4
        pmx = x[2] + x[1]^2 - x[3]^2
        pmy = x[4] - 2.0 * x[1] * x[3]
        return T[c[1] * x[1] + s[1] * pmx,
                 c[1] * pmx  - s[1] * x[1],
                 c[2] * x[3] + s[2] * pmy,
                 c[2] * pmy  - s[2] * x[3]]
    elseif length(x) == 6
        pmx = x[2] + x[1]^2 - x[3]^2
        pmy = x[4] - 2.0 * x[1] * x[3]
        n6  = x[6] + sin(x[5])
        return T[c[1] * x[1] + s[1] * pmx,
                 c[1] * pmx  - s[1] * x[1],
                 c[2] * x[3] + s[2] * pmy,
                 c[2] * pmy  - s[2] * x[3],
                 x[5] - n6,
                 n6]
    end
    error("Henon map is defined for 2, 4 or 6 variables")
end

function henon_iterate(x0, c, s, iterations)
    x = x0
    for _ in 1:iterations
        x = henon_step(x, c, s)          # rebinding here is what composes the map
    end
    return x
end

"""In-place Henon state: two coefficient sets swapped each step, one workspace."""
mutable struct HenonState
    x  :: Vector{CTPS{Float64}}
    nx :: Vector{CTPS{Float64}}
    ws :: PSWorkspace
    c  :: Vector{Float64}
    s  :: Vector{Float64}
end

function HenonState(desc::PSDesc, c, s)
    return HenonState(polyseries_vars(desc), polyseries_vars(desc),
                      PSWorkspace(desc, 16), c, s)
end

"""Restore the state to the identity map so every sample starts from turn zero."""
function reset_henon!(st::HenonState)
    for (i, p) in enumerate(st.x)
        zero!(p)
        p.c[i + 1] = 1.0
        p.degree_mask[] = UInt64(2)     # degree 1 only
    end
    for p in st.nx
        zero!(p)
    end
    return st
end

function henon_step!(st::HenonState)
    x, nx, ws, c, s = st.x, st.nx, st.ws, st.c, st.s
    if length(x) == 2
        t1  = borrow!(ws)
        pmx = borrow!(ws)
        mul!(t1, x[1], x[1])
        sub!(pmx, x[2], t1)                        # pmx = x2 - x1^2
        release!(ws, t1)
        scaleadd!(nx[1], c[1], x[1], s[1], pmx)
        scaleadd!(nx[2], c[1], pmx, -s[1], x[1])
        release!(ws, pmx)
    elseif length(x) == 4
        t1, t2, t3 = borrow!(ws), borrow!(ws), borrow!(ws)
        pmx, pmy = borrow!(ws), borrow!(ws)
        mul!(t1, x[1], x[1]); mul!(t2, x[3], x[3]); mul!(t3, x[1], x[3])
        add!(pmx, x[2], t1); subfrom!(pmx, t2)
        scaleadd!(pmy, 1.0, x[4], -2.0, t3)
        release!(ws, t3); release!(ws, t2); release!(ws, t1)
        scaleadd!(nx[1], c[1], x[1], s[1], pmx)
        scaleadd!(nx[2], c[1], pmx, -s[1], x[1])
        scaleadd!(nx[3], c[2], x[3], s[2], pmy)
        scaleadd!(nx[4], c[2], pmy, -s[2], x[3])
        release!(ws, pmy); release!(ws, pmx)
    else
        t1, t2, t3 = borrow!(ws), borrow!(ws), borrow!(ws)
        pmx, pmy = borrow!(ws), borrow!(ws)
        n6, tsin = borrow!(ws), borrow!(ws)
        mul!(t1, x[1], x[1]); mul!(t2, x[3], x[3]); mul!(t3, x[1], x[3])
        add!(pmx, x[2], t1); subfrom!(pmx, t2)
        scaleadd!(pmy, 1.0, x[4], -2.0, t3)
        release!(ws, t3); release!(ws, t2); release!(ws, t1)
        sin!(tsin, x[5])
        add!(n6, x[6], tsin)
        release!(ws, tsin)
        scaleadd!(nx[1], c[1], x[1], s[1], pmx)
        scaleadd!(nx[2], c[1], pmx, -s[1], x[1])
        scaleadd!(nx[3], c[2], x[3], s[2], pmy)
        scaleadd!(nx[4], c[2], pmy, -s[2], x[3])
        sub!(nx[5], x[5], n6)
        copy!(nx[6], n6)
        release!(ws, n6); release!(ws, pmy); release!(ws, pmx)
    end
    st.x, st.nx = st.nx, st.x          # swap: results become the next input
    return nothing
end

function henon_iterate!(st::HenonState, iterations::Int)
    for _ in 1:iterations
        henon_step!(st)
    end
    return st
end

function run_henon(configs, iterations)
    println("\n", "="^94)
    println("HENON MAP — $iterations composed iterations (state densifies, as in real tracking)")
    println("="^94)
    @printf("%-16s %10s %12s %12s %12s %9s %9s %8s\n",
            "config", "N", "alloc s", "in-place s", "GTPSA s", "G/alloc", "G/inpl", "active")
    println("─"^94)

    rows = []
    for (nv, order) in configs
        desc = PSDesc(nv, order)
        mu   = nv == 2 ? [2π * 0.205] : [2π * 0.205, 2π * 0.0125]
        c, s = cos.(mu), sin.(mu)                   # hoisted out of the timed region

        x0 = polyseries_vars(desc)
        t_alloc = timed(() -> henon_iterate(x0, c, s, iterations);
                        label="henon alloc nv=$nv order=$order")

        st = HenonState(desc, c, s)
        t_inplace = timed(() -> henon_iterate!(st, iterations);
                          reset! = () -> reset_henon!(st),
                          label="henon in-place nv=$nv order=$order")

        # How dense did the state actually become? 1.0 means every degree block
        # is active, i.e. the measurement left the sparse corner.
        reset_henon!(st)
        henon_iterate!(st, iterations)
        density = mean(active_fraction, st.x)

        t_gtpsa = try
            d  = GTPSA.Descriptor(nv, order)
            xg = gtpsa_vars(d)
            timed(() -> henon_iterate(xg, c, s, iterations);
                  label="henon GTPSA nv=$nv order=$order")
        catch err
            printstyled("    [skipped] GTPSA nv=$nv order=$order: ",
                        sprint(showerror, err), "\n"; color=:yellow)
            NaN
        end

        @printf("%-16s %10d %12.3e %12.3e %12.3e %9.2f %9.2f %8.2f\n",
                "nv=$nv order=$order", desc.N, t_alloc, t_inplace, t_gtpsa,
                ratio(t_gtpsa, t_alloc), ratio(t_gtpsa, t_inplace), density)
        push!(rows, (nv, order, desc.N, iterations, fmt(t_alloc), fmt(t_inplace),
                     fmt(t_gtpsa), fmt(ratio(t_gtpsa, t_alloc)),
                     fmt(ratio(t_gtpsa, t_inplace)), fmt(density)))
    end

    write_csv("benchmark_results.csv",
              "vars,order,N,iterations,polyseries_alloc_s,polyseries_inplace_s," *
              "gtpsa_s,ratio_alloc,ratio_inplace,active_degree_fraction", rows)
    return rows
end

# ── section 2: multiplication ────────────────────────────────────────────────

function run_multiplication(configs)
    println("\n", "="^94)
    println("MULTIPLICATION — the kernel underneath every other operation")
    println("="^94)
    @printf("%-16s %8s %8s %12s %12s %12s %9s %9s\n",
            "config", "operand", "N", "alloc s", "mul! s", "GTPSA s", "G/alloc", "G/mul!")
    println("─"^94)

    rows = []
    for (nv, order) in configs, density in ("sparse", "dense")
        desc = PSDesc(nv, order)
        vars = polyseries_vars(desc)
        a = density == "dense" ? dense_operand(vars, order) : sparse_operand(vars)
        b = density == "dense" ? dense_operand(vars, order) * 1.5 : 1.0 + vars[min(2, nv)]
        r = CTPS(Float64, desc)

        t_alloc = timed(() -> a * b; label="mul alloc $density nv=$nv order=$order")
        t_mul   = timed(() -> mul!(r, a, b); label="mul! $density nv=$nv order=$order")

        t_gtpsa = try
            d  = GTPSA.Descriptor(nv, order)
            vg = gtpsa_vars(d)
            ag = density == "dense" ? dense_operand(vg, order) : sparse_operand(vg)
            bg = density == "dense" ? dense_operand(vg, order) * 1.5 : 1.0 + vg[min(2, nv)]
            timed(() -> ag * bg; label="mul GTPSA $density nv=$nv order=$order")
        catch err
            printstyled("    [skipped] GTPSA mul $density nv=$nv: ",
                        sprint(showerror, err), "\n"; color=:yellow)
            NaN
        end

        @printf("%-16s %8s %8d %12.3e %12.3e %12.3e %9.2f %9.2f\n",
                "nv=$nv order=$order", density, desc.N, t_alloc, t_mul, t_gtpsa,
                ratio(t_gtpsa, t_alloc), ratio(t_gtpsa, t_mul))
        push!(rows, (nv, order, desc.N, density, fmt(t_alloc), fmt(t_mul), fmt(t_gtpsa),
                     fmt(ratio(t_gtpsa, t_alloc)), fmt(ratio(t_gtpsa, t_mul))))
    end

    write_csv("benchmark_mul_results.csv",
              "vars,order,N,operands,polyseries_alloc_s,polyseries_mul_bang_s," *
              "gtpsa_alloc_s,ratio_alloc,ratio_mul_bang", rows)
    return rows
end

# ── section 3: math functions ────────────────────────────────────────────────

function run_mathfunc(configs, fns)
    println("\n", "="^94)
    println("MATH FUNCTIONS — allocating and in-place, sparse and dense arguments")
    println("="^94)
    @printf("%-6s %-16s %8s %8s %12s %12s %12s %9s %9s\n",
            "fn", "config", "operand", "N", "alloc s", "in-place s", "GTPSA s",
            "G/alloc", "G/inpl")
    println("─"^94)

    rows = []
    for fn in fns, (nv, order) in configs, density in ("sparse", "dense")
        desc = PSDesc(nv, order)
        vars = polyseries_vars(desc)
        # Constant term 1 keeps log and sqrt in their domain for both libraries.
        x = density == "dense" ? dense_operand(vars, order) : sparse_operand(vars)
        out = CTPS(Float64, desc)
        fn_bang = getfield(PolySeries, Symbol(fn, "!"))

        t_alloc = timed(() -> fn(x); label="$fn alloc $density nv=$nv order=$order")
        t_inpl  = timed(() -> fn_bang(out, x); label="$(fn)! $density nv=$nv order=$order")

        t_gtpsa = try
            d  = GTPSA.Descriptor(nv, order)
            vg = gtpsa_vars(d)
            xg = density == "dense" ? dense_operand(vg, order) : sparse_operand(vg)
            timed(() -> fn(xg); label="$fn GTPSA $density nv=$nv order=$order")
        catch err
            printstyled("    [skipped] GTPSA $fn $density nv=$nv: ",
                        sprint(showerror, err), "\n"; color=:yellow)
            NaN
        end

        @printf("%-6s %-16s %8s %8d %12.3e %12.3e %12.3e %9.2f %9.2f\n",
                string(fn), "nv=$nv order=$order", density, desc.N,
                t_alloc, t_inpl, t_gtpsa,
                ratio(t_gtpsa, t_alloc), ratio(t_gtpsa, t_inpl))
        push!(rows, (string(fn), nv, order, desc.N, density, fmt(t_alloc), fmt(t_inpl),
                     fmt(t_gtpsa), fmt(ratio(t_gtpsa, t_alloc)),
                     fmt(ratio(t_gtpsa, t_inpl))))
    end

    write_csv("benchmark_mathfunc_results.csv",
              "func,vars,order,N,operand,polyseries_alloc_s,polyseries_inplace_s," *
              "gtpsa_s,ratio_alloc,ratio_inplace", rows)
    return rows
end

# ── section 4: composition ───────────────────────────────────────────────────

function run_composition(configs)
    println("\n", "="^94)
    println("COMPOSITION — f(g₁(x), …, g_nv(x))")
    println("="^94)
    @printf("%-16s %8s %8s %12s %12s %12s %9s %9s\n",
            "config", "source", "N", "compose s", "workspace s", "GTPSA s",
            "G/compose", "G/ws")
    println("─"^94)

    rows = []
    for (nv, order) in configs, density in ("sparse", "dense")
        desc = PSDesc(nv, order)
        vars = polyseries_vars(desc)
        f = density == "dense" ? dense_operand(vars, order) : vars[1]^order
        g = [CTPS(0.1, i, desc) for i in 1:nv]         # shifted substitution map
        out = CTPS(Float64, desc)
        ws  = CompositionWorkspace(desc)

        t_alloc = timed(() -> compose(f, g); label="compose $density nv=$nv order=$order")
        t_ws    = timed(() -> compose!(out, f, g, ws);
                        label="compose! $density nv=$nv order=$order")

        t_gtpsa = try
            d  = GTPSA.Descriptor(nv, order)
            vg = gtpsa_vars(d)
            fg = density == "dense" ? dense_operand(vg, order) : vg[1]^order
            gg = [0.1 + vg[i] for i in 1:nv]
            outer = [fg]
            # Base.∘ happily builds a ComposedFunction out of anything callable,
            # which would time as a no-op. Demand a real map back before trusting it.
            probe = outer ∘ gg
            probe isa AbstractVector || error(
                "GTPSA map composition returned $(typeof(probe)); this GTPSA version " *
                "does not implement ∘ for Vector{TPS}")
            timed(() -> outer ∘ gg; label="compose GTPSA $density nv=$nv order=$order")
        catch err
            printstyled("    [skipped] GTPSA compose $density nv=$nv: ",
                        sprint(showerror, err), "\n"; color=:yellow)
            NaN
        end

        @printf("%-16s %8s %8d %12.3e %12.3e %12.3e %9.2f %9.2f\n",
                "nv=$nv order=$order", density, desc.N, t_alloc, t_ws, t_gtpsa,
                ratio(t_gtpsa, t_alloc), ratio(t_gtpsa, t_ws))
        push!(rows, (nv, order, desc.N, density, fmt(t_alloc), fmt(t_ws), fmt(t_gtpsa),
                     fmt(ratio(t_gtpsa, t_alloc)), fmt(ratio(t_gtpsa, t_ws))))
    end

    write_csv("benchmark_compose_gtpsa_results.csv",
              "vars,order,N,source,polyseries_compose_s,polyseries_workspace_s," *
              "gtpsa_s,ratio_compose,ratio_workspace", rows)
    return rows
end

# ── driver ───────────────────────────────────────────────────────────────────

function main()
    println("PolySeries vs GTPSA")
    for line in PROVENANCE
        println("  ", line)
    end
    println("\nRatios are GTPSA ÷ PolySeries: above 1.0 means PolySeries is faster.")

    # Composed iteration densifies the state, so high orders at six variables
    # cost far more than they did under the old non-iterating harness.
    henon_configs = QUICK ? [(2, 4), (4, 4)] :
        [(2, 2), (2, 6), (2, 8), (2, 10), (2, 12),
         (4, 2), (4, 4), (4, 6), (4, 8),
         (6, 2), (6, 4), (6, 6)]
    kernel_configs = QUICK ? [(2, 6), (4, 6)] :
        [(2, 6), (2, 12), (4, 4), (4, 6), (4, 8), (6, 4), (6, 6)]
    math_configs = QUICK ? [(2, 6)] :
        [(2, 6), (2, 12), (4, 6), (4, 8), (6, 4), (6, 6)]
    compose_configs = QUICK ? [(3, 4)] : [(3, 6), (4, 6), (6, 8)]

    run_henon(henon_configs, 10)
    run_multiplication(kernel_configs)
    run_mathfunc(math_configs, (exp, log, sqrt, sin, cos))
    run_composition(compose_configs)

    println("\nDone.")
end

main()

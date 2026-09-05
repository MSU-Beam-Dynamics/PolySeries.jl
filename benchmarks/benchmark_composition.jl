# Run from the repository root:
# julia --compiled-modules=existing --project=. benchmarks/benchmark_composition.jl
# Add --large-baseline to measure the ~489 MiB retained-image path at N=8008.
using PolySeries

function composition_case(nv, order, sparse, shifted; large_baseline=false)
    desc = PSDesc(nv, order)
    g = [CTPS(shifted ? 0.1 : 0.0, i, desc) for i in 1:nv]
    f = CTPS(Float64, desc)
    if sparse
        f = CTPS(0.0, 1, desc)^order
    else
        fill!(f.c, 1.0)
        f.degree_mask[] = typemax(UInt64) >> (63-order)
    end
    out = CTPS(Float64, desc)
    ws = CompositionWorkspace(desc)
    strategies = [("ordinary", () -> compose!(out, f, g)),
                  ("workspace", () -> compose!(out, f, g, ws))]
    if desc.N <= 1000 || large_baseline
        push!(strategies, ("retained", () -> PolySeries._compose_retained!(out, f, g)))
    end
    compose!(out, f, g, ws) # Populate pruning metadata for every CSV row.
    for (name, run) in strategies
        run() # Compile and warm storage outside measurements.
        bytes = @allocated run()
        seconds = minimum(@elapsed(run()) for _ in 1:5)
        println(join((nv, order, desc.N, sparse, shifted, name, bytes,
                      seconds, count(ws.needed)), ','))
    end
    return nothing
end

println("nv,order,N,sparse,shifted,strategy,allocated_bytes,min_seconds,needed_nodes")
for (nv, order) in ((3, 6), (4, 6), (6, 10)), sparse in (false, true), shifted in (false, true)
    composition_case(nv, order, sparse, shifted;
                     large_baseline="--large-baseline" in ARGS)
end

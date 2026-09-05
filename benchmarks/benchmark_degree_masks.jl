# Run with: julia --project=benchmarks benchmarks/benchmark_degree_masks.jl
# Reports warmed median time and allocations; descriptor setup is excluded.
using BenchmarkTools
using PolySeries

function mask_input(desc, mask)
    p = CTPS(Float64)
    for d in 0:desc.order
        iszero(mask & (UInt64(1) << d)) && continue
        s = desc.off[d + 1]
        e = s + desc.Nd[d + 1] - 1
        p.c[s:e] .= 0.01
    end
    p.degree_mask[] = mask
    return p
end

function benchmark_degree_masks()
    println("nv,order,pattern,operation,median_ns,bytes,allocations")
    for (nv, order) in ((2, 4), (6, 6))
        desc = set_descriptor!(nv, order)
        full = (UInt64(1) << (order + 1)) - 1
        for (pattern, mask) in (("dense", full), ("contiguous", full & ~UInt64(3)),
                                ("gapped", full & UInt64(0x5555555555555555)))
            a = mask_input(desc, mask)
            b = mask_input(desc, mask)
            out = CTPS(Float64)
            args = ntuple(_ -> 0.01, nv)
            operations = (
                ("add!", () -> add!(out, a, b)),
                ("sub!", () -> sub!(out, a, b)),
                ("scaleadd!", () -> scaleadd!(out, 2.0, a, -1.0, b)),
                ("copy!", () -> copy!(out, a)),
                ("scale!", () -> scale!(out, a, 2.0)),
                ("_add_scaled!", () -> PolySeries._add_scaled!(out, a, 0.01)),
                ("addto!", () -> addto!(out, a)),
                ("subfrom!", () -> subfrom!(out, a)),
                ("evaluate", () -> a(args...)),
                ("mul!", () -> mul!(out, a, b)),
                ("exp!", () -> exp!(out, a)),
            )
            for (name, operation) in operations
                trial = @benchmark $operation() seconds=0.15 samples=1500
                estimate = median(trial)
                println(join((nv, order, pattern, name, estimate.time,
                              estimate.memory, estimate.allocs), ','))
            end
        end
    end
end

benchmark_degree_masks()

# julia --project=benchmarks benchmarks/benchmark_active_mul_schedules.jl
# Compare this output across revisions; inputs, compilation, and descriptor
# construction are outside the timed region. No Enzyme is loaded.
using PolySeries, BenchmarkTools

function schedule_benchmark_input(desc, mask, phase)
    p = CTPS(Float64, desc)
    fill!(p.c, NaN)
    p.degree_mask[] = mask
    for (lo, hi) in PolySeries.active_ranges(desc, mask), i in lo:hi
        p.c[i] = sin(i + phase) / 10
    end
    return p
end

function benchmark_active_schedules()
    println("nv,order,pattern,total_schedules,active_schedules,median_ns,bytes,allocations")
    for (nv, order) in ((2, 6), (2, 12), (4, 8), (6, 6), (1, 63))
        desc = PSDesc(nv, order)
        full = typemax(UInt64) >> (63 - order)
        holes = UInt64(1) | (UInt64(1) << 3) | (UInt64(1) << order)
        for (pattern, mask) in (("sparse", UInt64(3)), ("gapped", holes),
                                ("prefix", full >> 1), ("near_dense", full & ~UInt64(4)),
                                ("dense", full))
            a = schedule_benchmark_input(desc, mask, 0)
            b = schedule_benchmark_input(desc, mask, 1)
            out = CTPS(Float64, desc)
            active = count(_ -> true, PolySeries.ActiveMulSchedules(desc, mask, mask))
            mul!(out, a, b)
            estimate = median(@benchmark mul!($out, $a, $b) seconds=0.5)
            println(join((nv, order, pattern, length(desc.mul), active,
                          estimate.time, estimate.memory, estimate.allocs), ','))
        end
    end
end

benchmark_active_schedules()

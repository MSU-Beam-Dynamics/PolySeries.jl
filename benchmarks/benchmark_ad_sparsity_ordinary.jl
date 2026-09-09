# Ordinary counterpart to benchmark_ad_sparsity.jl. This process deliberately
# does not load Enzyme, so it measures the non-AD path in isolation.
# Run from the package root with:
#   julia --project=benchmarks benchmarks/benchmark_ad_sparsity_ordinary.jl

using BenchmarkTools
using PolySeries

scaled_constant(p) = cst(p * 2.0)
exponential_constant(p) = cst(exp(p))
Base.@noinline ordinary_scaled(p) = scaled_constant(p)
Base.@noinline ordinary_exp(p) = exponential_constant(p)

function measure(operation, input)
    operation(input)
    estimate = median(@benchmark $operation($input) seconds=0.5 samples=1000)
    return estimate.time, estimate.memory, estimate.allocs
end

println("nv,order,N,input,operation,median_ns,bytes,allocations")
for (nv, order) in ((1, 8), (3, 8), (6, 6), (6, 10))
    desc = PSDesc(nv, order)
    inactive = CTPS(Float64, desc)
    fill!(inactive.c, NaN)
    dense = CTPS(Float64, desc)
    dense.degree_mask[] = (UInt64(1) << (order + 1)) - UInt64(1)
    for (input_name, input) in (("inactive", inactive), ("dense", dense))
        for (name, operation) in (("scale", ordinary_scaled), ("exp", ordinary_exp))
            time, bytes, allocations = measure(operation, input)
            println(join((nv, order, desc.N, input_name, name,
                          round(Int, time), bytes, allocations), ','))
        end
    end
end

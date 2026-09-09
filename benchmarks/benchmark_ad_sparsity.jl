# Benchmark the cost of preserving derivatives for inactive CTPS coefficients.
# Run from the package root with:
#   julia --project=benchmarks benchmarks/benchmark_ad_sparsity.jl

using BenchmarkTools
using Enzyme
using PolySeries
using Printf

const BENCHMARK_SECONDS = 0.5
const BENCHMARK_SAMPLES = 1000

scaled_constant(p) = cst(p * 2.0)
exponential_constant(p) = cst(exp(p))

# Keep the benchmark harness from folding away calls whose result is known to
# be zero. The package functions themselves retain their normal inlining.
Base.@noinline ordinary_scaled(p) = scaled_constant(p)
Base.@noinline ordinary_exp(p) = exponential_constant(p)
Base.@noinline forward_scaled(p, tangent) = Enzyme.autodiff(Forward, scaled_constant,
                                                             Duplicated(p, tangent))[1]
Base.@noinline forward_exp(p, tangent) = Enzyme.autodiff(Forward, exponential_constant,
                                                          Duplicated(p, tangent))[1]
Base.@noinline reverse_scaled(p) = Enzyme.gradient(Reverse, scaled_constant, p)[1]
Base.@noinline reverse_exp(p) = Enzyme.gradient(Reverse, exponential_constant, p)[1]

function measure(operation, input)
    operation(input)                    # compile and warm up
    trial = @benchmark $operation($input) seconds=BENCHMARK_SECONDS samples=BENCHMARK_SAMPLES
    estimate = median(trial)
    return estimate.time, estimate.memory, estimate.allocs
end

function measure(operation, input, tangent)
    operation(input, tangent)           # compile and warm up
    trial = @benchmark $operation($input, $tangent) seconds=BENCHMARK_SECONDS samples=BENCHMARK_SAMPLES
    estimate = median(trial)
    return estimate.time, estimate.memory, estimate.allocs
end

function make_inputs(desc)
    inactive = CTPS(Float64, desc)
    fill!(inactive.c, NaN)              # prove inactive storage is never read

    dense = CTPS(Float64, desc)
    dense.degree_mask[] = (UInt64(1) << (desc.order + 1)) - UInt64(1)

    tangent = CTPS(1.0, desc)
    return inactive, dense, tangent
end

function run_benchmarks()
    println("nv,order,N,input,mode,operation,median_ns,bytes,allocations")
    for (nv, order) in ((1, 8), (3, 8), (6, 6), (6, 10))
        desc = PSDesc(nv, order)
        inactive, dense, tangent = make_inputs(desc)
        for (input_name, input) in (("inactive", inactive), ("dense", dense))
            operations = (
                ("ordinary", "scale", ordinary_scaled, (input,)),
                ("ordinary", "exp", ordinary_exp, (input,)),
                ("forward", "scale", forward_scaled, (input, tangent)),
                ("forward", "exp", forward_exp, (input, tangent)),
                ("reverse", "scale", reverse_scaled, (input,)),
                ("reverse", "exp", reverse_exp, (input,)),
            )
            for (mode, name, operation, args) in operations
                time, bytes, allocations = measure(operation, args...)
                @printf("%d,%d,%d,%s,%s,%s,%.0f,%d,%d\n",
                        nv, order, desc.N, input_name, mode, name,
                        time, bytes, allocations)
            end
        end
    end
end

run_benchmarks()

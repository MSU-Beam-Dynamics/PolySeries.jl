# Benchmark mathematical functions with TPSA
using TPSA
using BenchmarkTools

println("="^70)
println("TPSA Mathematical Functions Benchmark")
println("="^70)

# Setup
nv = 6
order = 6
set_descriptor!(nv, order)

println("\nConfiguration: $nv variables, order $order")
println("Total coefficients: ", get_descriptor().N)
println()

# Create test variables
x = CTPS(0.0, 1)
y = CTPS(0.0, 2)

# Create test functions for composition
g = 0.5*x + 0.1*x^2  # g(x) = 0.5x + 0.1x²

functions_to_test = [
    ("exp(x)", () -> TPSA.exp(x)),
    ("log(1+x)", () -> TPSA.log(1 + x)),
    ("sin(x)", () -> TPSA.sin(x)),
    ("cos(x)", () -> TPSA.cos(x)),
    ("sinh(x)", () -> TPSA.sinh(x)),
    ("cosh(x)", () -> TPSA.cosh(x)),
    ("sqrt(1+x)", () -> TPSA.sqrt(1 + x)),
    ("(1+x)^10", () -> (1 + x)^10),
    ("compose: sin(g)", () -> TPSA.sin(g)),
]

for (name, func) in functions_to_test
    println("-"^70)
    println("Function: $name")
    println("-"^70)
    result = @benchmark $func()
    println("  Median: ", median(result.times) / 1000, " μs")
    println("  Mean:   ", mean(result.times) / 1000, " μs")
    println("  Allocs: ", result.allocs)
    println()
end

println("="^70)
println("Benchmark complete!")
println("="^70)

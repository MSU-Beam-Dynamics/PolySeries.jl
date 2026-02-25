# Benchmark basic TPSA operations
using TPSA
using BenchmarkTools

println("="^70)
println("TPSA Basic Operations Benchmark")
println("="^70)

# Setup
nv = 3
order = 4
set_descriptor!(nv, order)

println("\nConfiguration: $nv variables, order $order")
println("Total coefficients: ", get_descriptor().N)
println()

# Create test variables
x = CTPS(0.0, 1)
y = CTPS(0.0, 2)
z = CTPS(0.0, 3)

# Create test polynomials
p1 = 1.0 + x + y + z
p2 = x^2 + y^2 + z^2

println("Test polynomials:")
println("  p1 = 1 + x + y + z")
println("  p2 = x² + y² + z²")
println()

# Benchmark addition
println("-"^70)
println("Addition: p1 + p2")
println("-"^70)
result = @benchmark $p1 + $p2
display(result)
println()

# Benchmark subtraction
println("-"^70)
println("Subtraction: p1 - p2")
println("-"^70)
result = @benchmark $p1 - $p2
display(result)
println()

# Benchmark multiplication
println("-"^70)
println("Multiplication: p1 * p2")
println("-"^70)
result = @benchmark $p1 * $p2
display(result)
println()

# Benchmark scalar multiplication
println("-"^70)
println("Scalar multiplication: 2.5 * p1")
println("-"^70)
result = @benchmark 2.5 * $p1
display(result)
println()

# Benchmark power
println("-"^70)
println("Power: p1^3")
println("-"^70)
result = @benchmark $p1^3
display(result)
println()

println("="^70)
println("Benchmark complete!")
println("="^70)

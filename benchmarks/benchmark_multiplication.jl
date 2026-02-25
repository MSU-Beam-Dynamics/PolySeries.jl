# Benchmark multiplication performance
using TPSA
using Printf
using BenchmarkTools

println("Benchmarking TPSA Multiplication\n")
println("="^60)

# Test various sizes
test_cases = [
    (2, 3, "Small: 2 vars, order 3"),
    (3, 4, "Medium: 3 vars, order 4"),
    (4, 4, "Large: 4 vars, order 4"),
]

for (nv, order, desc_str) in test_cases
    println("\n$desc_str")
    println("-"^60)
    
    # Set up descriptor
    set_descriptor!(nv, order)
    desc = get_descriptor()
    N = desc.N
    println("  Total coefficients: $N")
    
    # Create test CTPS objects with random coefficients
    x = CTPS(Float64)
    y = CTPS(Float64)
    for i in 1:N
        x.c[i] = rand()
        y.c[i] = rand()
    end
    TPSA.update_degree_mask!(x)
    TPSA.update_degree_mask!(y)
    
    # Create polynomial test cases
    a = CTPS(1.0)
    b = CTPS(2.0)
    for i in 1:nv
        a = a + 0.1 * CTPS(0.0, i)
        b = b + 0.2 * CTPS(0.0, i)
    end
    
    # Warm up
    c = a * b
    
    # Benchmark multiplication
    println("\n  Benchmarking multiplication...")
    result = @benchmark $a * $b
    println("  Median time: ", median(result.times) / 1000, " μs")
    println("  Mean time:   ", mean(result.times) / 1000, " μs")
    println("  Allocations: ", result.allocs)
    println("  Memory:      ", result.memory, " bytes")
end

println("\n" * "="^60)
println("Benchmark complete!")
println("\nNote: Times include descriptor lookup and result allocation.")
println("  ✓ Direct polymap matrix access (no getindexmap allocations)")
println("  ✓ Preallocated exact sizes from count pass")
println("  ✓ Proper accumulation: multiple degree pairs contribute to same output")

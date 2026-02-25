# TPSA.jl Test Suite

This directory contains the test suite for the TPSA.jl package, following Julia package conventions.

## Running Tests

To run all tests:
```julia
using Pkg
Pkg.test("TPSA")
```

Or from the package directory:
```julia
julia --project=. -e 'using Pkg; Pkg.test()'
```

To run a specific test file:
```julia
using Test, TPSA
include("test/polymap_tests.jl")
```

## Test Structure

- **runtests.jl**: Main test entry point that runs all test suites
- **polymap_tests.jl**: Tests for polynomial index mapping (`PolyMap`, `decomposite`, etc.)
- **index_tests.jl**: Tests for index correctness in multiplication operations
- **multiplication_tests.jl**: Tests for multiplication correctness (basic, sparse, dense, complex)
- **type_stability_tests.jl**: Tests for type stability and concrete types
- **threadsafe_tests.jl**: Tests for thread safety and descriptor caching

## Test Coverage

The test suite covers:
1. **PolyMap Functionality**: Index decomposition, bounds checking, view allocation
2. **Index Mapping**: Correctness of exponent-to-index mapping in various scenarios
3. **Multiplication**: Basic arithmetic, sparse/dense cases, complex numbers
4. **Type Stability**: Concrete types, type inference, schedule types
5. **Thread Safety**: Descriptor caching, immutability, deterministic results

## Writing New Tests

Follow Julia testing conventions:
```julia
@testset "Description of test group" begin
    @test condition
    @test_throws ErrorType function_call()
    @test value ≈ expected_value
end
```

Use `@inferred` to test type stability:
```julia
@inferred function_call(args...)
```

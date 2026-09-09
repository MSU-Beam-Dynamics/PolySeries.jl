# PolySeries.jl Test Suite

This directory contains the test suite for the PolySeries.jl package, following Julia package conventions.

## Running Tests

To run all tests:
```julia
using Pkg
Pkg.test("PolySeries")
```

Or from the package directory:
```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

To run a specific test file:
```julia
using Test, PolySeries
include("test/polymap_tests.jl")
```

## Test Structure

- **runtests.jl**: Main test entry point that runs all test suites
- **polymap_tests.jl**: Tests for polynomial index mapping (`PolyMap`, `decomposite`, etc.)
- **index_tests.jl**: Tests for index correctness in multiplication operations
- **multiplication_tests.jl**: Tests for multiplication correctness (basic, sparse, dense, complex)
- **mul_reference_tests.jl**: `mul!`, `*`, `pow!` and `^` against an independent exponent-convolution reference over random sparsity patterns, all aliasing forms, Float64/Float32/ComplexF64
- **type_stability_tests.jl**: Tests for type stability and concrete types
- **threadsafe_tests.jl**: Tests for thread safety and descriptor caching
- **documentation_examples_tests.jl**: Extracts executable README blocks into temporary scripts and runs every standalone example in a fresh Julia process
- **degree_mask_tests.jl**, **arithmetic_tests.jl**: Sparse degree gaps and poisoned inactive storage
- **order_limits_tests.jl**, **math_alias_tests.jl**, **log_pool_tests.jl**: Representation limits, aliasing, and pool cleanup on failure
- **math_alias_tests.jl**, **enzyme_alias_tests.jl**: Odd/even buffer swaps, mixed pooled/heap temporaries, and forward/reverse AD through aliased math
- **release_edge_tests.jl**: Singular square-root centers and negative-power overflow
- **error_path_tests.jl**: Guard rails (constructor, accessor, domain, workspace, composition). `@test_broken` lines are pre-release targets that flip to "Unexpected Pass" when the corresponding fix lands
- **macro_tests.jl**, **composition_tests.jl**: Expression lowering, workspace reuse, and composition
- **ext_enzyme_test.jl**: Enzyme regressions, including zero coefficients, prebuilt inputs, descriptor changes, and aliasing
- **enzyme_normalized_series_tests.jl**: Forward and reverse sensitivities of scaled high-order coefficients; ordinary, in-place, and aliased coefficient checks are in **order_limits_tests.jl**

Use `Pkg.test()` for the full suite: it supplies Enzyme, Printf, and
LinearAlgebra as test extras. Directly including the test entry point in the
package environment does not install those extras.


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

# PolySeries.jl Examples

This directory contains self-contained scripts demonstrating how to use
PolySeries.jl. Each script imports the standard libraries it needs.

## Example Files

### 01_basic_operations.jl
Learn the fundamentals:
- Creating TPSA variables
- Addition and subtraction
- Multiplication
- Building polynomials
- Polynomial expansion

### 02_math_functions.jl
Mathematical functions with TPSA:
- Exponential and logarithm (exp, log)
- Trigonometric functions (sin, cos, tan)
- Hyperbolic functions (sinh, cosh, tanh)
- Power functions (pow, sqrt)
- Combined operations

### 03_plugin_values.jl
Evaluating TPSA at specific points:
- Plugging in numerical values
- Evaluating a series with its call syntax
- Extracting specific coefficients
- Partial substitution
- Using TPSA results for numerical computation

### 04_index_mapping.jl
Working with indices and monomials:
- Understanding the index-to-monomial mapping
- Finding specific terms in a TPSA
- Searching for indices of given monomials
- Iterating through terms by degree
- Extracting coefficients efficiently

### 05_derivatives_integration.jl
Calculus operations:
- Computing partial derivatives
- Numerical differentiation from coefficients
- Integration (antiderivatives)
- Function composition
- Chain rule applications

### 06_matrix_construction.jl
Building matrices from TPSA results:
- Extracting Jacobian matrices (transfer matrices)
- Building offset vectors

### 07_enzyme_ad.jl
Combining TPSA with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) for nested AD:
- Differentiating Taylor map coefficients w.r.t. scalar design parameters
- Two levels of differentiation: TPSA (phase-space) + Enzyme (parameter sensitivity)
- Multi-output Jacobians
- Finite-difference verification
- Notes on differentiating allocating TPSA operations
- Extracting higher-order tensors
- Symplecticity checks
- Applications in beam dynamics and nonlinear systems

## Running Examples

From the package checkout, run any example directly:
```julia
julia --project=. examples/01_basic_operations.jl
```

Or include within a Julia session:
```julia
using PolySeries
include("examples/02_math_functions.jl")
```

## Prerequisites

Examples 01–06 need only PolySeries.jl and Julia's standard libraries. Example
07 also needs Enzyme.jl, which is included in the package test environment.

## Learning Path

Recommended order for learning:
1. Start with `01_basic_operations.jl` to understand TPSA fundamentals
2. Move to `02_math_functions.jl` for function operations
3. Learn coefficient access with `04_index_mapping.jl`
4. Practice evaluation with `03_plugin_values.jl`
5. Explore calculus in `05_derivatives_integration.jl`
6. Apply to real problems with `06_matrix_construction.jl`

## Development Scripts

Old debugging and development scripts have been moved to `../dev_scripts/` for reference.

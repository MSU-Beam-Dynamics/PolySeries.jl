# TPSA.jl Documentation

Welcome to the TPSA.jl (Truncated Power Series Algebra) documentation.

## Overview

TPSA.jl is a Julia package for automatic differentiation and nonlinear analysis using truncated power series algebra. It efficiently computes Taylor expansions of arbitrary functions to high orders, making it ideal for:

- Beam dynamics and accelerator physics
- Nonlinear dynamics and chaos theory
- Perturbation analysis
- Automatic differentiation to arbitrary order
- Normal form analysis

## Quick Start

```julia
using TPSA

# Set up the global descriptor (do this once at the start)
set_descriptor!(3, 4)  # 3 variables, maximum order 4

# Create variables - simple and clean!
x = CTPS(0.0, 1)  # variable x
y = CTPS(0.0, 2)  # variable y
z = CTPS(0.0, 3)  # variable z

# Create constants
c = CTPS(5.0)     # constant 5.0

# Perform calculations
f = x^2 + 2*x*y + y^2
g = exp(x) * sin(y)
h = (1 + x + y)^3
```

## Table of Contents

- [Installation](#installation)
- [Basic Concepts](#basic-concepts)
- [Core Types](#core-types)
- [Operations](#operations)
- [Mathematical Functions](#mathematical-functions)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Performance Tips](#performance-tips)

## Installation

```julia
using Pkg
Pkg.add("TPSA")
```

Or from the Julia REPL package manager:
```
] add TPSA
```

## Basic Concepts

### Truncated Power Series

A truncated power series represents a multivariate function as:

$$f(x_1, x_2, ..., x_n) = \sum_{|\alpha| \leq d} c_\alpha x_1^{\alpha_1} x_2^{\alpha_2} \cdots x_n^{\alpha_n}$$

where $|\alpha| = \alpha_1 + \alpha_2 + ... + \alpha_n$ is the total degree and $d$ is the maximum order.

### Variables and Descriptors

- **Variables (nv)**: Number of independent variables in your system
- **Order**: Maximum degree of monomials to track
- **Descriptor**: Shared metadata that defines the TPSA space

All TPSA objects with the same `(nv, order)` share a descriptor for efficiency.

## Core Types

### CTPS (Complex/Real TPSA)

```julia
CTPS(value::Number, nv::Int, order::Int, [var_index::Int])
```

Creates a TPSA object representing either:
- A constant (if `var_index` is omitted)
- A variable (if `var_index` is provided)

**Parameters:**
- `value`: Initial constant value
- `nv`: Number of variables
- `order`: Maximum order
- `var_index`: Optional index to make this a variable (2 ≤ var_index ≤ nv+1)

**Examples:**
```julia
# Create a constant
c = CTPS(5.0, 3, 4)

# Create variables
x = CTPS(1.0, 3, 4, 2)  # First variable
y = CTPS(1.0, 3, 4, 3)  # Second variable
z = CTPS(1.0, 3, 4, 4)  # Third variable
```

### TPSADesc

The descriptor contains:
- Polynomial index mapping (PolyMap)
- Multiplication schedules (optimized for CPU and GPU)
- Reverse lookup tables
- Cached computation plans

Descriptors are automatically cached and reused for efficiency.

## Operations

### Arithmetic Operations

All standard arithmetic operations are supported:

```julia
# Addition and subtraction
result = x + y
result = x - y
result = 2*x + 3*y

# Multiplication
result = x * y
result = (1 + x) * (1 + y)

# Division (if implemented)
result = x / y

# Powers
result = x^3
result = (x + y)^2
```

### Comparison

```julia
# Coefficients can be compared
isapprox(tps1.c, tps2.c, rtol=1e-10)
```

## Mathematical Functions

TPSA.jl supports a wide range of mathematical functions:

### Exponential and Logarithmic
```julia
exp(x)      # Exponential
log(x)      # Natural logarithm
log10(x)    # Base-10 logarithm
```

### Trigonometric
```julia
sin(x)      # Sine
cos(x)      # Cosine
tan(x)      # Tangent
asin(x)     # Arcsine
acos(x)     # Arccosine
atan(x)     # Arctangent
```

### Hyperbolic
```julia
sinh(x)     # Hyperbolic sine
cosh(x)     # Hyperbolic cosine
tanh(x)     # Hyperbolic tangent
```

### Power and Root
```julia
sqrt(x)     # Square root
pow(x, n)   # Power function
```

## Examples

See the `examples/` directory for comprehensive examples:

1. **Basic Operations** - Addition, multiplication, polynomials
2. **Math Functions** - Trigonometric, exponential, etc.
3. **Evaluation** - Plugging in numerical values
4. **Index Mapping** - Accessing specific coefficients
5. **Derivatives** - Computing derivatives and integrals
6. **Matrix Construction** - Building Jacobians and transfer matrices

## API Reference

### Core Functions

#### `CTPS`
```julia
CTPS(value, nv, order, [var_index])
```
Create a TPSA object.

#### `element`
```julia
element(tpsa::CTPS, index::Int) -> Number
```
Get coefficient at specific index.

#### `findindex`
```julia
findindex(desc::TPSADesc, exponents) -> Int
```
Find index corresponding to monomial with given exponents.

#### `assign!`
```julia
assign!(dest::CTPS, src::CTPS)
```
Copy coefficients from `src` to `dest`.

### Mathematical Functions

All functions from `Base` and `SpecialFunctions` that make sense for TPSA are supported:
- `exp, log, sqrt, pow`
- `sin, cos, tan, asin, acos, atan`
- `sinh, cosh, tanh`
- `+, -, *, /, ^`

## Performance Tips

### 1. Reuse Descriptors
TPSA objects with the same `(nv, order)` automatically share descriptors. Create them consistently.

### 2. Avoid Unnecessary Allocations
```julia
# Good: Reuse output
result = CTPS(Float64, nv, order)
TPSA.mul!(result, x, y)

# Less optimal: Creates new object
result = x * y
```

### 3. Order Selection
- Use the minimum order needed for your analysis
- Higher orders have exponentially more terms: $\binom{n+d}{d}$
- For `nv=4, order=6`: 210 coefficients
- For `nv=4, order=10`: 1001 coefficients

### 4. Type Stability
Use concrete types consistently:
```julia
# Good
x = CTPS(Float64, nv, order)

# Avoid type instability
x = CTPS(some_dynamic_value, nv, order)
```

### 5. Parallel Computation
TPSA.jl has thread-safe operations for parallel processing:
```julia
using Base.Threads
@threads for i in 1:n
    results[i] = compute_tpsa(inputs[i])
end
```

## Advanced Topics

### Index Mapping

The package uses an efficient index-to-monomial mapping:

```julia
desc = tpsa.desc
exp_vec = TPSA.getindexmap(desc.polymap, index)
# exp_vec[1] = degree
# exp_vec[2:end] = exponents for each variable
```

### Multiplication Schedules

Two schedule formats are available:
- **Input-major**: Optimized for sequential CPU execution
- **Output-major**: Optimized for parallel/GPU execution

### Composition

Function composition is supported through direct evaluation:
```julia
# Compose h(g(x)) where g(x) = 1+x and h(u) = u^2
g = 1 + x
h = g^2  # Result is (1+x)^2
```

## Contributing

Contributions are welcome! Please see the development guidelines in the repository.

## References

- Original C++ implementation: PyTPSA by Y. Hao
- Julia port: Dr. Jinyu Wan (JuTrack package)
- Current optimizations: Memory and performance improvements

## License

[Include license information]

## Citation

If you use TPSA.jl in your research, please cite:

```bibtex
[Citation information to be added]
```

## Support

For questions and issues:
- GitHub Issues: [repository URL]
- Documentation: [docs URL]
- Examples: See `examples/` directory

---

*This documentation corresponds to TPSA.jl version X.Y.Z*

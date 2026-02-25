# API Reference

Complete API reference for TPSA.jl

## Types

### `CTPS{T<:Number}`

The main TPSA type representing a truncated power series.

**Fields:**
- `c::Vector{T}`: Coefficient vector
- `desc::TPSADesc`: Descriptor (shared metadata)
- `degree_mask::Vector{Bool}`: Degree activation mask

**Constructors:**

```julia
CTPS(value::T, nv::Int, order::Int) where T<:Number
```
Create a constant TPSA.

```julia
CTPS(value::T, nv::Int, order::Int, var_index::Int) where T<:Number
```
Create a TPSA representing a variable.

**Examples:**
```julia
# Constant
c = CTPS(5.0, 3, 4)

# Variable
x = CTPS(1.0, 3, 4, 2)
```

### `TPSADesc`

Descriptor containing shared metadata for TPSA operations.

**Fields:**
- `nv::Int`: Number of variables
- `order::Int`: Maximum order
- `N::Int`: Total number of coefficients
- `Nd::Vector{Int}`: Number of terms per degree
- `off::Vector{Int}`: Offset for each degree
- `polymap::PolyMap`: Index to exponent mapping
- `exp_to_idx::Dict`: Reverse mapping
- `mul::Vector{MulScheduleInputMajor}`: Input-major multiplication schedules
- `mul_output::Vector{MulScheduleOutputMajor}`: Output-major schedules
- `comp_plan::CompPlan`: Composition plan

### `PolyMap`

Polynomial index mapping structure.

**Fields:**
- `dim::Int`: Number of dimensions
- `max_order::Int`: Maximum order
- `map::Matrix{Int}`: Index to exponent matrix

## Core Functions

### Creation

```julia
CTPS(value, nv, order, [var_index])
```
Create a TPSA object.

**Arguments:**
- `value`: Initial value (constant term)
- `nv`: Number of variables
- `order`: Maximum order
- `var_index`: Optional variable index (2 to nv+1)

**Returns:** `CTPS` object

### Assignment

```julia
assign!(dest::CTPS, src::CTPS)
```
Copy all coefficients from `src` to `dest`.

```julia
reassign!(tpsa::CTPS, value)
```
Reset TPSA to a new value.

### Coefficient Access

```julia
element(tpsa::CTPS, index::Int)
```
Get coefficient at given index.

```julia
cst(tpsa::CTPS)
```
Get constant term (coefficient at index 1).

### Index Operations

```julia
findindex(desc::TPSADesc, exponents)
```
Find index for monomial with given exponent vector.

```julia
TPSA.getindexmap(polymap::PolyMap, index::Int)
```
Get exponent vector for given index.

**Returns:** View of exponent array `[degree, exp1, exp2, ...]`

## Arithmetic Operations

### Basic Arithmetic

```julia
+(x::CTPS, y::CTPS)
+(x::CTPS, c::Number)
+(c::Number, x::CTPS)
```
Addition of TPSA objects or with scalars.

```julia
-(x::CTPS, y::CTPS)
-(x::CTPS, c::Number)
-(c::Number, x::CTPS)
-(x::CTPS)  # Unary minus
```
Subtraction and negation.

```julia
*(x::CTPS, y::CTPS)
*(x::CTPS, c::Number)
*(c::Number, x::CTPS)
```
Multiplication.

```julia
/(x::CTPS, c::Number)
```
Division by scalar.

```julia
^(x::CTPS, n::Integer)
```
Integer power.

### In-place Operations

```julia
TPSA.mul!(result::CTPS, x::CTPS, y::CTPS)
```
In-place multiplication: `result = x * y`

```julia
TPSA.update_degree_mask!(tpsa::CTPS)
```
Update the degree activation mask based on non-zero coefficients.

## Mathematical Functions

### Exponential and Logarithmic

```julia
TPSA.exp(x::CTPS)
```
Exponential function $e^x$.

```julia
TPSA.log(x::CTPS)
```
Natural logarithm.

```julia
TPSA.log10(x::CTPS)
```
Base-10 logarithm.

### Trigonometric Functions

```julia
TPSA.sin(x::CTPS)
TPSA.cos(x::CTPS)
TPSA.tan(x::CTPS)
```
Sine, cosine, and tangent.

```julia
TPSA.asin(x::CTPS)
TPSA.acos(x::CTPS)
TPSA.atan(x::CTPS)
```
Inverse trigonometric functions.

### Hyperbolic Functions

```julia
TPSA.sinh(x::CTPS)
TPSA.cosh(x::CTPS)
TPSA.tanh(x::CTPS)
```
Hyperbolic sine, cosine, and tangent.

### Power and Root Functions

```julia
TPSA.sqrt(x::CTPS)
```
Square root.

```julia
TPSA.pow(x::CTPS, n::Int)
```
Power function $x^n$.

## Internal Functions

### Index Mapping

```julia
TPSA.decomposite(n::Int, dim::Int)
```
Decompose integer to exponent vector.

**Arguments:**
- `n`: Index to decompose
- `dim`: Number of dimensions

**Returns:** Vector of exponents `[degree, exp1, exp2, ..., exp_dim]`

### Descriptor Management

Descriptors are automatically cached. Manual cache access:

```julia
TPSA.DESC_CACHE
```
Global cache dictionary mapping `(nv, order)` to `TPSADesc`.

## Utilities

### Display

```julia
Base.show(io::IO, tpsa::CTPS)
```
Display TPSA object with non-zero coefficients.

### Type Queries

```julia
eltype(tpsa::CTPS)
```
Get element type of coefficient vector.

```julia
length(tpsa::CTPS)
```
Get number of coefficients.

## Example Usage

### Basic Example

```julia
using TPSA

# Setup
nv, order = 3, 4
x = CTPS(1.0, nv, order, 2)
y = CTPS(1.0, nv, order, 3)

# Operations
f = x^2 + 2*x*y + y^2
g = TPSA.exp(x) * TPSA.sin(y)

# Access coefficients
constant = f.c[1]
desc = f.desc
exp_vec = TPSA.getindexmap(desc.polymap, 5)
```

### Advanced Example

```julia
# Extract Jacobian matrix
outputs = [f1, f2, f3, f4]  # CTPS objects
jacobian = zeros(4, 4)

for (i, output) in enumerate(outputs)
    desc = output.desc
    for j in 1:4
        # Find linear term for variable j
        for idx in 1:desc.N
            exp_vec = TPSA.getindexmap(desc.polymap, idx)
            if exp_vec[1] == 1 && exp_vec[j+1] == 1
                jacobian[i, j] = output.c[idx]
                break
            end
        end
    end
end
```

## Performance Considerations

### Memory Layout

Coefficients are stored contiguously in the `c` vector ordered by:
1. Degree (0, 1, 2, ...)
2. Lexicographic order within each degree

### Multiplication Schedules

Two schedule types optimize different execution contexts:

- **Input-major**: Better cache locality for sequential CPU
- **Output-major**: Thread-safe, ideal for GPU/parallel

The package automatically uses the appropriate schedule.

### Type Stability

For best performance, use concrete types:
```julia
# Good - type stable
x = CTPS(Float64, nv, order)

# Avoid - type unstable
x = CTPS(some_value, nv, order)  # where some_value has dynamic type
```

## See Also

- [Examples](../examples/)
- [Main Documentation](index.md)
- [Package README](../README.md)

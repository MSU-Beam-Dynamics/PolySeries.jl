# PolySeries.jl Tutorial

A step-by-step introduction to the key features of PolySeries.jl.

---

## 1. Setup

Register a **task-local default descriptor** to choose the number of variables and maximum polynomial order for new objects. Each `CTPS` retains its construction descriptor: changing or clearing the default does not change existing polynomials. Operations between different descriptors throw `DimensionMismatch` before writing coefficients.

```julia
using PolySeries

set_descriptor!(4, 6)   # 4 variables, maximum order 6
desc = get_descriptor()
println("Coefficients per series: ", desc.N)   # → 210
```

---

## 2. Creating TPSA Objects

```julia
# Independent variables (expansion point = 0.0, index = 1..nv)
x1 = CTPS(0.0, 1)
x2 = CTPS(0.0, 2)
x3 = CTPS(0.0, 3)
x4 = CTPS(0.0, 4)

# Constant (scalar packaged as a CTPS)
c = CTPS(3.14)

# Zero CTPS — useful as a pre-allocated output slot
buf = CTPS(Float64)
```

`CTPS(a, i)` creates the identity map variable $x_i$ expanded around $a$:  
$$x_i(\delta) = a + \delta_i$$
Its coefficient vector satisfies `c[1] = a` and `c[i+1] = 1`.

---

## 3. Arithmetic

Standard Julia operators work naturally:

```julia
f = x1^2 + 2*x1*x2 + x2^2    # (x1 + x2)²
g = (1 + x1)*(1 + x2)         # 1 + x1 + x2 + x1*x2
h = 3*x1 - x2/2               # mixed scalar/CTPS
k = g^3                        # integer power via binary exponentiation
```

All operations return a new `CTPS`; the inputs are not modified.

---

## 4. Mathematical Functions

Transcendental functions expand automatically in Taylor series around the constant term:

```julia
set_descriptor!(2, 8)
x = CTPS(0.0, 1)
y = CTPS(0.0, 2)

e_x   = exp(x)              # e^x through order 8
s_y   = sin(y)              # sin(y) through order 8
cs    = cos(x) * sin(y)
lg    = log(1 + x)          # ln(1+x)
sq    = sqrt(1 + x)         # √(1+x)
sh    = sinh(x) + cosh(x)   # should equal exp(x)

# Also available: tan, asin, acos, pow(f, n::Int)
```

---

## 5. Accessing Coefficients

Coefficients are stored in `ctps.c` indexed by degree-lexicographic order.  
Use `findindex` or `element` to look up a specific monomial by its exponent vector:

```julia
set_descriptor!(3, 4)
x = CTPS(0.0, 1);  y = CTPS(0.0, 2);  z = CTPS(0.0, 3)

f = (x + y + z)^2   # = x²+y²+z²+2xy+2xz+2yz

# Exponent vectors have length nv; entries are per-variable powers
idx_x2  = findindex(f, [2, 0, 0])   # x²
idx_xy  = findindex(f, [1, 1, 0])   # xy
idx_xyz = findindex(f, [0, 0, 0])   # constant term

println(f.c[idx_x2],  " (expected 1.0)")
println(f.c[idx_xy],  " (expected 2.0)")
println(f.c[idx_xyz], " (expected 0.0)")

# Shorthand
println(element(f, [2, 0, 0]))   # same as f.c[idx_x2]
println(cst(f))                  # constant term (index 1)
```

To iterate over all non-zero terms, use the `PolyMap`:

```julia
desc = f.desc
for i in 1:desc.N
    v = f.c[i]
    iszero(v) && continue
    exps = PolySeries.getindexmap(desc.polymap, i)   # returns view [degree, e1, e2, e3]
    println("degree=", exps[1], " exponents=", exps[2:end], " coeff=", v)
end
```

---

## 6. Extracting Derivatives and the Jacobian

Because the coefficients *are* the Taylor coefficients, derivatives come for free:

$$\frac{\partial^{|\alpha|} f}{\partial x^\alpha}\bigg|_0 = \alpha!\; c_\alpha$$

```julia
set_descriptor!(2, 4)
x = CTPS(0.0, 1);  y = CTPS(0.0, 2)

f = x^3 + 2*x^2*y + x*y^2 + y^3

# ∂f/∂x at 0:  coefficient of x¹ times 1! = 3!*0 + ... (only pure x^3 contributes nothing linear)
# Linear terms are at degree 1
idx_x = findindex(f, [1, 0])   # x¹ coefficient = ∂f/∂x|₀
idx_y = findindex(f, [0, 1])   # y¹ coefficient = ∂f/∂y|₀
println("∂f/∂x|0 = ", f.c[idx_x])   # 0 (no linear x term)
println("∂f/∂y|0 = ", f.c[idx_y])   # 0

# Build the Jacobian of a 4D map (linear part only)
set_descriptor!(4, 3)
x1 = CTPS(0.0,1); x2 = CTPS(0.0,2); x3 = CTPS(0.0,3); x4 = CTPS(0.0,4)

nx1 = x1 + 0.1*x3
nx2 = x2 + 0.1*x4
nx3 = -0.5*x1 + x3
nx4 = -0.5*x2 + x4

outputs = [nx1, nx2, nx3, nx4]
J = [element(outputs[r], [i==c ? 1 : 0 for i in 1:4]) for r in 1:4, c in 1:4]
println("Jacobian:")
display(J)
```

---

## 7. Zero-Allocation In-Place Operations

For performance-critical loops (e.g. long-term tracking), allocating new `CTPS` objects at every step is expensive.  PolySeries.jl provides in-place primitives and a workspace pool.

### In-place arithmetic

```julia
set_descriptor!(4, 10)
desc = get_descriptor()

a = CTPS(0.0, 1);  b = CTPS(0.0, 2)
out = CTPS(Float64)   # pre-allocated output

mul!(out, a, b)                        # out = a * b
add!(out, a, b)                        # out = a + b
sub!(out, a, b)                        # out = a - b
scale!(out, a, 2.0)                    # out = 2*a
scaleadd!(out, cos(0.5), a, sin(0.5), b)   # out = cos(θ)*a + sin(θ)*b (fused)
```

### Workspace pool

`PSWorkspace` pre-allocates a pool of CTPS slots.  `borrow!` hands out a zeroed slot; `release!` returns it.

```julia
ws  = PSWorkspace(desc, 16)   # pool of 16 Float64 CTPS objects

t1 = borrow!(ws)
t2 = borrow!(ws)

mul!(t1, a, a)           # t1 = a²
mul!(t2, b, b)           # t2 = b²
sub!(out, t1, t2)        # out = a² - b²

release!(ws, t1)
release!(ws, t2)
# out holds the result; ws is back to full capacity
```

### In-place math functions

All major transcendental functions have `!` variants that write into a pre-allocated output:

```julia
sin!(out, a);   cos!(out, a)
exp!(out, a);   log!(out, a);   sqrt!(out, a)
sinh!(out, a);  cosh!(out, a)
pow!(out, a, 3)   # out = a^3
```

---

## 8. The `@tpsa` Macro

For complex expressions, writing the in-place chain manually is tedious.  The `@tpsa` macro compiles an arithmetic expression into optimal zero-allocation code automatically:

```julia
set_descriptor!(4, 6)
desc = get_descriptor()
ws   = PSWorkspace(desc, 20)

x1 = CTPS(0.0,1); x2 = CTPS(0.0,2)
x3 = CTPS(0.0,3); x4 = CTPS(0.0,4)

μ  = 2π * 0.205
nx1 = CTPS(0.0, 1)   # pre-allocated output

# Expands to a series of mul!/add!/sub! calls — zero heap allocations
@tpsa ws  nx1 = cos(μ)*x1 + sin(μ)*(x2 + x1^2 - x3^2)
```

**Rules:**
- The second argument must be an assignment `lhs = rhs`.
- `lhs` must be a pre-allocated `CTPS` object.
- Supported ops: `+`, `-`, `*`, unary `-`, `^n` (Int), `sin`, `cos`, `exp`, `log`, `sqrt`, `sinh`, `cosh`.
- Scalars (`Real`) may appear as either operand to `+`, `-`, `*`.

---

## 9. Thread Safety

Descriptors may be shared. Each task must initialize its own default or pass a descriptor explicitly. Workspaces must not be shared by concurrent tasks:

```julia
using Base.Threads

results = Vector{Float64}(undef, nthreads())
@threads for tid in 1:nthreads()
    set_descriptor!(4, 6)           # task-local initialization
    desc = get_descriptor()
    ws   = PSWorkspace(desc, 16)
    # ... compute ...
end
```

---

## Next steps

- See [API Reference](api.md) for the complete function reference.
- See the `examples/` directory for self-contained runnable scripts.
- Run the scripts in `benchmarks/` to measure performance on your own hardware.

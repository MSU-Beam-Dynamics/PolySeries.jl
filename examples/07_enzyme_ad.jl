# Combining TPSA with Enzyme.jl for nested automatic differentiation
#
# TPSA computes a full Taylor map (all partial derivatives up to the chosen
# order w.r.t. the phase-space expansion variables).  Enzyme.jl differentiates
# *through* that computation, giving exact first-order sensitivities of any
# Taylor coefficient w.r.t. scalar parameters — without finite differences or
# symbolic formulas.
#
# Requirements: add Enzyme once with
#   import Pkg; Pkg.add("Enzyme")

using PolySeries
using Enzyme
using Printf

println("=== PolySeries + Enzyme: nested differentiation ===\n")

# ─── Key rule ───────────────────────────────────────────────────────────
# set_descriptor! writes to task-local storage.  Enzyme does not re-execute
# side-effectful global mutations in its reverse pass, so the descriptor MUST
# be set *outside* the differentiated function.  All CTPS constructors and
# arithmetic / math functions are fully compatible with Enzyme.
# ────────────────────────────────────────────────────────────────────────────

# ─── Example 1: d/dx of a transcendental function ───────────────────────────
# f(x₀) = exp(x₀)   →   f'(x₀) = exp(x₀)
# Use the expansion center x₀ as the differentiation parameter.

set_descriptor!(1, 10)    # ← OUTSIDE the differentiated function

exp_value = exp(CTPS(0.0, 1))

x0 = 1.0
val  = exp_value(x0)
grad = Enzyme.gradient(Reverse, exp_value, x0)[1]
@printf("exp(x) at x=1:  value = %.6f   Enzyme grad = %.6f   exact = %.6f\n",
        val, grad, exp(1.0))
@assert abs(grad - exp(x0)) < 1e-6
println("Example 1 passed ✓\n")


# ─── Example 2: all seven supported math functions ────────────────────────────

set_descriptor!(1, 4)

math_fns = [
    ("exp",  x0 -> exp(CTPS(0.0, 1))(x0),  x0 ->  exp(x0)),
    ("log",  x0 -> log(CTPS(0.0, 1))(x0),  x0 ->  1/x0),
    ("sqrt", x0 -> sqrt(CTPS(0.0, 1))(x0), x0 ->  1/(2*sqrt(x0))),
    ("sin",  x0 -> sin(CTPS(0.0, 1))(x0),  x0 ->  cos(x0)),
    ("cos",  x0 -> cos(CTPS(0.0, 1))(x0),  x0 -> -sin(x0)),
    ("sinh", x0 -> sinh(CTPS(0.0, 1))(x0), x0 ->  cosh(x0)),
    ("cosh", x0 -> cosh(CTPS(0.0, 1))(x0), x0 ->  sinh(x0)),
]
x0 = 0.8
println("First derivatives via Enzyme at x=$x0 (descriptor set outside):")
for (name, fn, exact) in math_fns
    g = Enzyme.gradient(Reverse, fn, x0)[1]
    e = exact(x0)
    @printf("  %-6s  Enzyme = % .10f   exact = % .10f   ok = %s\n",
            name, g, e, abs(g - e) < 1e-6)
    @assert abs(g - e) < 1e-6  "Failed for $name"
end
println("Example 2 passed ✓\n")


# ─── Example 3: arbitrary Taylor coefficient via element() ────────────────────
# The k-th coefficient of exp(x₀ + δx) is exp(x₀)/k!.
# Its derivative w.r.t. x₀ is also exp(x₀)/k!.

set_descriptor!(1, 5)

function exp_coeff(x0::Float64, k::Int)
    t = CTPS(x0, 1)
    return element(exp(t), [k])   # use short exponent vector [k] for 1 variable
end

x0 = 0.5
println("Taylor coefficients and their x₀-derivatives for exp(x) at x₀=$x0:")
for k in 0:5
    cv = exp_coeff(x0, k)
    cg = Enzyme.gradient(Reverse, x0 -> exp_coeff(x0, k), x0)[1]
    ex = exp(x0) / factorial(k)
    @printf("  k=%d  coeff = %.8f   d/dx₀ = %.8f   expected = %.8f   ok = %s\n",
            k, cv, cg, ex, abs(cg - ex) < 1e-10)
    @assert abs(cg - ex) < 1e-10
end
println("Example 3 passed ✓\n")


# ─── Example 4: multi-variable TPSA with Enzyme ────────────────────────────
# f(x₀, y₀=0.5) = sin(x)*cos(y) + exp(x)  →  ∂f/∂x₀ = cos(x₀)cos(y₀) + exp(x₀)

set_descriptor!(2, 10)    # 2 variables, set outside

x = CTPS(0.0, 1)
y = CTPS(0.0, 2)
f_multi = sin(x) * cos(y) + exp(x)

x0, y0 = 0.7, 0.5
g_multi = Enzyme.gradient(Reverse, x0 -> f_multi(x0, y0), x0)[1] # Define fixed y0
exact   = cos(x0)*cos(y0) + exp(x0)
@printf("∂/∂x₀[sin(x)cos(y)+exp(x)] at (%.1f, %.1f):  Enzyme = %.8f   exact = %.8f   ok = %s\n",
        x0, y0, g_multi, exact, abs(g_multi - exact) < 1e-10)
@assert abs(g_multi - exact) < 1e-10
println("Example 4 passed ✓\n")


# ─── Example 5: gradient of a higher-order coefficient via element() ──────────
# Coefficient of x¹y⁰ in sin(x)*cos(y):  cos(x₀)*cos(y₀)
# ∂/∂x₀ = -sin(x₀)*cos(y₀)

set_descriptor!(2, 4)

function linear_coeff(x0::Float64)
    x = CTPS(x0, 1)
    y = CTPS(0.5, 2)
    f = sin(x) * cos(y)
    return element(f, [1, 0])   # exponents [e_x, e_y]
end

x0 = 0.5
g_lc  = Enzyme.gradient(Reverse, linear_coeff, x0)[1]
exact = -sin(x0) * cos(0.5)
@printf("∂/∂x₀[coeff x¹y⁰ of sin(x)cos(y)] at x₀=%.1f:  Enzyme = %.8f   exact = %.8f   ok = %s\n",
        x0, g_lc, exact, abs(g_lc - exact) < 1e-10)
@assert abs(g_lc - exact) < 1e-10
println("Example 5 passed ✓\n")


# ─── Summary of rules ───────────────────────────────────────────────────────────────
println("""
Key rules for PolySeries + Enzyme
────────────────────────────
1. Call set_descriptor!(nv, order) OUTSIDE the function passed to Enzyme.
   Enzyme does not re-execute task-local storage mutations in its reverse pass.

2. Use the expansion center as the differentiation parameter:
     x = CTPS(x0, var_index)    ←  x0 is the scalar Float64 parameter

3. All allocating arithmetic and math functions are Enzyme-compatible:
     +, -, *, /, ^n,  exp, log, sqrt, sin, cos, tan, asin, acos, sinh, cosh

4. Access results with cst(f) or element(f, exponents) — both work.

5. Do NOT differentiate through in-place/pool-based variants:
     exp!(out, f),  sin!(out, f),  mul!(out, a, b), …
   These mutate shared workspace slots that Enzyme cannot trace through.
   Use the allocating forms (exp, sin, *, etc.) inside differentiated code.
""")

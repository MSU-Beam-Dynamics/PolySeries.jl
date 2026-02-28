# ==============================================================================
# Auto-differentiation with TPSA
#
# TPSA provides two complementary differentiation mechanisms:
#
#   1. Built-in TPS derivatives (fast, exact, zero overhead)
#      – CTPS coefficients *are* the Taylor-map derivatives.
#      – element(ctps, [i,j,...]) returns the exact partial derivative ∂^n f/∂x^i∂y^j...
#
#   2. AD through CTPS coefficients (coefficient-level sensitivity)
#      – Useful when you want d/dθ of a *map coefficient* with respect to a
#        physical parameter θ (e.g., magnet strength, drift length).
#      – Two supported frameworks:
#          a) ForwardDiff.jl   — works out of the box; ZERO code changes needed.
#          b) Enzyme.jl        — works via the TPSAEnzymeExt package extension.
#
# Dependencies (install once):
#   julia> using Pkg; Pkg.add(["ForwardDiff", "BenchmarkTools", "Enzyme"])
# ==============================================================================

using TPSA
using BenchmarkTools
using Printf

# ------------------------------------------------------------------------------
# Section 1: ForwardDiff through CTPS coefficients
#
# Because CTPS{T} is fully generic in T, ForwardDiff's Dual numbers propagate
# through every arithmetic and math operation with no special handling.
# ------------------------------------------------------------------------------

println("="^70)
println("AD Through TPSA Map Coefficients — ForwardDiff")
println("="^70)

using ForwardDiff

set_descriptor!(1, 8)

# ── Example 1: exp(k·x) ────────────────────────────────────────────────────
# exp(k·x) = 1 + k·x + k²/2·x² + k³/6·x³ + ...
# c₂ (linear coeff) = k           →  dc₂/dk = 1
# c₃ (quadratic coeff) = k²/2     →  dc₃/dk = k
# c₄ (cubic coeff) = k³/6         →  dc₄/dk = k²/2

function coeff_exp(k)
    x = CTPS(zero(k), 1)
    r = exp(k * x)
    return r.c[2]     # linear coefficient = k
end

function coeff_exp_cubic(k)
    x = CTPS(zero(k), 1)
    r = exp(k * x)
    return r.c[4]     # cubic coefficient = k³/6
end

k0 = 2.0

println("\n--- exp(k·x) ---")
g_linear = ForwardDiff.derivative(coeff_exp, k0)
println("  dc₂/dk at k=$k0: got $g_linear  expected 1.0")

g_cubic = ForwardDiff.derivative(coeff_exp_cubic, k0)
println("  dc₄/dk at k=$k0: got $g_cubic   expected $(k0^2/2)")

# ── Example 2: The Jacobian of multiple map coefficients w.r.t. multiple params
# sin(k·x): c₄ = -k³/6 → dc₄/dk = -k²/2
function coeff_sin_cubic(k)
    x = CTPS(zero(k), 1)
    r = sin(k * x)
    return r.c[4]    # = -k³/6
end

println("\n--- sin(k·x) ---")
g_sin = ForwardDiff.derivative(coeff_sin_cubic, k0)
println("  dc₄/dk at k=$k0: got $g_sin  expected $(-k0^2/2)")

# ── Example 3: Gradient w.r.t. 2 parameters simultaneously (using gradient)
# f(a, b) = coefficient of x² in (a + b·x)^3
# = coefficient of x² in a³ + 3a²bx + 3ab²x² + b³x³
# = 3ab²
# ∇ = [3b², 6ab]

function coeff_poly(params)
    a, b = params[1], params[2]
    x = CTPS(zero(a), 1)
    r = (a + b * x)^3
    return r.c[3]    # x² coefficient = 3ab²
end

params0 = [2.0, 3.0]
grad = ForwardDiff.gradient(coeff_poly, params0)
a0, b0 = params0
println("\n--- (a + b·x)³ quad coeff gradient ---")
println("  ∇(3a·b²) at a=$a0, b=$b0: got $grad  expected $([ 3b0^2, 6a0*b0 ])")

# ── Example 4: Physics case — thin sextupole map
# A thin sextupole of integrated strength K₂L adds a kick proportional to x².
# The output is  x_out = x - K₂L/6 · x²  (normalized coordinates).
# The quadratic coefficient of x_out is  c₃ = -K₂L/6
# → dc₃/d(K₂L) = -1/6

function sext_coeff(K2L)
    x = CTPS(zero(K2L), 1)
    r = x - (K2L / 6) * x^2
    return r.c[3]    # = -K2L/6
end

K2L0 = 3.0
g_sext = ForwardDiff.derivative(sext_coeff, K2L0)
println("\n--- Thin sextupole: dc₃/d(K₂L) ---")
println("  at K₂L=$K2L0: got $g_sext  expected $(-1/6)")

# ── Benchmark ForwardDiff vs finite differences ─────────────────────────────
println("\n--- Benchmark ---")
h = 1e-7

println("ForwardDiff.derivative(coeff_exp, k0):")
b1 = @benchmark ForwardDiff.derivative($coeff_exp, $k0)
display(b1); println()

println("Finite difference ((f(k+h)-f(k))/h):")
b2 = @benchmark (coeff_exp($k0 + $h) - coeff_exp($k0)) / $h
display(b2); println()

# ------------------------------------------------------------------------------
# Section 2: Enzyme.jl — first and second derivatives (pure Enzyme, no ForwardDiff)
#
# Two Enzyme modes are available:
#
#   a) gradient(Reverse, f, k)
#        → df/dk via reverse-mode AD.  Works for all allocating TPSA ops.
#
#   b) autodiff(set_runtime_activity(Forward), f, Duplicated(k, 1.0))[1]
#        → df/dk via forward-mode AD.  Requires set_runtime_activity because
#          CTPS uses lazy-undef allocation (the degree_mask invariant means
#          only active positions are written; Enzyme's static analysis sees
#          the uninitialized positions and conservatively raises an error
#          without the runtime flag).
#
#   c) Composing (b) with itself → d²f/dk² entirely within Enzyme.
#
# The TPSAEnzymeExt package extension (loaded automatically alongside Enzyme)
# registers inactive_type rules for all TPSA-internal types (TPSADesc,
# DescPool, PolyMap, MulSchedule2D, CompPlan) — no user setup required.
# ------------------------------------------------------------------------------

println("="^70)
println("Enzyme.jl — First and Second Derivatives of TPSA Coefficients")
println("="^70)

using Enzyme

# ── Setup ─────────────────────────────────────────────────────────────────────
# set_descriptor! must be called OUTSIDE the differentiated function.
# The expansion center is the differentiation parameter: CTPS(k, var_n).

set_descriptor!(1, 6)

# Functions whose scalar parameter is k:
#   f_exp(k)    = cst(exp(CTPS(k, 1)))    = exp(k)
#   f_sin(k)    = cst(sin(CTPS(k, 1)))    = sin(k)
#   f_coeff2(k) = element(exp(...), [2])  = exp(k)/2     (2nd Taylor coeff)
#   f_coeff3(k) = element(exp(...), [3])  = exp(k)/6     (3rd Taylor coeff)

f_exp    = k -> cst(exp(CTPS(k, 1)))
f_sin    = k -> cst(sin(CTPS(k, 1)))
f_coeff2 = k -> element(exp(CTPS(k, 1)), [2])  # = exp(k)/2
f_coeff3 = k -> element(exp(CTPS(k, 1)), [3])  # = exp(k)/6

# ── First derivatives via Reverse mode ───────────────────────────────────────

println("\n--- First derivatives via Enzyme Reverse mode ---")
k0 = 1.0
@printf("  d/dk exp(k)     at k=1: got %.10f  expected %.10f\n",
        Enzyme.gradient(Reverse, f_exp,    k0)[1], exp(k0))
@printf("  d/dk sin(k)     at k=1: got %.10f  expected %.10f\n",
        Enzyme.gradient(Reverse, f_sin,    k0)[1], cos(k0))
@printf("  d/dk coeff[2]   at k=1: got %.10f  expected %.10f\n",
        Enzyme.gradient(Reverse, f_coeff2, k0)[1], exp(k0)/2)
@printf("  d/dk coeff[3]   at k=1: got %.10f  expected %.10f\n",
        Enzyme.gradient(Reverse, f_coeff3, k0)[1], exp(k0)/6)

# ── First derivatives via Forward mode (set_runtime_activity) ────────────────
#
# set_runtime_activity is required because CTPS uses lazy-undef allocation:
# only positions within the active degree range are initialized, so Enzyme's
# static analysis flags them as potentially uninitialized active memory.
# Runtime activity resolves this correctly.

println("\n--- First derivatives via Enzyme Forward mode ---")
@printf("  d/dk exp(k)     at k=1: got %.10f  expected %.10f\n",
        Enzyme.autodiff(set_runtime_activity(Forward), f_exp,    Duplicated(k0, 1.0))[1], exp(k0))
@printf("  d/dk sin(k)     at k=1: got %.10f  expected %.10f\n",
        Enzyme.autodiff(set_runtime_activity(Forward), f_sin,    Duplicated(k0, 1.0))[1], cos(k0))
@printf("  d/dk coeff[2]   at k=1: got %.10f  expected %.10f\n",
        Enzyme.autodiff(set_runtime_activity(Forward), f_coeff2, Duplicated(k0, 1.0))[1], exp(k0)/2)

# ── Second derivatives via double Forward (Fwd∘Fwd) ──────────────────────────
#
# d²f/dk² = apply Forward mode to the first-derivative function.
#
#   d1(k) = autodiff(set_runtime_activity(Forward), f, Duplicated(k, 1.0))[1]
#   d2(k) = autodiff(set_runtime_activity(Forward), d1, Duplicated(k, 1.0))[1]
#
# The outer Forward mode differentiates through the inner Forward computation,
# which itself allocates CTPS objects and runs the TPSA arithmetic/math
# functions.  set_runtime_activity on both levels handles the undef-allocation
# pattern correctly.

println("\n--- Second derivatives via Enzyme Fwd∘Fwd (no ForwardDiff) ---")

d1_exp    = k -> Enzyme.autodiff(set_runtime_activity(Forward), f_exp,    Duplicated(k, 1.0))[1]
d1_sin    = k -> Enzyme.autodiff(set_runtime_activity(Forward), f_sin,    Duplicated(k, 1.0))[1]
d1_coeff2 = k -> Enzyme.autodiff(set_runtime_activity(Forward), f_coeff2, Duplicated(k, 1.0))[1]
d1_coeff3 = k -> Enzyme.autodiff(set_runtime_activity(Forward), f_coeff3, Duplicated(k, 1.0))[1]

d2_exp    = Enzyme.autodiff(set_runtime_activity(Forward), d1_exp,    Duplicated(k0, 1.0))[1]
d2_sin    = Enzyme.autodiff(set_runtime_activity(Forward), d1_sin,    Duplicated(k0, 1.0))[1]
d2_coeff2 = Enzyme.autodiff(set_runtime_activity(Forward), d1_coeff2, Duplicated(k0, 1.0))[1]
d2_coeff3 = Enzyme.autodiff(set_runtime_activity(Forward), d1_coeff3, Duplicated(k0, 1.0))[1]

@printf("  d²/dk² exp(k)     at k=1: got %.10f  expected %.10f  (= exp(k))\n",
        d2_exp,    exp(k0))
@printf("  d²/dk² sin(k)     at k=1: got %.10f  expected %.10f  (= -sin(k))\n",
        d2_sin,    -sin(k0))
@printf("  d²/dk² coeff[2]   at k=1: got %.10f  expected %.10f  (= exp(k)/2)\n",
        d2_coeff2, exp(k0)/2)
@printf("  d²/dk² coeff[3]   at k=1: got %.10f  expected %.10f  (= exp(k)/6)\n",
        d2_coeff3, exp(k0)/6)

# ── Physics example: quadrupole gradient sensitivity ─────────────────────────
#
# A thin quadrupole of integrated strength K₁L rotates the phase-space angle:
#   x_out = x - K₁L * x   (simplest model: focusing kick)
# More interesting: the beta-function Courant-Snyder invariant in x is
#   ε = x² / β,  where β depends on K₁L via the Twiss equations.
#
# Here we use a simple 2nd-order map: a thin sextupole rotated by K1
#   x_out = cos(K1) * x - sin(K1) * px
#   px_out = sin(K1) * x + cos(K1) * px  (rotation by K1)
# and ask: what is d²/dK1² of the quadratic coefficient of x_out?
#
# Physical quantity: element of x_out (1-variable CTPS in x) quadratic coeff.
#   f(K1) = element of (cos(K1)*x - sin(K1)*1) expanded in x,
#            coefficient of x^1  = cos(K1)
#   df/dK1 = -sin(K1),   d²f/dK1² = -cos(K1)

set_descriptor!(1, 3)

function rotation_coeff(K1)
    x = CTPS(K1, 1)    # expansion around K1 in the "x" variable (unusual but valid)
    # linear term coefficient of cos(K1)*x  — expand the whole expression in x
    t = cos(K1) * x    # scalar * CTPS,  coefficient of x^1 = cos(K1)
    return element(t, [1])
end

# Simpler and cleaner: the K1 itself is the parameter
function rotation_linear_coeff(K1)
    # linear coeff of (cos(K1) * x): just cos(K1) as a function of K1
    # use CTPS(K1,1) so the expansion center carries K1-dependence
    x = CTPS(0.0, 1)
    t = cos(K1) * x        # scalar(K1) * CTPS — only scalar multiplication
    return element(t, [1]) # = cos(K1)
end

# First and second derivatives of the linear coeff w.r.t. K1
d1_rot = k -> Enzyme.autodiff(set_runtime_activity(Forward), rotation_linear_coeff, Duplicated(k, 1.0))[1]

K1 = π/6
d1_K1 = d1_rot(K1)
d2_K1 = Enzyme.autodiff(set_runtime_activity(Forward), d1_rot, Duplicated(K1, 1.0))[1]
@printf("\nThin rotation: d/dK1[cos(K1)] at K1=π/6:   got %.10f  expected %.10f\n", d1_K1, -sin(K1))
@printf("Thin rotation: d²/dK1²[cos(K1)] at K1=π/6: got %.10f  expected %.10f\n", d2_K1, -cos(K1))

println()
println("="^70)
println("Recommendation")
println("="^70)
println("""
For differentiating TPSA map coefficients w.r.t. physical parameters:

  • Use ForwardDiff.jl for first derivatives — zero setup, works for all
    TPSA operations via the T-generic CTPS{T} type.

  • Use Enzyme.jl (Reverse mode) for first derivatives — automatic via the
    TPSAEnzymeExt extension; use the expansion center as the parameter.

  • Use Enzyme.jl double-Forward (Fwd∘Fwd) for second derivatives — pure
    Enzyme, no ForwardDiff needed.  Requires set_runtime_activity on both
    levels to handle CTPS's lazy-undef allocation pattern.
    Pattern:
      d1  = k -> autodiff(set_runtime_activity(Forward), f, Duplicated(k, 1.0))[1]
      d2  =      autodiff(set_runtime_activity(Forward), d1, Duplicated(k0, 1.0))[1]
""")

# Building a Truncated Matrix from TPSA Results
# Demonstrates how to extract linear terms and build Jacobian/transfer matrices

using PolySeries
using LinearAlgebra
using Printf

println("=== Building Matrices from TPSA Results ===\n")

# Example: Particle beam dynamics or nonlinear map
# We have a map: (x', y', px', py') = F(x, y, px, py)
# where each output is a TPSA representing a nonlinear function

# Set up this task's default descriptor
set_descriptor!(4, 3)  # 4 phase space variables, order 3

println("Phase space variables: x, y, px, py")
println("Maximum order: 3")
println()

# Create initial coordinates as TPSA variables
x  = CTPS(0.0, 1)   # variable 1
y  = CTPS(0.0, 2)   # variable 2
px = CTPS(0.0, 3)   # variable 3
py = CTPS(0.0, 4)   # variable 4

# Define a nonlinear map: a thin quadrupole + sextupole kick followed by a
# drift of length L. Kick-then-drift is an exact symplectic map, so the
# symplecticity check at the end of this script holds by construction.
k = 0.5  # focusing strength
s = 0.1  # sextupole strength
L = 0.1  # drift length

px_out = px - k*x - 2*s*x^2     # kick: px' = px - ∂V/∂x, V = k(x² - y²)/2 + (2s/3)x³
py_out = py + k*y               #       py' = py - ∂V/∂y
x_out  = x + L*px_out           # drift with the kicked momenta
y_out  = y + L*py_out

println("--- Nonlinear Map Definition (kick, then drift) ---")
println("px' = px - 0.5·x - 0.2·x²")
println("py' = py + 0.5·y")
println("x'  = x + 0.1·px'  = 0.95·x + 0.1·px - 0.02·x²")
println("y'  = y + 0.1·py'  = 1.05·y + 0.1·py")
println()

# Extract the Jacobian (transfer matrix) - first order terms only
println("--- Method 1: Extract Jacobian Matrix (Linear Part) ---")
desc = x.desc

# Build 4x4 Jacobian matrix
jacobian = zeros(4, 4)

# Map output rows to input-variable columns.
outputs = [x_out, y_out, px_out, py_out]

for (row, output) in enumerate(outputs)
    for col in 1:4
        exponents = [variable == col ? 1 : 0 for variable in 1:4]
        jacobian[row, col] = element(output, exponents)
    end
end

println("Jacobian Matrix (∂output/∂input):")
println("       x        y        px       py")
display(jacobian)
println()
println()

# Extract constant terms (0th order)
println("--- Method 2: Extract Constant Terms (Offset) ---")
offset = zeros(4)
for (i, output) in enumerate(outputs)
    offset[i] = cst(output)
end
println("Offset vector: ", offset)
println()

# PolySeries.mul! is a method of LinearAlgebra.mul!, so both packages can be
# used together without qualifying the name.
x_out_sq = CTPS(Float64, desc)
mul!(x_out_sq, x_out, x_out)
@assert element(x_out_sq, [2, 0, 0, 0]) ≈ 0.95^2

# Extract second-order terms (for second-order matrix)
println("--- Method 3: Extract Second-Order Terms ---")
println("Second-order terms (truncated to x² and y² for display):")
println()

for (i, output) in enumerate(outputs)
    output_names = ["x'", "y'", "px'", "py'"]
    println("$(output_names[i]):")
    
    term_count = 0
    for idx in 1:desc.N
        exp_vec = PolySeries.getindexmap(desc.polymap, idx)
        exponents = Int.(exp_vec[2:end])
        coefficient = element(output, exponents)
        if exp_vec[1] == 2 && !iszero(coefficient)
            term_count += 1
            # Convert indices to variable names
            vars = ["x", "y", "px", "py"]
            exps = exponents
            
            # Build term string
            term = ""
            for (j, e) in enumerate(exps)
                if e > 0
                    if term != ""
                        term *= "·"
                    end
                    term *= vars[j]
                    if e > 1
                        term *= "^$e"
                    end
                end
            end
            
            @printf("  %s: %8.4f\n", term, coefficient)
        end
    end
    
    if term_count == 0
        println("  (no second-order terms)")
    end
    println()
end

println("--- Method 4: Build Full Polynomial Matrix ---")
println("For advanced applications, you can extract all orders:")
println("Order 0 (constant): offset vector")
println("Order 1 (linear):   Jacobian matrix M₁")
println("Order 2 (quadratic): tensor M₂[i,j,k]")
println("Order 3 (cubic):     tensor M₃[i,j,k,l]")
println()
println("These can be used for:")
println("  • Normal form analysis")
println("  • Perturbation theory")
println("  • Taylor map concatenation")
println("  • Symplectic tracking")
println()

# Example: Check symplecticity of linear map
println("--- Symplecticity Check (Linear Part) ---")
S = [0 0 1 0;
     0 0 0 1;
    -1 0 0 0;
     0 -1 0 0]

M_transpose_S_M = jacobian' * S * jacobian
println("M^T S M (equals S for a symplectic map):")
display(M_transpose_S_M)
println()
println("Is symplectic? ", isapprox(M_transpose_S_M, S, atol=1e-10))

@assert offset == zeros(4)
@assert jacobian ≈ [1 - L*k  0        L  0;
                    0        1 + L*k  0  L;
                    -k       0        1  0;
                    0        k        0  1]
@assert isapprox(M_transpose_S_M, S, atol=1e-12)

println("\n✓ Matrix construction demonstration completed!")

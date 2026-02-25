# Building a Truncated Matrix from TPSA Results
# Demonstrates how to extract linear terms and build Jacobian/transfer matrices

using TPSA
using LinearAlgebra

println("=== Building Matrices from TPSA Results ===\n")

# Example: Particle beam dynamics or nonlinear map
# We have a map: (x', y', px', py') = F(x, y, px, py)
# where each output is a TPSA representing a nonlinear function

# Set up global descriptor
set_descriptor!(4, 3)  # 4 phase space variables, order 3

println("Phase space variables: x, y, px, py")
println("Maximum order: 3")
println()

# Create initial coordinates as TPSA variables
x  = CTPS(0.0, 1)   # variable 1
y  = CTPS(0.0, 2)   # variable 2
px = CTPS(0.0, 3)   # variable 3
py = CTPS(0.0, 4)   # variable 4

# Define a nonlinear map (example: simple quadrupole + sextupole-like terms)
k = 0.5  # focusing strength
s = 0.1  # sextupole strength

x_out  = x + 0.1*px - k*x^2 - s*x^3
y_out  = y + 0.1*py + k*y^2  
px_out = px - k*x - 2*s*x^2
py_out = py + k*y

println("--- Nonlinear Map Definition ---")
println("x'  = x + 0.1·px - 0.5·x² - 0.1·x³")
println("y'  = y + 0.1·py + 0.5·y²")
println("px' = px - 0.5·x - 0.2·x²")
println("py' = py + 0.5·y")
println()

# Extract the Jacobian (transfer matrix) - first order terms only
println("--- Method 1: Extract Jacobian Matrix (Linear Part) ---")
desc = x.desc

# Build 4x4 Jacobian matrix
jacobian = zeros(4, 4)

# Map output to row, input variables to columns
outputs = [x_out, y_out, px_out, py_out]
input_indices = [2, 3, 4, 5]  # Coefficient indices for x, y, px, py (variables 1-4 are at indices 2-5)

for (row, output) in enumerate(outputs)
    for (col, var_idx) in enumerate(input_indices)
        # Find the linear term coefficient
        # Linear terms have degree=1 and single non-zero exponent
        for idx in 1:desc.N
            exp_vec = TPSA.getindexmap(desc.polymap, idx)
            if exp_vec[1] == 1  # degree 1 (linear)
                if exp_vec[col+1] == 1  # This variable (col+1 because exp_vec[1] is degree)
                    jacobian[row, col] = output.c[idx]
                    break
                end
            end
        end
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
    offset[i] = output.c[1]  # Index 1 is always the constant term
end
println("Offset vector: ", offset)
println()

# Extract second-order terms (for second-order matrix)
println("--- Method 3: Extract Second-Order Terms ---")
println("Second-order terms (truncated to x² and y² for display):")
println()

for (i, output) in enumerate(outputs)
    output_names = ["x'", "y'", "px'", "py'"]
    println("$(output_names[i]):")
    
    term_count = 0
    for idx in 1:desc.N
        exp_vec = TPSA.getindexmap(desc.polymap, idx)
        if exp_vec[1] == 2 && abs(output.c[idx]) > 1e-10  # degree 2
            term_count += 1
            # Convert indices to variable names
            vars = ["x", "y", "px", "py"]
            exps = exp_vec[2:5]
            
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
            
            @printf("  %s: %8.4f\n", term, output.c[idx])
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
println("M^T S M (should equal S for symplectic map):")
display(M_transpose_S_M)
println()
println("Is symplectic? ", isapprox(M_transpose_S_M, S, atol=1e-10))

println("\n✓ Matrix construction demonstration completed!")

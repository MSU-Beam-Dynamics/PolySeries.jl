# Plugging in Numerical Values
# Demonstrates how to evaluate TPSA at specific numerical points

using TPSA

println("=== Plugging in Numerical Values ===\n")

# Set up global descriptor
set_descriptor!(3, 4)

# Create a polynomial: f(x,y,z) = 1 + 2x + 3y + x^2 + xy + y^2
x = CTPS(0.0, 1)
y = CTPS(0.0, 2)
z = CTPS(0.0, 3)

f = 1 + 2*x + 3*y + x*x + x*y + y*y

println("Polynomial: f(x,y,z) = 1 + 2x + 3y + x² + xy + y²")
println()

# Method 1: Manual evaluation using coefficients
println("--- Method 1: Manual Evaluation ---")
x_val = 0.5
y_val = 0.3
z_val = 0.0

result = f.c[1]  # constant term
result += f.c[2] * x_val  # x term
result += f.c[3] * y_val  # y term

# Add second-order terms
desc = f.desc
for i in 1:desc.N
    exp_vec = TPSA.getindexmap(desc.polymap, i)
    degree = exp_vec[1]
    
    if degree == 2
        # Calculate the monomial value
        monomial = x_val^exp_vec[2] * y_val^exp_vec[3] * z_val^exp_vec[4]
        result += f.c[i] * monomial
    end
end

println("f(0.5, 0.3, 0.0) = ", result)
println()

# Verify manually
expected = 1 + 2*(0.5) + 3*(0.3) + (0.5)^2 + (0.5)*(0.3) + (0.3)^2
println("Expected (manual calculation): ", expected)
println("Match: ", isapprox(result, expected))
println()

# Method 2: Using TPSA evaluation (if implemented)
println("--- Method 2: Extract Coefficients for Custom Use ---")
println("Constant term: ", f.c[1])
println("Linear terms:")
println("  ∂f/∂x|₀: ", f.c[2])
println("  ∂f/∂y|₀: ", f.c[3])
println("  ∂f/∂z|₀: ", f.c[4])
println()

println("Quadratic terms:")
for i in 1:desc.N
    exp_vec = TPSA.getindexmap(desc.polymap, i)
    if exp_vec[1] == 2  # degree 2
        term_name = "x^$(exp_vec[2]) y^$(exp_vec[3]) z^$(exp_vec[4])"
        if f.c[i] != 0.0
            println("  $term_name: ", f.c[i])
        end
    end
end
println()

# Method 3: Substitute specific values
println("--- Method 3: Partial Substitution ---")
# Create a new TPSA with x = 0.5 substituted
# This creates g(y,z) = f(0.5, y, z)
g = CTPS(Float64)

for i in 1:desc.N
    exp_vec = TPSA.getindexmap(desc.polymap, i)
    # Evaluate x component at x=0.5
    x_contribution = 0.5^exp_vec[2]
    
    # Keep y and z as variables
    # For simplicity, we'll just show the concept
    if exp_vec[2] == 0  # Terms without x
        g.c[i] = f.c[i]
    end
end

println("After substituting x=0.5, remaining function in y,z")
println("Constant term: ", g.c[1])
println("(Note: Full partial substitution requires more complex logic)")

println("\n✓ Value plugging demonstration completed!")

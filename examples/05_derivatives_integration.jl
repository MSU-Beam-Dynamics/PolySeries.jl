# Composition, Derivatives, and Integration
# Demonstrates differentiation, integration, and function composition

using TPSA

println("=== Derivatives, Integration, and Composition ===\n")

# Set up global descriptor
set_descriptor!(2, 4)

x = CTPS(0.0, 1)
y = CTPS(0.0, 2)

println("--- Derivatives ---")
# Create a polynomial: f(x,y) = x³ + 2x²y + xy² + y³
f = x^3 + 2*(x^2)*y + x*(y^2) + y^3

println("f(x,y) = x³ + 2x²y + xy² + y³")
println()

# Partial derivative with respect to x: ∂f/∂x
# Manual computation: 3x² + 4xy + y²
println("∂f/∂x (computed from coefficients):")
df_dx = CTPS(Float64)

desc = f.desc
for idx in 1:desc.N
    exp_vec = TPSA.getindexmap(desc.polymap, idx)
    degree = exp_vec[1]
    x_exp = exp_vec[2]
    y_exp = exp_vec[3]
    
    if x_exp > 0  # Has x component
        # Derivative: reduce x exponent by 1, multiply coefficient by x_exp
        new_x_exp = x_exp - 1
        new_degree = degree - 1
        
        # Find the index for the new monomial
        for new_idx in 1:desc.N
            new_exp_vec = TPSA.getindexmap(desc.polymap, new_idx)
            if new_exp_vec[1] == new_degree && 
               new_exp_vec[2] == new_x_exp && 
               new_exp_vec[3] == y_exp
                df_dx.c[new_idx] += f.c[idx] * x_exp
                break
            end
        end
    end
end

# Display non-zero terms
for idx in 1:desc.N
    if abs(df_dx.c[idx]) > 1e-10
        exp_vec = TPSA.getindexmap(desc.polymap, idx)
        x_exp = exp_vec[2]
        y_exp = exp_vec[3]
        monomial = "x^$x_exp y^$y_exp"
        @printf("  %s: %.1f\n", monomial, df_dx.c[idx])
    end
end
println("Expected: 3x² + 4xy + y²")
println()

# Partial derivative with respect to y: ∂f/∂y
println("∂f/∂y (computed from coefficients):")
df_dy = CTPS(Float64)

for idx in 1:desc.N
    exp_vec = TPSA.getindexmap(desc.polymap, idx)
    degree = exp_vec[1]
    x_exp = exp_vec[2]
    y_exp = exp_vec[3]
    
    if y_exp > 0  # Has y component
        new_y_exp = y_exp - 1
        new_degree = degree - 1
        
        for new_idx in 1:desc.N
            new_exp_vec = TPSA.getindexmap(desc.polymap, new_idx)
            if new_exp_vec[1] == new_degree && 
               new_exp_vec[2] == x_exp && 
               new_exp_vec[3] == new_y_exp
                df_dy.c[new_idx] += f.c[idx] * y_exp
                break
            end
        end
    end
end

for idx in 1:desc.N
    if abs(df_dy.c[idx]) > 1e-10
        exp_vec = TPSA.getindexmap(desc.polymap, idx)
        x_exp = exp_vec[2]
        y_exp = exp_vec[3]
        monomial = "x^$x_exp y^$y_exp"
        @printf("  %s: %.1f\n", monomial, df_dy.c[idx])
    end
end
println("Expected: 2x² + 2xy + 3y²")
println()

println("--- Integration ---")
# Integrate x² with respect to x: ∫x² dx = x³/3
g = x^2
println("g(x,y) = x²")

int_g = CTPS(Float64)
for idx in 1:desc.N
    exp_vec = TPSA.getindexmap(desc.polymap, idx)
    degree = exp_vec[1]
    x_exp = exp_vec[2]
    y_exp = exp_vec[3]
    
    if degree < order  # Can integrate (won't exceed max order)
        new_x_exp = x_exp + 1
        new_degree = degree + 1
        
        for new_idx in 1:desc.N
            new_exp_vec = TPSA.getindexmap(desc.polymap, new_idx)
            if new_exp_vec[1] == new_degree && 
               new_exp_vec[2] == new_x_exp && 
               new_exp_vec[3] == y_exp
                int_g.c[new_idx] = g.c[idx] / new_x_exp
                break
            end
        end
    end
end

println("∫x² dx:")
for idx in 1:desc.N
    if abs(int_g.c[idx]) > 1e-10
        exp_vec = TPSA.getindexmap(desc.polymap, idx)
        x_exp = exp_vec[2]
        y_exp = exp_vec[3]
        monomial = "x^$x_exp y^$y_exp"
        @printf("  %s: %.6f\n", monomial, int_g.c[idx])
    end
end
println("Expected: x³/3 = 0.333333 x³")
println()

println("--- Composition (Conceptual) ---")
# Composition: h(g(x)) where g(x) = 1 + x and h(u) = u²
# Result: (1+x)² = 1 + 2x + x²
println("Let g(x) = 1 + x and h(u) = u²")
println("Composition h(g(x)) = (1+x)²:")

g_x = 1 + x
h_of_g = g_x^2

for idx in 1:min(10, desc.N)
    if abs(h_of_g.c[idx]) > 1e-10
        exp_vec = TPSA.getindexmap(desc.polymap, idx)
        x_exp = exp_vec[2]
        y_exp = exp_vec[3]
        monomial = "x^$x_exp y^$y_exp"
        @printf("  %s: %.1f\n", monomial, h_of_g.c[idx])
    end
end
println("Expected: 1 + 2x + x²")

println("\n✓ Calculus operations demonstration completed!")

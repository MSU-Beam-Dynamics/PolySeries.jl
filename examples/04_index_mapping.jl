# Index Mapping and Coefficient Access
# Demonstrates how to find specific terms, map between indices and exponents

using TPSA

println("=== Index Mapping and Coefficient Access ===\n")

# Set up global descriptor
set_descriptor!(3, 3)

# Create a TPSA
x = CTPS(0.0, 1)
desc = x.desc

println("TPSA Configuration:")
println("  Number of variables: ", desc.nv)
println("  Maximum order: ", desc.order)
println("  Total coefficients: ", desc.N)
println()

println("--- Method 1: Iterate Through All Indices ---")
println("First 15 indices and their corresponding monomials:")
println("Index | Degree | x^i y^j z^k | Exponents")
println("------|--------|-------------|----------")

for idx in 1:min(15, desc.N)
    exp_vec = TPSA.getindexmap(desc.polymap, idx)
    degree = exp_vec[1]
    x_exp = exp_vec[2]
    y_exp = exp_vec[3]
    z_exp = exp_vec[4]
    
    monomial = "x^$x_exp y^$y_exp z^$z_exp"
    exponents = "[$x_exp, $y_exp, $z_exp]"
    @printf("%5d | %6d | %11s | %s\n", idx, degree, monomial, exponents)
end
println()

println("--- Method 2: Find Specific Terms ---")
# Create a polynomial to search through
poly = (1 + x)^3
y_obj = CTPS(0.0, 2)
poly = poly * (1 + y_obj)^2

println("Polynomial: (1+x)³(1+y)²")
println("\nNon-zero coefficients:")

for idx in 1:desc.N
    if abs(poly.c[idx]) > 1e-10
        exp_vec = TPSA.getindexmap(desc.polymap, idx)
        x_exp = exp_vec[2]
        y_exp = exp_vec[3]
        z_exp = exp_vec[4]
        monomial = "x^$x_exp y^$y_exp z^$z_exp"
        @printf("  Index %3d: %s = %.6f\n", idx, monomial, poly.c[idx])
    end
end
println()

println("--- Method 3: Find Index for Specific Monomial ---")
# To find the index for a specific monomial (e.g., x²y)
target_x_exp = 2
target_y_exp = 1
target_z_exp = 0

found_idx = 0
for idx in 1:desc.N
    exp_vec = TPSA.getindexmap(desc.polymap, idx)
    if exp_vec[2] == target_x_exp && exp_vec[3] == target_y_exp && exp_vec[4] == target_z_exp
        found_idx = idx
        break
    end
end

if found_idx > 0
    println("Monomial x²y¹z⁰ is at index: $found_idx")
    println("Coefficient value: ", poly.c[found_idx])
else
    println("Monomial not found (may be truncated)")
end
println()

println("--- Method 4: Extract Terms by Degree ---")
println("Terms grouped by degree:")

for target_degree in 0:min(3, desc.order)
    println("\nDegree $target_degree:")
    has_terms = false
    
    for idx in 1:desc.N
        exp_vec = TPSA.getindexmap(desc.polymap, idx)
        if exp_vec[1] == target_degree && abs(poly.c[idx]) > 1e-10
            has_terms = true
            x_exp = exp_vec[2]
            y_exp = exp_vec[3]
            z_exp = exp_vec[4]
            monomial = "x^$x_exp y^$y_exp z^$z_exp"
            @printf("  %s: %.6f\n", monomial, poly.c[idx])
        end
    end
    
    if !has_terms
        println("  (no non-zero terms)")
    end
end

println("\n✓ Index mapping demonstration completed!")

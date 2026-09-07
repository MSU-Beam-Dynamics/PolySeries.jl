# Evaluating and substituting numerical values

using PolySeries

println("=== Evaluating and Substituting Values ===\n")

desc = PSDesc(3, 4)
x = CTPS(0.0, 1, desc)
y = CTPS(0.0, 2, desc)
z = CTPS(0.0, 3, desc)

# f(x,y,z) = 1 + 2x + 3y + x² + xy + y²
f = 1 + 2*x + 3*y + x^2 + x*y + y^2
println("Polynomial: f(x,y,z) = 1 + 2x + 3y + x² + xy + y²")

println("\n--- Direct Evaluation ---")
x_value, y_value, z_value = 0.5, 0.3, 0.0
value = f(x_value, y_value, z_value)
expected = 1 + 2*x_value + 3*y_value + x_value^2 + x_value*y_value + y_value^2
println("f(0.5, 0.3, 0.0) = ", value)
println("Expected: ", expected)
println("Match: ", isapprox(value, expected))

println("\n--- Safe Coefficient Access ---")
println("Constant term: ", cst(f))
println("∂f/∂x|₀: ", element(f, [1, 0, 0]))
println("∂f/∂y|₀: ", element(f, [0, 1, 0]))
println("∂f/∂z|₀: ", element(f, [0, 0, 1]))
println("Coefficient of xy: ", element(f, [1, 1, 0]))

# Substitute x = 0.5 while leaving y and z as polynomial variables.
println("\n--- Partial Substitution with compose ---")
x_half = CTPS(0.5, desc)
g = compose(f, [x_half, y, z])
println("f(0.5, y, z) has constant term: ", cst(g))
println("f(0.5, y, z) has y coefficient: ", element(g, [0, 1, 0]))
println("Check at y=0.3, z=0: ", g(0.0, 0.3, 0.0))

@assert value ≈ expected
@assert cst(g) ≈ 2.25
@assert element(g, [0, 1, 0]) ≈ 3.5
@assert g(0.0, 0.3, 0.0) ≈ expected

println("\n✓ Value evaluation demonstration completed!")

# Basic TPSA Operations: Addition and Multiplication
# This example demonstrates creating TPSA objects and performing basic arithmetic

using PolySeries

# Set up this task's default descriptor for new TPSA objects
set_descriptor!(3, 4)  # 3 variables, maximum order 4

println("=== Basic TPSA Operations ===\n")

# Create TPSA variables - much simpler now!
x = CTPS(0.0, 1)  # variable x
y = CTPS(0.0, 2)  # variable y
z = CTPS(0.0, 3)  # variable z

# Create a constant
c = CTPS(5.0)     # constant value 5.0

println("Created variables:")
println("  x: linear coefficient = ", element(x, [1, 0, 0]))
println("  y: linear coefficient = ", element(y, [0, 1, 0]))
println("  z: linear coefficient = ", element(z, [0, 0, 1]))
println("  c: constant value = ", cst(c))
println()

# Addition examples
println("--- Addition ---")
sum1 = x + y
println("x + y:")
println("  Constant: ", cst(sum1))
println("  x coeff:  ", element(sum1, [1, 0, 0]))
println("  y coeff:  ", element(sum1, [0, 1, 0]))
println()

sum2 = c + x + 2*y + 3*z
println("5 + x + 2y + 3z:")
println("  Constant: ", cst(sum2))
println("  x coeff:  ", element(sum2, [1, 0, 0]))
println("  y coeff:  ", element(sum2, [0, 1, 0]))
println("  z coeff:  ", element(sum2, [0, 0, 1]))
println()

# Multiplication examples
println("--- Multiplication ---")
prod1 = x * y
println("x * y:")
println("  Constant: ", cst(prod1))
println("  xy coeff: ", element(prod1, [1, 1, 0]))
println()

# More complex multiplication
prod2 = (1 + x) * (1 + y)
println("(1 + x) * (1 + y) = 1 + x + y + xy:")
println("  Constant: ", cst(prod2), " (expected 1)")
println("  x coeff:  ", element(prod2, [1, 0, 0]), " (expected 1)")
println("  y coeff:  ", element(prod2, [0, 1, 0]), " (expected 1)")
println("  xy coeff: ", element(prod2, [1, 1, 0]), " (expected 1)")
println()

# Polynomial expansion
println("--- Polynomial Expansion ---")
poly = (1 + x + y)^2
println("(1 + x + y)^2 = 1 + 2x + 2y + x^2 + 2xy + y^2:")
println("  Constant: ", cst(poly))
println("  x coeff:  ", element(poly, [1, 0, 0]))
println("  y coeff:  ", element(poly, [0, 1, 0]))
println("  x² coeff: ", element(poly, [2, 0, 0]))
println("  xy coeff: ", element(poly, [1, 1, 0]))
println("  y² coeff: ", element(poly, [0, 2, 0]))

println("\n✓ Basic operations completed successfully!")

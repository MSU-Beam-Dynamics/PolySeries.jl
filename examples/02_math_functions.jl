# Mathematical Functions with TPSA
# Demonstrates using mathematical functions (sin, cos, exp, log, etc.) with TPSA

using PolySeries

println("=== Mathematical Functions ===\n")

# Set up this task's default descriptor
set_descriptor!(2, 4)

# Create variables
x = CTPS(0.0, 1)
y = CTPS(0.0, 2)

# Plug in a numerical value for testing
# For example, let's evaluate at x = 0.1, y = 0.2
x_val = 0.1
y_val = 0.2

println("--- Exponential and Logarithm ---")
# exp(x)
exp_x = PolySeries.exp(x)
println("exp(x) at x=0:")
println("  Constant term: ", cst(exp_x), " (expected 1.0)")
println("  Linear term:   ", element(exp_x, [1, 0]), " (expected 1.0)")

# log(1 + x)
log_expr = PolySeries.log(1 + x)
println("\nlog(1 + x) at x=0:")
println("  Constant term: ", cst(log_expr), " (expected 0.0)")
println("  Linear term:   ", element(log_expr, [1, 0]), " (expected 1.0)")
println()

println("--- Trigonometric Functions ---")
# sin(x)
sin_x = PolySeries.sin(x)
println("sin(x) at x=0:")
println("  Constant term: ", cst(sin_x), " (expected 0.0)")
println("  Linear term:   ", element(sin_x, [1, 0]), " (expected 1.0)")

# cos(x)
cos_x = PolySeries.cos(x)
println("\ncos(x) at x=0:")
println("  Constant term: ", cst(cos_x), " (expected 1.0)")
println("  Linear term:   ", element(cos_x, [1, 0]), " (expected 0.0)")
println()

println("--- Hyperbolic Functions ---")
# sinh(x)
sinh_x = PolySeries.sinh(x)
println("sinh(x) at x=0:")
println("  Constant term: ", cst(sinh_x), " (expected 0.0)")
println("  Linear term:   ", element(sinh_x, [1, 0]), " (expected 1.0)")

# cosh(x)
cosh_x = PolySeries.cosh(x)
println("\ncosh(x) at x=0:")
println("  Constant term: ", cst(cosh_x), " (expected 1.0)")
println("  Linear term:   ", element(cosh_x, [1, 0]), " (expected 0.0)")
println()

println("--- Power Functions ---")
# Square
x_squared = PolySeries.pow(x, 2)
println("x^2:")
println("  Constant term: ", cst(x_squared))
println("  Linear term:   ", element(x_squared, [1, 0]))

# Square root of (1 + x)
sqrt_expr = PolySeries.sqrt(1 + x)
println("\nsqrt(1 + x) at x=0:")
println("  Constant term: ", cst(sqrt_expr), " (expected 1.0)")
println("  Linear term:   ", element(sqrt_expr, [1, 0]), " (expected 0.5)")
println()

println("--- Combined Operations ---")
# Complex expression: exp(x) * sin(y)
result = PolySeries.exp(x) * PolySeries.sin(y)
println("exp(x) * sin(y) at x=0, y=0:")
println("  Constant term: ", cst(result), " (expected 0.0)")
println("  x coefficient: ", element(result, [1, 0]), " (expected 0.0)")
println("  y coefficient: ", element(result, [0, 1]), " (expected 1.0)")

println("\n✓ Mathematical functions completed successfully!")

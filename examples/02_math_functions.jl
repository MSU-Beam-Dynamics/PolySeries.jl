# Mathematical Functions with TPSA
# Demonstrates using mathematical functions (sin, cos, exp, log, etc.) with TPSA

using TPSA

println("=== Mathematical Functions ===\n")

# Set up global descriptor
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
exp_x = TPSA.exp(x)
println("exp(x) at x=0:")
println("  Constant term: ", exp_x.c[1], " (expected 1.0)")
println("  Linear term:   ", exp_x.c[2], " (expected 1.0)")

# log(1 + x)
log_expr = TPSA.log(1 + x)
println("\nlog(1 + x) at x=0:")
println("  Constant term: ", log_expr.c[1], " (expected 0.0)")
println("  Linear term:   ", log_expr.c[2], " (expected 1.0)")
println()

println("--- Trigonometric Functions ---")
# sin(x)
sin_x = TPSA.sin(x)
println("sin(x) at x=0:")
println("  Constant term: ", sin_x.c[1], " (expected 0.0)")
println("  Linear term:   ", sin_x.c[2], " (expected 1.0)")

# cos(x)
cos_x = TPSA.cos(x)
println("\ncos(x) at x=0:")
println("  Constant term: ", cos_x.c[1], " (expected 1.0)")
println("  Linear term:   ", cos_x.c[2], " (expected 0.0)")
println()

println("--- Hyperbolic Functions ---")
# sinh(x)
sinh_x = TPSA.sinh(x)
println("sinh(x) at x=0:")
println("  Constant term: ", sinh_x.c[1], " (expected 0.0)")
println("  Linear term:   ", sinh_x.c[2], " (expected 1.0)")

# cosh(x)
cosh_x = TPSA.cosh(x)
println("\ncosh(x) at x=0:")
println("  Constant term: ", cosh_x.c[1], " (expected 1.0)")
println("  Linear term:   ", cosh_x.c[2], " (expected 0.0)")
println()

println("--- Power Functions ---")
# Square
x_squared = TPSA.pow(x, 2)
println("x^2:")
println("  Constant term: ", x_squared.c[1])
println("  Linear term:   ", x_squared.c[2])

# Square root of (1 + x)
sqrt_expr = TPSA.sqrt(1 + x)
println("\nsqrt(1 + x) at x=0:")
println("  Constant term: ", sqrt_expr.c[1], " (expected 1.0)")
println("  Linear term:   ", sqrt_expr.c[2], " (expected 0.5)")
println()

println("--- Combined Operations ---")
# Complex expression: exp(x) * sin(y)
result = TPSA.exp(x) * TPSA.sin(y)
println("exp(x) * sin(y) at x=0, y=0:")
println("  Constant term: ", result.c[1], " (expected 0.0)")
println("  x coefficient: ", result.c[2], " (expected 0.0)")
println("  y coefficient: ", result.c[3], " (expected 1.0)")

println("\n✓ Mathematical functions completed successfully!")

# Basic TPSA Operations: Addition and Multiplication
# This example demonstrates creating TPSA objects and performing basic arithmetic

using TPSA

# Set up the global descriptor once - all TPSA objects will use this
set_descriptor!(3, 4)  # 3 variables, maximum order 4

println("=== Basic TPSA Operations ===\n")

# Create TPSA variables - much simpler now!
x = CTPS(0.0, 1)  # variable x
y = CTPS(0.0, 2)  # variable y
z = CTPS(0.0, 3)  # variable z

# Create a constant
c = CTPS(5.0)     # constant value 5.0

println("Created variables:")
println("  x: linear coefficient = ", x.c[2])
println("  y: linear coefficient = ", y.c[3])
println("  z: linear coefficient = ", z.c[4])
println("  c: constant value = ", c.c[1])
println()

# Addition examples
println("--- Addition ---")
sum1 = x + y
println("x + y:")
println("  Constant: ", sum1.c[1])
println("  x coeff:  ", sum1.c[2])
println("  y coeff:  ", sum1.c[3])
println()

sum2 = c + x + 2*y + 3*z
println("5 + x + 2y + 3z:")
println("  Constant: ", sum2.c[1])
println("  x coeff:  ", sum2.c[2])
println("  y coeff:  ", sum2.c[3])
println("  z coeff:  ", sum2.c[4])
println()

# Multiplication examples
println("--- Multiplication ---")
prod1 = x * y
println("x * y:")
println("  Constant: ", prod1.c[1])

# Find the xy term (degree 2, exponents [0, 1, 1, 0])
desc = prod1.desc
for i in 1:desc.N
    exp_vec = TPSA.getindexmap(desc.polymap, i)
    if exp_vec[1] == 2 && exp_vec[2] == 1 && exp_vec[3] == 1  # xy term
        println("  xy coeff: ", prod1.c[i])
    end
end
println()

# More complex multiplication
prod2 = (1 + x) * (1 + y)
println("(1 + x) * (1 + y) = 1 + x + y + xy:")
println("  Constant: ", prod2.c[1], " (expected 1)")
println("  x coeff:  ", prod2.c[2], " (expected 1)")
println("  y coeff:  ", prod2.c[3], " (expected 1)")
for i in 1:desc.N
    exp_vec = TPSA.getindexmap(desc.polymap, i)
    if exp_vec[1] == 2 && exp_vec[2] == 1 && exp_vec[3] == 1  # xy term
        println("  xy coeff: ", prod2.c[i], " (expected 1)")
    end
end
println()

# Polynomial expansion
println("--- Polynomial Expansion ---")
poly = (1 + x + y)^2
println("(1 + x + y)^2 = 1 + 2x + 2y + x^2 + 2xy + y^2:")
println("  Constant: ", poly.c[1])
println("  x coeff:  ", poly.c[2])
println("  y coeff:  ", poly.c[3])

for i in 1:desc.N
    exp_vec = TPSA.getindexmap(desc.polymap, i)
    if exp_vec[1] == 2
        if exp_vec[2] == 2  # x^2
            println("  x² coeff: ", poly.c[i])
        elseif exp_vec[2] == 1 && exp_vec[3] == 1  # xy
            println("  xy coeff: ", poly.c[i])
        elseif exp_vec[3] == 2  # y^2
            println("  y² coeff: ", poly.c[i])
        end
    end
end

println("\n✓ Basic operations completed successfully!")

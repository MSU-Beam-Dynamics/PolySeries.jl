# Differentiation, integration, and composition from Taylor coefficients

using PolySeries
using Printf

println("=== Derivatives, Integration, and Composition ===\n")

desc = PSDesc(2, 4)
x = CTPS(0.0, 1, desc)
y = CTPS(0.0, 2, desc)
variables = [x, y]

function monomial(variables, exponents)
    result = CTPS(1.0, variables[1].desc)
    for (variable, exponent) in zip(variables, exponents)
        exponent == 0 || (result = result * variable^exponent)
    end
    return result
end

function differentiate(p, variable_index, variables)
    result = CTPS(0.0, p.desc)
    for index in 1:p.desc.N
        row = PolySeries.getindexmap(p.desc.polymap, index)
        exponents = Int.(row[2:end])
        power = exponents[variable_index]
        power == 0 && continue
        coefficient = element(p, exponents)
        iszero(coefficient) && continue
        exponents[variable_index] -= 1
        result = result + coefficient * power * monomial(variables, exponents)
    end
    return result
end

function integrate(p, variable_index, variables)
    result = CTPS(0.0, p.desc)
    for index in 1:p.desc.N
        row = PolySeries.getindexmap(p.desc.polymap, index)
        row[1] == p.desc.order && continue
        exponents = Int.(row[2:end])
        coefficient = element(p, exponents)
        iszero(coefficient) && continue
        exponents[variable_index] += 1
        result = result + coefficient / exponents[variable_index] * monomial(variables, exponents)
    end
    return result
end

f = x^3 + 2*x^2*y + x*y^2 + y^3
df_dx = differentiate(f, 1, variables)
df_dy = differentiate(f, 2, variables)

println("f(x,y) = x³ + 2x²y + xy² + y³")
println("∂f/∂x = 3x² + 4xy + y²")
@printf("  x²: %.1f, xy: %.1f, y²: %.1f\n",
        element(df_dx, [2, 0]), element(df_dx, [1, 1]), element(df_dx, [0, 2]))
println("∂f/∂y = 2x² + 2xy + 3y²")
@printf("  x²: %.1f, xy: %.1f, y²: %.1f\n",
        element(df_dy, [2, 0]), element(df_dy, [1, 1]), element(df_dy, [0, 2]))

integral = integrate(x^2, 1, variables)
println("\n∫x² dx = ", element(integral, [3, 0]), "x³")

h = x^2
composed = compose(h, [1 + x, y])
println("\nh(u) = u²; h(1+x) = (1+x)²")
println("  constant: ", cst(composed))
println("  x: ", element(composed, [1, 0]))
println("  x²: ", element(composed, [2, 0]))

@assert element(df_dx, [1, 1]) == 4.0
@assert element(df_dy, [2, 0]) == 2.0
@assert element(integral, [3, 0]) ≈ 1/3
@assert (cst(composed), element(composed, [1, 0]), element(composed, [2, 0])) == (1.0, 2.0, 1.0)

println("\n✓ Calculus operations demonstration completed!")

# Index mapping and safe coefficient access

using PolySeries
using Printf

println("=== Index Mapping and Coefficient Access ===\n")

desc = PSDesc(3, 3)
x = CTPS(0.0, 1, desc)
y = CTPS(0.0, 2, desc)

exponents_at(desc, index) = Int.(PolySeries.getindexmap(desc.polymap, index)[2:end])

println("TPSA Configuration:")
println("  Number of variables: ", desc.nv)
println("  Maximum order: ", desc.order)
println("  Total coefficients: ", desc.N)

println("\n--- Index-to-Monomial Mapping ---")
println("Index | Degree | x^i y^j z^k | Exponents")
println("------|--------|-------------|----------")
for index in 1:min(15, desc.N)
    row = PolySeries.getindexmap(desc.polymap, index)
    exponents = Int.(row[2:end])
    monomial = "x^$(exponents[1]) y^$(exponents[2]) z^$(exponents[3])"
    @printf("%5d | %6d | %11s | %s\n", index, row[1], monomial, exponents)
end

poly = (1 + x)^3 * (1 + y)^2
println("\n--- Nonzero Coefficients of (1+x)³(1+y)² ---")
for index in 1:desc.N
    exponents = exponents_at(desc, index)
    coefficient = element(poly, exponents)
    iszero(coefficient) && continue
    monomial = "x^$(exponents[1]) y^$(exponents[2]) z^$(exponents[3])"
    @printf("  Index %3d: %s = %.6f\n", index, monomial, coefficient)
end

target = [2, 1, 0]
target_index = findindex(poly, target)
println("\nMonomial x²y¹z⁰ is at index: ", target_index)
println("Coefficient value: ", element(poly, target))

println("\n--- Terms Grouped by Degree ---")
for degree in 0:desc.order
    println("\nDegree $degree:")
    count = 0
    for index in 1:desc.N
        row = PolySeries.getindexmap(desc.polymap, index)
        row[1] == degree || continue
        exponents = Int.(row[2:end])
        coefficient = element(poly, exponents)
        iszero(coefficient) && continue
        count += 1
        monomial = "x^$(exponents[1]) y^$(exponents[2]) z^$(exponents[3])"
        @printf("  %s: %.6f\n", monomial, coefficient)
    end
    count == 0 && println("  (no nonzero terms)")
end

@assert element(poly, [2, 1, 0]) == 6.0
println("\n✓ Index mapping demonstration completed!")

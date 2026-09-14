# julia --project=benchmarks examples/08_parameter_gradients.jl
# Requires Enzyme in the active environment.
using PolySeries, Enzyme

desc = PSDesc(1, 6)

function selected_coefficient_loss(parameters, desc)
    a = sum(parameters) / length(parameters)
    b = sum(abs2, parameters) / length(parameters)
    x = CTPS(0.0, 1, desc)
    series = exp(a*x) * (1 + b*x^2)
    # One reverse pass for a weighted combination of selected coefficients.
    return element(series, [1]) + 2element(series, [3])
end

parameters = collect(range(0.0, 0.1; length=1000))
loss = p -> selected_coefficient_loss(p, desc)
gradient = Enzyme.gradient(Reverse, loss, parameters)[1]

# The selected coefficients are a and a³/6 + a*b.
n = length(parameters)
a, b = sum(parameters)/n, sum(abs2, parameters)/n
expected = [(1 + a^2 + 2b + 4a*p)/n for p in parameters]
@assert isapprox(gradient, expected; rtol=1e-12)
println("Gradient of selected coefficients: ", length(gradient), " parameters")
println("Maximum absolute error: ", maximum(abs.(gradient .- expected)))

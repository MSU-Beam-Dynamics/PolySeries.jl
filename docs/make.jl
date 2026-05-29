using Documenter
using PolySeries

makedocs(
    sitename = "PolySeries.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.MathJax3(),
    ),
    modules = [PolySeries],
    warnonly = [:missing_docs],
    pages = [
        "Home"          => "index.md",
        "Tutorial"      => "tutorial.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/MSU-Beam-Dynamics/PolySeries.jl.git",
    devbranch = "main",
)

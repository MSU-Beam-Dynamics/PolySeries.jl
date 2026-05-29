using Documenter
using PolySeries

makedocs(
    sitename = "PolySeries.jl",
    format = Documenter.HTML(
        prettyurls = true,
        mathengine = Documenter.MathJax3(),
        edit_link = "main",
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
    branch = "gh-pages",
    devbranch = "main",
)

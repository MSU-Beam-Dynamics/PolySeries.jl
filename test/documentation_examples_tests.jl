using Test

# Each example runs in-process inside its own anonymous module, with output
# captured to a temporary file that is printed only on failure. This avoids a
# fresh Julia process (and a fresh compile of PolySeries and Enzyme) per script.
# The scripts are plain top-level code that only uses PolySeries and standard
# libraries, so module isolation is sufficient.
function run_example_inprocess(path::AbstractString; label=basename(path))
    sandbox = Module(Symbol("Example_", replace(label, r"[^A-Za-z0-9]" => "_")))
    log = tempname()
    try
        redirect_stdio(stdout=log, stderr=log) do
            Base.include(sandbox, path)
        end
        return true
    catch err
        println("--- example $label failed: ", sprint(showerror, err))
        isfile(log) && print(read(log, String))
        return false
    finally
        isfile(log) && rm(log; force=true)
    end
end

@testset "README executable blocks" begin
    package_root = normpath(joinpath(@__DIR__, ".."))
    readme = read(joinpath(package_root, "README.md"), String)
    pattern = r"<!-- readme-test -->\s*```julia\r?\n(.*?)\r?\n```"s
    blocks = [match.captures[1] for match in eachmatch(pattern, readme)]

    @test length(blocks) == 2
    # Every Julia block except the installation command must opt into testing.
    julia_blocks = [m.captures[1] for m in eachmatch(r"```julia\r?\n(.*?)\r?\n```"s, readme)]
    @test length(julia_blocks) == length(blocks) + 1
    @test count(code -> occursin("Pkg.add(", code), julia_blocks) == 1
    mktempdir() do sandbox
        for (index, code) in enumerate(blocks)
            @testset "block $index" begin
                script = joinpath(sandbox, "readme_example_$index.jl")
                write(script, code)
                @test run_example_inprocess(script; label="README block $index")
            end
        end
    end
end

@testset "standalone example scripts" begin
    package_root = normpath(joinpath(@__DIR__, ".."))
    examples_dir = joinpath(package_root, "examples")
    scripts = sort(filter(path -> endswith(path, ".jl"), readdir(examples_dir; join=true)))

    # Enzyme is a test extra and is available under Pkg.test. Keep direct
    # include("test/runtests.jl") useful in a source-only environment too.
    if Base.find_package("Enzyme") === nothing
        filter!(path -> basename(path) != "07_enzyme_ad.jl", scripts)
    end

    for script in scripts
        @testset "$(basename(script))" begin
            @test run_example_inprocess(script)
        end
    end
end

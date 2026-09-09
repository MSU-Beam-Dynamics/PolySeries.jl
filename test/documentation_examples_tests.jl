using Test

function run_example_script(path::AbstractString, project::AbstractString)
    output = IOBuffer()
    # `--compiled-modules=existing` (Julia ≥ 1.11) avoids recompiling the package
    # in every child process; older releases simply skip the flag.
    flags = VERSION >= v"1.11" ? ["--compiled-modules=existing"] : String[]
    command = `$(Base.julia_cmd()) --startup-file=no $flags --project=$project $path`
    process = run(pipeline(ignorestatus(command), stdout=output, stderr=output))
    return success(process), String(take!(output))
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
                # Pkg.test supplies a resolved temporary environment even in
                # a clean checkout without a root Manifest.toml.
                passed, output = run_example_script(script, dirname(Base.active_project()))
                passed || println(output)
                @test passed
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

    project = dirname(Base.active_project())
    for script in scripts
        @testset "$(basename(script))" begin
            passed, output = run_example_script(script, project)
            passed || println(output)
            @test passed
        end
    end
end

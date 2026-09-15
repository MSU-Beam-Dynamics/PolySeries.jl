# Compare the working tree with an isolated, read-only snapshot of a commit.
# julia --compiled-modules=existing --project=benchmarks benchmarks/benchmark_multiplication.jl
# Optional: POLYSERIES_BENCH_BASE=<commit> (default is the commit before dedicated squaring).
using PolySeries, Enzyme, BenchmarkTools, Dates, Printf, SHA

const BASE_COMMIT = get(ENV,"POLYSERIES_BENCH_BASE","e2259a0")
const BASE_DIRECTORY = mktempdir()
for name in split(readchomp(`git ls-tree -r --name-only $BASE_COMMIT src ext`),'\n')
    path = joinpath(BASE_DIRECTORY,name)
    mkpath(dirname(path))
    contents = read(`git show $BASE_COMMIT:$name`,String)
    name == "ext/PolySeriesEnzymeExt.jl" &&
        (contents = replace(contents,"using PolySeries"=>"using ..PolySeries"))
    name == "src/PolySeries.jl" &&
        (contents = replace(contents,"using EnzymeCore: within_autodiff"=>
            "using Main.PolySeries: within_autodiff"))
    write(path,contents)
end
module BeforeRebuild
    include(joinpath(Main.BASE_DIRECTORY,"src","PolySeries.jl"))
    include(joinpath(Main.BASE_DIRECTORY,"ext","PolySeriesEnzymeExt.jl"))
end

bench_gradient(label,fn,p) = measure(label,()->Enzyme.gradient(Reverse,fn,p))

function measure(label,f)
    f()
    duration = @elapsed f()
    evaluations = clamp(round(Int,1e-4/max(duration,1e-9)),1,1000)
    trial = @benchmark $f() evals=evaluations samples=250 seconds=0.3
    @printf("%s,%.3f,%.3f,%d,%d\n",label,minimum(trial).time,median(trial).time,trial.memory,trial.allocs)
    flush(stdout)
end

println("# $(Dates.now()), Julia $VERSION, $(Sys.CPU_NAME), threads=$(Threads.nthreads()), base=$BASE_COMMIT")
println("# Enzyme=$(pkgversion(Enzyme)), PolySeries=$(pkgversion(PolySeries))")
for path in ("src/ctps.jl","src/multiplication.jl")
    println("# sha256 $path $(bytes2hex(sha256(read(path))))")
end
println("label,min_ns,median_ns,julia_bytes,julia_allocations")

function run_multiplications(M,label,nv,order)
        d=M.PSDesc(nv,order)
        a,b,out=[M.CTPS(Float64,d) for _ in 1:3]
        a.c .= [sin(i)/10 for i in 1:d.N]
        b.c .= [cos(i)/10 for i in 1:d.N]
        M.update_degree_mask!(a);M.update_degree_mask!(b)
        fn=M.mul!
        measure("$label/independent/$nv/$order",()->fn(out,a,b))
        measure("$label/square/$nv/$order",()->fn(out,a,a))
        # Dense degree masks with many numerical zero coefficients.
        for i in 1:d.N
            i%11 != 0 && (a.c[i]=0)
            i%13 != 0 && (b.c[i]=0)
        end
        measure("$label/zero_heavy/$nv/$order",()->fn(out,a,b))
        measure("$label/zero_heavy_square/$nv/$order",()->fn(out,a,a))
end

function layout_square!(out,a,schedules,block)
    fill!(out,zero(eltype(out)))
    PolySeries._square_schedules!(out,a,schedules,block)
end
function run_layouts()
    for (nv,order) in ((4,6),(6,6),(6,8),(6,10),(6,12),(8,8))
        d=PSDesc(nv,order)
        a=sin.(1:d.N)./10
        out=similar(a)
        grouped=sort(d.mul;by=s -> (Int(s.di)+Int(s.dj),s.di))
        for (name,schedules) in (("original",d.mul),("output_degree",grouped)), block in (0,64,256)
            strategy=Val(block)
            measure("layout/$name/$block/$nv/$order",()->layout_square!(out,a,schedules,strategy))
        end
    end
end
function run_ad(M,label)
    fn=M._bench_square_loss
    for (nv,order) in ((4,6),(6,6),(6,8))
        d=M.PSDesc(nv,order)
        p=M.CTPS(Float64,d)
        p.c .= sin.(1:d.N)./10
        M.update_degree_mask!(p)
        bench_gradient("$label/square_reverse/$nv/$order",fn,p)
    end
end
for M in (BeforeRebuild.PolySeries,PolySeries)
    Core.eval(M,quote
        function _bench_square_loss(p)
            y=p*p
            ex=zeros(Int,p.desc.nv)
            ex[1]=p.desc.order
            cst(y)+element(y,ex)
        end
    end)
end
if "--layouts-only" in ARGS
    run_layouts()
else
    if !("--ad-only" in ARGS)
        # Pair the implementations by descriptor to reduce drift over the run.
        for (nv,order) in ((1,63),(2,12),(4,6),(6,6),(6,8),(6,10),(6,12),(8,8))
            for (M,label) in ((BeforeRebuild.PolySeries,"before"),(PolySeries,"after"))
                run_multiplications(M,label,nv,order)
            end
        end
    end
    if !("--ordinary-only" in ARGS)
        for (M,label) in ((BeforeRebuild.PolySeries,"before"),(PolySeries,"after"))
            run_ad(M,label)
        end
    end
end

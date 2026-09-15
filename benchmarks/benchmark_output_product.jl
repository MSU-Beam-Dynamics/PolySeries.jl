# Compare the working tree with an isolated, read-only snapshot of a commit.
# julia --compiled-modules=existing --project=benchmarks benchmarks/benchmark_output_product.jl
# Optional: POLYSERIES_BENCH_BASE=<commit> (default is the commit before output-coefficient products).
using PolySeries, Enzyme, BenchmarkTools, Dates, Printf, SHA, Statistics

const BASE_COMMIT = get(ENV,"POLYSERIES_BENCH_BASE","c11343a")
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
for path in ("src/ctps.jl","src/multiplication.jl","ext/PolySeriesEnzymeExt.jl")
    println("# sha256 $path $(bytes2hex(sha256(read(path))))")
end
println("label,min_ns,median_ns,julia_bytes,julia_allocations")

function ordinary(M,label,nv,n,T)
    d=M.PSDesc(nv,n)
    a,b,out=[M.CTPS(T,d) for _ in 1:3]
    a.c .= T.(sin.(1:d.N)./10);b.c .= T.(cos.(1:d.N)./10)
    M.update_degree_mask!(a);M.update_degree_mask!(b)
    fn=M.mul!
    measure("$label/dense/$T/$nv/$n",()->fn(out,a,b))
    for i in 1:d.N
        i%11!=0 && (a.c[i]=0)
        i%13!=0 && (b.c[i]=0)
    end
    measure("$label/zero_heavy/$T/$nv/$n",()->fn(out,a,b))
end
function construction(M,label,nv,n)
    # Cache misses deliberately retain a handful of extra descriptors only.
    times=Float64[]; bytes=Int[]
    for repetition in 1:6
        delete!(M.DESC_CACHE,(nv,n))
        start=time_ns()
        allocated=@allocated M.PSDesc(nv,n)
        elapsed=time_ns()-start
        repetition==1 && continue
        push!(times,elapsed);push!(bytes,allocated)
    end
    @printf("%s/construction/%d/%d,%.3f,%.3f,%d,-1\n",label,nv,n,minimum(times),median(times),round(Int,median(bytes)))
end
for M in (BeforeRebuild.PolySeries,PolySeries)
    Core.eval(M,quote
        function _bench_output_loss(a,b)
            y=a*b
            ex=zeros(Int,a.desc.nv);ex[1]=a.desc.order
            2cst(y)+element(y,ex)
        end
    end)
end
function ad_case(M,label,nv,n)
    d=M.PSDesc(nv,n)
    a,b=[M.CTPS(Float64,d) for _ in 1:2]
    a.c .= sin.(1:d.N)./10;b.c .= cos.(1:d.N)./10
    M.update_degree_mask!(a);M.update_degree_mask!(b)
    fn=M._bench_output_loss
    measure("$label/reverse/$nv/$n",()->Enzyme.gradient(Reverse,fn,a,b))
end
if !("--ad-only" in ARGS)
    for (nv,n) in ((1,6),(1,8),(1,20),(1,63),(2,6),(2,8),(2,12),(2,20),(2,40),(3,6),(6,8)), T in (Float32,Float64)
        for (M,label) in ((BeforeRebuild.PolySeries,"before"),(PolySeries,"after"))
            ordinary(M,label,nv,n,T)
        end
    end
    for (nv,n) in ((1,63),(2,12),(2,20))
        for (M,label) in ((BeforeRebuild.PolySeries,"before"),(PolySeries,"after"))
            construction(M,label,nv,n)
        end
        d=PSDesc(nv,n);p=d.output_product
        storage=sizeof(p.offsets)+sizeof(p.left)+sizeof(p.right)+sizeof(p.diagonal)
        println("# extra_plan_storage/$nv/$n=$storage bytes (array payloads)")
        measure("after/plan_build/$nv/$n",()->PolySeries.build_output_product_plan(d.N,d.mul,UInt16))
    end
end
if !("--ordinary-only" in ARGS)
    for (nv,n) in ((1,20),(1,63),(2,12),(2,20),(6,8))
        for (M,label) in ((BeforeRebuild.PolySeries,"before"),(PolySeries,"after"))
            ad_case(M,label,nv,n)
        end
    end
end

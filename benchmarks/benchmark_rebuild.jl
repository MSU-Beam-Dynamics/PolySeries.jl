# Compare the working tree with an isolated, read-only snapshot of a commit.
# julia --compiled-modules=existing --project=benchmarks benchmarks/benchmark_rebuild.jl
# Optional: POLYSERIES_BENCH_BASE=<commit> (default is the rebuild's starting point).
using PolySeries, Enzyme, BenchmarkTools, Dates, Printf, SHA

const BASE_COMMIT = get(ENV,"POLYSERIES_BENCH_BASE","9c2d667")
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

bench_compose(label,fn,out,f,g,ws) = measure(label,()->fn(out,f,g,ws))
bench_gradient(label,fn,p) = measure(label,()->Enzyme.gradient(Reverse,fn,p))

function measure(label,f)
    f()
    duration = @elapsed f()
    evaluations = clamp(round(Int,1e-4/max(duration,1e-9)),1,1000)
    trial = @benchmark $f() evals=evaluations samples=250 seconds=0.3
    @printf("%s,%.3f,%.3f,%d,%d\n",label,minimum(trial).time,median(trial).time,trial.memory,trial.allocs)
    flush(stdout)
end

# Identical functions are compiled against each implementation. No package
# mutation, checkout, installation, or baseline manifest is required.
for M in (BeforeRebuild.PolySeries,PolySeries)
    Core.eval(M,quote
        function _bench_parameter_loss(p,d)
            a,b = sum(p)/length(p),sum(abs2,p)/length(p)
            x = CTPS(0.0,1,d)
            y = exp(a*x)*(1+b*x^2)
            return element(y,[1])+2element(y,[3])
        end
        function _bench_product_loss(p)
            y = p*p
            ex = zeros(Int,p.desc.nv)
            ex[1] = p.desc.order
            return cst(y)+element(y,ex)
        end
        function _bench_exp_loss(p)
            y = exp(p)
            ex = zeros(Int,p.desc.nv)
            ex[1] = p.desc.order
            return cst(y)+element(y,ex)
        end
    end)
end

function ordinary_cases(M,label)
    for (nv,order) in ((2,12),(4,6),(6,6),(6,8))
        d = M.PSDesc(nv,order)
        x = [M.CTPS(0.0,i,d) for i in 1:nv]
        dense = (1+0.1sum(x))^order
        affine = 0.1+x[1]
        quadratic = 0.1+x[1]+0.02x[2]^2
        sparse1,sparse2 = 1+x[1],1+x[2]
        out,ws = M.CTPS(Float64,d),M.CompositionWorkspace(d)
        mulfn,compfn,expfn = M.mul!,M.compose!,M.exp!
        measure("$label/mul_dense/$nv/$order",()->mulfn(out,dense,dense))
        measure("$label/mul_sparse/$nv/$order",()->mulfn(out,sparse1,sparse2))
        measure("$label/mul_affine/$nv/$order",()->mulfn(out,dense,affine))
        measure("$label/mul_quadratic/$nv/$order",()->mulfn(out,dense,quadratic))
        measure("$label/exp_dense/$nv/$order",()->expfn(out,dense))
        for kind in (:shift,:nonlinear)
            g = kind == :shift ? [0.1+v for v in x] :
                [0.1+x[i]+0.02x[mod1(i+1,nv)]^2 for i in 1:nv]
            bench_compose("$label/compose_$kind/$nv/$order",compfn,out,dense,g,ws)
        end
        g = [0.1+x[i]+0.02x[mod1(i+1,nv)]^2 for i in 1:nv]
        f = x[1]^order
        bench_compose("$label/compose_sparse/$nv/$order",compfn,out,f,g,ws)
        if M === PolySeries
            plan = CompositionPlan(f)
            bench_compose("$label/compose_planned/$nv/$order",compfn,out,plan,g,ws)
        end
    end
end

function ad_cases(M,label)
    parameter_desc = M.PSDesc(1,6)
    parameter_fn = M._bench_parameter_loss
    loss = p -> parameter_fn(p,parameter_desc)
    for n in (100,1000)
        p = collect(range(0.0,0.1;length=n))
        gradient = Enzyme.gradient(Reverse,Const(loss),p)[1]
        a,b = sum(p)/n,sum(abs2,p)/n
        @assert gradient ≈ [(1+a^2+2b+4a*pi)/n for pi in p]
        bench_gradient("$label/parameter_reverse/$n",Const(loss),p)
    end
    for (nv,order) in ((4,6),(6,6))
        d = M.PSDesc(nv,order)
        p = M.CTPS(Float64,d)
        fill!(p.c,0.1)
        M.update_degree_mask!(p)
        product_fn = M._bench_product_loss
        bench_gradient("$label/product_reverse/$nv/$order",product_fn,p)
        exponential_fn = M._bench_exp_loss
        bench_gradient("$label/exp_reverse/$nv/$order",exponential_fn,p)
    end
end

println("# $(Dates.now()), Julia $VERSION, $(Sys.CPU_NAME), threads=$(Threads.nthreads()), base=$BASE_COMMIT")
println("# Enzyme=$(pkgversion(Enzyme)), PolySeries=$(pkgversion(PolySeries))")
for path in ("src/ctps.jl","src/composition.jl","ext/multiplication_rules.jl","ext/exponential_rules.jl")
    println("# sha256 $path $(bytes2hex(sha256(read(path))))")
end
println("label,min_ns,median_ns,julia_bytes,julia_allocations")
for (M,label) in ((BeforeRebuild.PolySeries,"before"),(PolySeries,"after"))
    "--ad-only" in ARGS || ordinary_cases(M,label)
    "--ordinary-only" in ARGS || ad_cases(M,label)
end

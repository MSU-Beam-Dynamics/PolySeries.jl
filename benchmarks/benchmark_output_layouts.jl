# Reproduce rejected and accepted output-coefficient reduction experiments.
# julia --project=benchmarks benchmarks/benchmark_output_layouts.jl
using PolySeries, BenchmarkTools, Printf, Dates, SHA

function output!(out,a,b,p,::Val{style}) where style
    @inbounds for k in eachindex(out)
        z=zero(eltype(out))
        if style==0
            for t in p.offsets[k]:p.offsets[k+1]-1
                i,j=p.left[t],p.right[t]
                z+=a[i]*b[j]+a[j]*b[i]
            end
        elseif style==1
            @simd for t in p.offsets[k]:p.offsets[k+1]-1
                i,j=p.left[t],p.right[t]
                z+=a[i]*b[j]+a[j]*b[i]
            end
        elseif style==2
            z2=zero(z)
            @simd for t in p.offsets[k]:p.offsets[k+1]-1
                i,j=p.left[t],p.right[t]
                z+=a[i]*b[j]
                z2+=a[j]*b[i]
            end
            z+=z2
        elseif style==3
            z0,z1,z2,z3 = zero(z),zero(z),zero(z),zero(z)
            start,stop=p.offsets[k],p.offsets[k+1]-1
            t=start
            while t <= stop-3
                i,j=p.left[t+0],p.right[t+0]
                z0+=a[i]*b[j]+a[j]*b[i]
                i,j=p.left[t+1],p.right[t+1]
                z1+=a[i]*b[j]+a[j]*b[i]
                i,j=p.left[t+2],p.right[t+2]
                z2+=a[i]*b[j]+a[j]*b[i]
                i,j=p.left[t+3],p.right[t+3]
                z3+=a[i]*b[j]+a[j]*b[i]
                t+=4
            end
            z=z0+z1+z2+z3
            for t in t:stop
                i,j=p.left[t],p.right[t]
                z+=a[i]*b[j]+a[j]*b[i]
            end
        elseif style==4
            z0,z1,z2,z3,z4,z5,z6,z7 = zero(z),zero(z),zero(z),zero(z),zero(z),zero(z),zero(z),zero(z)
            start,stop=p.offsets[k],p.offsets[k+1]-1
            t=start
            while t <= stop-7
                i,j=p.left[t+0],p.right[t+0]
                z0+=a[i]*b[j]+a[j]*b[i]
                i,j=p.left[t+1],p.right[t+1]
                z1+=a[i]*b[j]+a[j]*b[i]
                i,j=p.left[t+2],p.right[t+2]
                z2+=a[i]*b[j]+a[j]*b[i]
                i,j=p.left[t+3],p.right[t+3]
                z3+=a[i]*b[j]+a[j]*b[i]
                i,j=p.left[t+4],p.right[t+4]
                z4+=a[i]*b[j]+a[j]*b[i]
                i,j=p.left[t+5],p.right[t+5]
                z5+=a[i]*b[j]+a[j]*b[i]
                i,j=p.left[t+6],p.right[t+6]
                z6+=a[i]*b[j]+a[j]*b[i]
                i,j=p.left[t+7],p.right[t+7]
                z7+=a[i]*b[j]+a[j]*b[i]
                t+=8
            end
            z=z0+z1+z2+z3+z4+z5+z6+z7
            for t in t:stop
                i,j=p.left[t],p.right[t]
                z+=a[i]*b[j]+a[j]*b[i]
            end
        end
        q=p.diagonal[k]
        out[k]=q==0 ? z : z+a[q]*b[q]
    end
    out
end
function measure(label,f)
    f()
    trial=@benchmark $f() samples=150 seconds=.2
    @printf("%s,%.3f,%.3f,%d,%d\n",label,minimum(trial).time,median(trial).time,trial.memory,trial.allocs)
    flush(stdout)
end
function schedule_product!(out,a,b,d)
    fill!(out,zero(eltype(out)))
    mask=typemax(UInt64)>>(63-d.order)
    PolySeries._mul_schedules!(out,a,b,mask,mask,d.mul,Val(true))
end
function run_layouts()
    for (nv,n) in ((2,12),(4,6),(6,6),(6,8),(6,10),(6,12),(8,8))
        d=PSDesc(nv,n)
        a=sin.(1:d.N)./10;b=cos.(1:d.N)./10;out=similar(a)
        schedule_product!(out,a,b,d);reference=copy(out)
        measure("$nv/$n/schedule_kernel",()->schedule_product!(out,a,b,d))
        for I in (Int32,UInt16)
            p=PolySeries.build_output_product_plan(d.N,d.mul,I)
            storage=sizeof(p.offsets)+sizeof(p.left)+sizeof(p.right)+sizeof(p.diagonal)
            println("# extra_plan_storage/$nv/$n/$I=$storage bytes (array payloads)")
            for (name,style) in (("serial",0),("simd",1),("four_accumulators",3),("eight_accumulators",4))
                strategy=Val(style)
                output!(out,a,b,p,strategy)
                @assert out ≈ reference
                measure("$nv/$n/$I/$name",()->output!(out,a,b,p,strategy))
            end
        end
    end
end
println("# $(Dates.now()), Julia $VERSION, $(Sys.CPU_NAME), threads=$(Threads.nthreads())")
for path in ("src/multiplication.jl","benchmarks/benchmark_output_layouts.jl")
    println("# sha256 $path $(bytes2hex(sha256(read(path))))")
end
println("label,min_ns,median_ns,julia_bytes,julia_allocations")
run_layouts()

# Rules for the real dense coefficient convolution boundary. Numeric zeros
# never gate coefficient adjoints. Complex arithmetic keeps the existing
# differentiated implementation and Enzyme's complex conventions.
const RealCoefficientVector = Vector{T} where T<:Union{Float32,Float64}
const ProductForward = Union{Enzyme.Duplicated{<:RealCoefficientVector},
    Enzyme.DuplicatedNoNeed{<:RealCoefficientVector},
    Enzyme.BatchDuplicated{<:RealCoefficientVector},
    Enzyme.BatchDuplicatedNoNeed{<:RealCoefficientVector}}
const ProductReverse = Union{Enzyme.Duplicated{<:RealCoefficientVector},
    Enzyme.BatchDuplicated{<:RealCoefficientVector}}
const ProductConst = Enzyme.Const{<:RealCoefficientVector}

@inline function _product_shadow(config, arg, lane)
    arg isa Enzyme.Const && return nothing
    shadow = _shadow(arg, lane)
    return EnzymeRules.runtime_activity(config) && shadow === arg.val ? nothing : shadow
end

function EnzymeRules.forward(config::EnzymeRules.FwdConfig,
        func::Enzyme.Const{typeof(PolySeries._dense_product!)}, ::Type{<:Enzyme.Const},
        out::ProductForward, a::Union{ProductForward,ProductConst},
        b::Union{ProductForward,ProductConst}, desc::Enzyme.Const{PolySeries.PSDesc})
    d = desc.val
    mask = typemax(UInt64) >> (63-d.order)
    for lane in 1:EnzymeRules.width(config)
        dc = _product_shadow(config, out, lane)
        dc === nothing && continue
        da, db = _product_shadow(config,a,lane), _product_shadow(config,b,lane)
        fill!(dc, zero(eltype(dc)))
        da === nothing || PolySeries._mul_schedules!(dc, da, b.val, mask, mask, d.mul, Val(true))
        db === nothing || PolySeries._mul_schedules!(dc, a.val, db, mask, mask, d.mul, Val(true))
    end
    func.val(out.val, a.val, b.val, d)
    return nothing
end

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig,
        func::Enzyme.Const{typeof(PolySeries._dense_product!)}, ::Type{<:Enzyme.Const},
        out::ProductReverse, a::Union{ProductReverse,ProductConst},
        b::Union{ProductReverse,ProductConst}, desc::Enzyme.Const{PolySeries.PSDesc})
    # Only copy operands that the caller may overwrite before the reverse
    # sweep. Otherwise the tape retains references to the original arrays.
    overwritten = EnzymeRules.overwritten(config)
    # overwritten includes the function in slot 1 and the output in slot 2.
    av = overwritten[3] ? copy(a.val) : a.val
    bv = overwritten[4] ? copy(b.val) : b.val
    func.val(out.val, a.val, b.val, desc.val)
    return EnzymeRules.AugmentedReturn(nothing, nothing, (av,bv))
end

# Transpose of truncated convolution, expressed using exactly the descriptor's
# valid multiplication triples. Adjoint arrays accumulate, since callers can
# use an input more than once or supply the same input on both sides.
function _product_pullback!(da, db, dc, a, b, desc)
    @inbounds for sched in desc.mul
        ib, jb = Int(sched.i_start)-1, Int(sched.j_start)-1
        for i in 1:Int(sched.Ni), j in 1:Int(sched.Nj)
            seed = dc[sched.k_local[j,i]]
            da === nothing || (da[ib+i] += seed*b[jb+j])
            db === nothing || (db[jb+j] += seed*a[ib+i])
            if sched.di != sched.dj
                da === nothing || (da[jb+j] += seed*b[ib+i])
                db === nothing || (db[ib+i] += seed*a[jb+j])
            end
        end
    end
    fill!(dc, zero(eltype(dc)))
    return nothing
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfig,
        ::Enzyme.Const{typeof(PolySeries._dense_product!)}, ::Type{<:Enzyme.Const}, tape,
        out::ProductReverse, a::Union{ProductReverse,ProductConst},
        b::Union{ProductReverse,ProductConst}, desc::Enzyme.Const{PolySeries.PSDesc})
    av,bv = tape
    for lane in 1:EnzymeRules.width(config)
        dc = _product_shadow(config,out,lane)
        dc === nothing && continue
        _product_pullback!(_product_shadow(config,a,lane), _product_shadow(config,b,lane),
                           dc, av, bv, desc.val)
    end
    return (nothing,nothing,nothing,nothing)
end

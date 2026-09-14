# D exp(f)[df] = exp(f)*df, truncated to the descriptor's order. Reuse the
# convolution pullback; do not differentiate the graded recurrence's loops.
function EnzymeRules.forward(config::EnzymeRules.FwdConfig,
        func::Enzyme.Const{typeof(PolySeries._dense_exp!)}, ::Type{<:Enzyme.Const},
        out::ProductForward, f::Union{ProductForward,ProductConst},
        desc::Enzyme.Const{PolySeries.PSDesc})
    func.val(out.val,f.val,desc.val)
    for lane in 1:EnzymeRules.width(config)
        dy = _product_shadow(config,out,lane)
        dy === nothing && continue
        df = _product_shadow(config,f,lane)
        if df === nothing
            fill!(dy,zero(eltype(dy)))
        else
            PolySeries._dense_product!(dy,out.val,df,desc.val)
        end
    end
    return nothing
end

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig,
        func::Enzyme.Const{typeof(PolySeries._dense_exp!)}, ::Type{<:Enzyme.Const},
        out::ProductReverse, f::Union{ProductReverse,ProductConst},
        desc::Enzyme.Const{PolySeries.PSDesc})
    func.val(out.val,f.val,desc.val)
    # Slot 1 is the function; slot 2 is the output coefficient array.
    value = EnzymeRules.overwritten(config)[2] ? copy(out.val) : out.val
    return EnzymeRules.AugmentedReturn(nothing,nothing,value)
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfig,
        ::Enzyme.Const{typeof(PolySeries._dense_exp!)}, ::Type{<:Enzyme.Const}, value,
        out::ProductReverse, f::Union{ProductReverse,ProductConst},
        desc::Enzyme.Const{PolySeries.PSDesc})
    for lane in 1:EnzymeRules.width(config)
        dy = _product_shadow(config,out,lane)
        dy === nothing && continue
        df = _product_shadow(config,f,lane)
        _product_pullback!(df,nothing,dy,nothing,value,desc.val)
    end
    return (nothing,nothing,nothing)
end

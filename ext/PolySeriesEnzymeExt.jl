module PolySeriesEnzymeExt

using PolySeries
import Enzyme
import Enzyme.EnzymeRules

# Mark all non-differentiable TPSA-internal types as inactive so that Enzyme
# does not try to build shadow storage for them when differentiating through
# CTPS computations.
#
# PSDesc / PolyMap / MulSchedule2D / CompPlan are pure combinatorial index
# tables (compile-time constants after set_descriptor! is called).
# DescPool holds pre-allocated scratch buffers shared across calls; it is
# never part of the differentiable computation path.
#
# These rules are loaded automatically whenever both TPSA and Enzyme are
# present in the same session — no user action required.

EnzymeRules.inactive_type(::Type{<:PolySeries.PSDesc})      = true
EnzymeRules.inactive_type(::Type{<:PolySeries.DescPool})      = true
EnzymeRules.inactive_type(::Type{<:PolySeries.PolyMap})       = true
EnzymeRules.inactive_type(::Type{<:PolySeries.MulSchedule2D}) = true
EnzymeRules.inactive_type(::Type{<:PolySeries.CompPlan})      = true
EnzymeRules.inactive_type(::Type{<:PolySeries.DescriptorRegistry}) = true

# The task default is construction metadata, not a differentiable input. Keep
# Enzyme out of the task-local dictionary used to retrieve this descriptor.
EnzymeRules.inactive(::typeof(PolySeries.get_descriptor)) = nothing


# Enzyme treats integer masks as inactive metadata, not adjoints. Its CTPS
# shadows therefore use fully initialized dense coefficient storage and an
# independent full mask, so gradients at previously inactive degrees are visible.
function Enzyme.make_zero(::Type{CTPS{T}}, seen::IdDict, p::CTPS{T},
                          copy_inactive::Val=Val(false)) where T
    haskey(seen, p) && return seen[p]::CTPS{T}
    coeffs = Enzyme.make_zero(Vector{T}, seen, p.c, copy_inactive)
    if !isbitstype(T) && coeffs !== p.c
        for i in eachindex(coeffs)
            isassigned(coeffs, i) || (coeffs[i] = zero(T))
        end
    end
    mask = Ref(typemax(UInt64) >> (63 - p.desc.order))
    shadow = CTPS{T}(coeffs, p.desc, mask)
    seen[p] = shadow
    return shadow
end

# Put rules on the coefficient vector read, not the CTPS aggregate. Enzyme
# deliberately shares inactive integer metadata in aggregate shadows; using
# that metadata to gate tangent reads would discard valid coefficient tangents.
const ForwardCoefficients = Union{Enzyme.Duplicated{<:Vector}, Enzyme.DuplicatedNoNeed{<:Vector},
                                  Enzyme.BatchDuplicated{<:Vector}, Enzyme.BatchDuplicatedNoNeed{<:Vector}}
const ReverseCoefficients = Union{Enzyme.Duplicated{<:Vector}, Enzyme.BatchDuplicated{<:Vector}}
@inline _shadow(p, lane) = p.dval
@inline _shadow(p::Union{Enzyme.BatchDuplicated, Enzyme.BatchDuplicatedNoNeed}, lane) = p.dval[lane]

@inline function _coefficient_tangent(config, p, lane, index)
    shadow = _shadow(p, lane)
    if EnzymeRules.runtime_activity(config) && shadow === p.val
        return zero(eltype(p.val))
    end
    return shadow[index]
end

function EnzymeRules.forward(config::EnzymeRules.FwdConfig,
        func::Enzyme.Const{typeof(PolySeries._coefficient)},
        ::Type{<:Enzyme.Annotation}, p::ForwardCoefficients,
        mask::Enzyme.Const{UInt64}, index::Enzyme.Const{Int}, degree::Enzyme.Const{Int})
    primal = EnzymeRules.needs_primal(config) ? func.val(p.val, mask.val, index.val, degree.val) : nothing
    EnzymeRules.needs_shadow(config) || return primal
    shadow = if EnzymeRules.width(config) == 1
        _coefficient_tangent(config, p, 1, index.val)
    else
        ntuple(i -> _coefficient_tangent(config, p, i, index.val), EnzymeRules.width(config))
    end
    if EnzymeRules.needs_primal(config)
        return EnzymeRules.width(config) == 1 ? Enzyme.Duplicated(primal, shadow) :
                                               Enzyme.BatchDuplicated(primal, shadow)
    end
    return shadow
end

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig,
        func::Enzyme.Const{typeof(PolySeries._coefficient)},
        ::Type{<:Union{Enzyme.Active,Enzyme.Const}}, p::ReverseCoefficients,
        mask::Enzyme.Const{UInt64}, index::Enzyme.Const{Int}, degree::Enzyme.Const{Int})
    primal = EnzymeRules.needs_primal(config) ? func.val(p.val, mask.val, index.val, degree.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, index.val)
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfig,
        ::Enzyme.Const{typeof(PolySeries._coefficient)}, dret, index,
        p::ReverseCoefficients, ::Enzyme.Const{UInt64}, ::Enzyme.Const{Int}, ::Enzyme.Const{Int})
    dret isa Type{<:Enzyme.Const} && return (nothing, nothing, nothing, nothing)
    for i in 1:EnzymeRules.width(config)
        shadow = _shadow(p, i)
        if !(EnzymeRules.runtime_activity(config) && shadow === p.val)
            value = EnzymeRules.width(config) == 1 ? dret.val : dret[i].val
            shadow[index] += value
        end
    end
    return (nothing, nothing, nothing, nothing)
end

end # module

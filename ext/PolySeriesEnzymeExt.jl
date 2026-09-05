module PolySeriesEnzymeExt

using PolySeries
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

end # module

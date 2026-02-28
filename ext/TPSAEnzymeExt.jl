module TPSAEnzymeExt

using TPSA
import Enzyme.EnzymeRules

# Mark all non-differentiable TPSA-internal types as inactive so that Enzyme
# does not try to build shadow storage for them when differentiating through
# CTPS computations.
#
# TPSADesc / PolyMap / MulSchedule2D / CompPlan are pure combinatorial index
# tables (compile-time constants after set_descriptor! is called).
# DescPool holds pre-allocated scratch buffers shared across calls; it is
# never part of the differentiable computation path.
#
# These rules are loaded automatically whenever both TPSA and Enzyme are
# present in the same session — no user action required.

EnzymeRules.inactive_type(::Type{<:TPSA.TPSADesc})      = true
EnzymeRules.inactive_type(::Type{<:TPSA.DescPool})      = true
EnzymeRules.inactive_type(::Type{<:TPSA.PolyMap})       = true
EnzymeRules.inactive_type(::Type{<:TPSA.MulSchedule2D}) = true
EnzymeRules.inactive_type(::Type{<:TPSA.CompPlan})      = true

end # module

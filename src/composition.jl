# Execution strategies for ordinary composition. Enzyme uses the general
# differentiable path, so value-based strategy selection cannot lose tangents.
function _is_translation(g, desc::PSDesc)
    desc.order == 0 && return true
    for variable in 1:desc.nv
        p = g[variable]
        p.degree_mask[] & ~UInt64(3) == 0 || return false
        p.degree_mask[] & UInt64(2) != 0 || return false
        @inbounds for j in 1:desc.nv
            p.c[j+1] == (j == variable ? one(eltype(p.c)) : zero(eltype(p.c))) || return false
        end
    end
    return true
end

# Coordinate-wise triangular Taylor shift. Each pass adds the next coefficient
# to the current one; ascending degree keeps that next coefficient untouched
# within a pass. No factorials, powers, monomial images, or temporary vectors.
function _translate!(out::CTPS{T}, f::CTPS{T}, g) where T
    desc = f.desc
    copy!(out, f)
    fm = f.degree_mask[]
    fm == 0 && return out
    highest = 63 - leading_zeros(fm)
    highest == 0 && return out
    # Identity is an exact copy, including its degree holes.
    all(p -> iszero(cst(p)), g) && return out
    full = typemax(UInt64) >> (63 - highest)
    for (s,e) in active_ranges(desc, full & ~fm)
        @inbounds for i in s:e
            out.c[i] = zero(T)
        end
    end
    for variable in 1:desc.nv
        shift = cst(g[variable])
        iszero(shift) && continue
        for pass in highest-1:-1:0
            if pass == 0
                out.c[1] += shift * out.c[variable+1]
            end
            for degree in max(1,pass):highest-1
                sched = desc.mul[desc.mul_offsets[degree+1]+1]
                base = Int(sched.i_start)-1
                @inbounds for i in 1:Int(sched.Ni)
                    desc.polymap.map[base+i,variable+1] >= pass || continue
                    out.c[base+i] += shift * out.c[sched.k_local[variable,i]]
                end
            end
        end
    end
    out.degree_mask[] = full
    return out
end

"""
    CompositionPlan(f::CTPS)

Snapshot a fixed source polynomial and precompute its needed monomial traversal.
Use `compose(plan, g)` or `compose!(out, plan, g, workspace)` to repeatedly
evaluate it at changing substitution maps. Later changes to `f` do not change
the plan: construct a new plan when the source changes. Treat the plan's
internal arrays as read-only. Plans may be shared; workspaces must not be.

This separates reusable source planning from mutable scratch storage. During
Enzyme AD the general retained-image path preserves zero coefficient directions.
"""
struct CompositionPlan{T}
    source::CTPS{T}
    nodes::Vector{Int32}
    depths::Vector{UInt8}
end

function CompositionPlan(f::CTPS{T}) where T
    # Snapshot initialization must not read inactive (possibly poisoned) slots.
    source = CTPS(f)
    needed = falses(f.desc.N)
    for (s,e) in active_ranges(f.desc, f.degree_mask[]), i in s:e
        needed[i] = !iszero(f.c[i])
    end
    for i in f.desc.N:-1:2
        needed[i] && (needed[Int(f.desc.comp_plan.par_idx[i])] = true)
    end
    nodes = Int32[]
    depths = UInt8[]
    _plan_children!(nodes, depths, f.desc.comp_plan, needed, 1, 1)
    return CompositionPlan(source, nodes, depths)
end

function _plan_children!(nodes, depths, tree, needed, parent, depth)
    i = Int(tree.first_child[parent])
    while i != 0
        if needed[i]
            push!(nodes, Int32(i))
            push!(depths, UInt8(depth))
            _plan_children!(nodes, depths, tree, needed, i, depth+1)
        end
        i = Int(tree.next_sibling[i])
    end
    return nothing
end

"""
    compose!(out, plan::CompositionPlan, g, workspace)

Compose the source snapshot in `plan` using reusable scratch storage. No
source-support scan is needed in ordinary execution. The same descriptor and
nonaliasing requirements as `compose!(out, f, g, workspace)` apply.
"""
function compose!(out::CTPS{T}, plan::CompositionPlan{T}, g::AbstractVector{<:CTPS{T}},
                  ws::CompositionWorkspace{T}) where T
    f = plan.source
    within_autodiff() && return _compose_retained!(out, f, g)
    _check_composition(out, f, g)
    ws.desc === f.desc || throw(DimensionMismatch("Composition workspace descriptor must match inputs"))
    _is_translation(g, f.desc) && return _translate!(out, f, g)
    _zero_active!(out)
    root = ws.images[1]
    root.c[1] = one(T)
    root.degree_mask[] = UInt64(1)
    out.c[1] = cst(f)
    out.degree_mask[] = iszero(out.c[1]) ? UInt64(0) : UInt64(1)
    for position in eachindex(plan.nodes)
        i, depth = Int(plan.nodes[position]), Int(plan.depths[position])
        img = ws.images[depth+1]
        mul!(img, g[Int(f.desc.comp_plan.par_var[i])], ws.images[depth])
        degree = Int(f.desc.polymap.map[i,1])
        if (f.degree_mask[] >> degree) & UInt64(1) != 0
            coeff = f.c[i]
            iszero(coeff) || _add_scaled!(out, img, coeff)
        end
    end
    return out
end

function compose!(out::CTPS{T}, plan::CompositionPlan{T}, g::AbstractVector{<:CTPS{T}}) where T
    within_autodiff() && return _compose_retained!(out, plan.source, g)
    _check_composition(out, plan.source, g)
    _is_translation(g, plan.source.desc) && return _translate!(out, plan.source, g)
    return compose!(out, plan, g, CompositionWorkspace(plan.source.desc, T))
end

"""
    compose(plan::CompositionPlan, g)

Allocate the composition of a fixed source snapshot with `g`. For repeated
evaluation, reuse an output and `CompositionWorkspace` with `compose!`.
"""
function compose(plan::CompositionPlan{T}, g::AbstractVector{<:CTPS{T}}) where T
    out = _ctps_zero(T, plan.source.desc)
    return compose!(out, plan, g)
end

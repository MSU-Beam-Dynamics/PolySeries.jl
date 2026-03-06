
# Multiplication schedule - 2D k-map format
#
# Key insight: for deglex monomial ordering, every (i_local, j_local) pair in a
# degree-pair (di, dj) is valid (di + dj = dk ≤ order) and the j access for a
# fixed i is naturally sequential.  Therefore:
#   – jidx is redundant: j = j_start + j_local - 1  (implicit stride-1)
#   – kidx is replaced by k_local[j_local, i_local], a compact Int32 matrix
#     (Julia column-major → k_local[j_local, i_local] scans j sequentially)
#
# Memory savings vs old (jidx+kidx Int32 flat arrays):
#   old: 2 × Ni×Nj × 4 bytes   new: 1 × Ni×Nj × 4 bytes  (≈ 50% smaller)
#
# Cache improvements:
#   c2[j_start + j_local - 1]  — sequential read  (hardware-prefetchable)
#   k_local[j_local, i_local]  — sequential column read (column-major)
#   cr[k_start + k_local]       — bounded scatter into degree-dk slice
struct MulSchedule2D
    k_local::Matrix{Int32}  # k_local[j_local, i_local] = 1-based absolute index into c[]
    i_start::Int32           # global 1-based start of di block
    j_start::Int32           # global 1-based start of dj block
    k_start::Int32           # global 1-based start of dk block
    Ni::Int32                # Nd[di]
    Nj::Int32                # Nd[dj]
    di::UInt8                # degree of first operand  (for mask check)
    dj::UInt8                # degree of second operand (for mask check)
end

# Empty sentinel
MulSchedule2D() = MulSchedule2D(Matrix{Int32}(undef, 0, 0),
                                  Int32(0), Int32(0), Int32(0),
                                  Int32(0), Int32(0), 0x00, 0x00)

# Composition plan for efficient function composition
struct CompPlan
    # Placeholder for composition optimization data
    data::Vector{Int}
end

# Thread-local pool of pre-allocated Float64 coefficient buffers.
# Eliminates the dominant zeros(N) allocation inside math function temporaries.
# Each thread owns one pool per descriptor; acquire/release are lock-free.
const CTPS_POOL_SIZE = 32

mutable struct DescPool
    bufs  :: Vector{Vector{Float64}}         # raw coefficient buffers
    refs  :: Vector{Base.RefValue{UInt64}}   # pre-allocated degree_mask refs
    ctps  :: Vector{Any}                     # stores CTPS{Float64}; typed Any to avoid forward-ref cycle
    avail :: Vector{UInt8}                   # stack of available slot indices (1-based)
    sp    :: Int                             # stack pointer (CTPS_POOL_SIZE = full, 0 = empty)
end

function DescPool(N::Int)
    bufs  = [zeros(Float64, N) for _ in 1:CTPS_POOL_SIZE]
    refs  = [Ref(UInt64(0)) for _ in 1:CTPS_POOL_SIZE]
    ctps  = Vector{Any}(undef, CTPS_POOL_SIZE)   # filled by _init_desc_pools!
    avail = UInt8.(1:CTPS_POOL_SIZE)
    return DescPool(bufs, refs, ctps, avail, CTPS_POOL_SIZE)
end

# Phase-2 initialiser: populate pre-allocated CTPS wrappers once desc is known.
# Called right after PSDesc construction so pool.ctps[i] share bufs[i] and refs[i].
function _init_desc_pools!(pools::Vector{DescPool}, desc)  # desc::PSDesc (forward ref)
    for pool in pools
        for i in 1:CTPS_POOL_SIZE
            pool.ctps[i] = CTPS{Float64}(pool.bufs[i], pool.refs[i])
        end
    end
end

# TPSA Descriptor - immutable, shared metadata
struct PSDesc
    nv::Int                       # number of variables
    order::Int                    # maximum order
    N::Int                        # total number of coefficients
    Nd::Vector{Int}               # size per degree
    off::Vector{Int}              # start offset per degree (1-based)
    polymap::PolyMap              # index mapping (index → exponent)
    exp_to_idx::Dict              # reverse map: SVector{nv+1,UInt8} → Int (concrete per instance)
    mul::Vector{MulSchedule2D}    # 2D k-map multiplication schedules indexed as (di,dj)
    comp_plan::CompPlan           # composition build plan
    _pools::Vector{DescPool}      # per-thread coefficient buffer pools (Float64 only)
end

# Thread-safe cache for PSDesc instances
const DESC_CACHE = Dict{Tuple{Int,Int}, PSDesc}()

# Global default descriptor
const GLOBAL_DESC = Ref{Union{PSDesc, Nothing}}(nothing)

"""
    set_descriptor!(nv::Int, order::Int)

Set the global default descriptor for all CTPS operations.
This should be called once at the beginning of your program.

# Arguments
- `nv::Int`: Number of variables
- `order::Int`: Maximum order

# Example
```julia
using PolySeries
set_descriptor!(3, 4)  # 3 variables, order 4
x = CTPS(0.0, 1)       # Create variable x
y = CTPS(0.0, 2)       # Create variable y
```
"""
function set_descriptor!(nv::Int, order::Int)
    GLOBAL_DESC[] = PSDesc(nv, order)
    return GLOBAL_DESC[]
end

"""
    get_descriptor()

Get the current global descriptor. Throws an error if not set.
"""
function get_descriptor()
    if GLOBAL_DESC[] === nothing
        error("No global descriptor set. Call set_descriptor!(nv, order) first.")
    end
    return GLOBAL_DESC[]
end

"""
    clear_descriptor!()

Clear the global descriptor.
"""
function clear_descriptor!()
    GLOBAL_DESC[] = nothing
end
const DESC_CACHE_LOCK = ReentrantLock()

# Constructor with caching (thread-safe)
function PSDesc(nv::Int, order::Int)
    key = (nv, order)
    
    # Thread-safe cache lookup: always acquire lock
    # Note: Base.Dict is NOT safe for concurrent read/write, so we must lock even for reads
    return lock(DESC_CACHE_LOCK) do
        # Check if already cached
        desc = get(DESC_CACHE, key, nothing)
        if desc !== nothing
            return desc
        end
        
        # Compute total number of coefficients
        N = binomial(nv + order, order)
        
        # Compute size per degree (number of monomials at each degree)
        Nd = zeros(Int, order + 1)
        Nd[1] = 1  # degree 0: just constant
        for d in 1:order
            Nd[d + 1] = binomial(nv + d, d) - binomial(nv + d - 1, d - 1)
        end
        
        # Compute start offset per degree (1-based indexing)
        off = zeros(Int, order + 1)
        off[1] = 1
        for d in 1:order
            off[d + 1] = off[d] + Nd[d]
        end
        
        # Create polymap and reverse lookup
        polymap = PolyMap(nv, order)
        
        # Build reverse map: exponent → index (using SVector for type stability)
        # SVector{K,UInt8} is stack-allocated and type-stable for small nv
        K = nv + 1
        exp_to_idx = Dict{SVector{K,UInt8}, Int32}()
        for idx in 1:N
            # Read directly from matrix to avoid allocation (no getindexmap slice)
            exp_svec = SVector{K,UInt8}(UInt8(polymap.map[idx, v]) for v in 1:K)
            exp_to_idx[exp_svec] = Int32(idx)
        end
        
        # Build 2D k-map multiplication schedules for all degree pairs (di, dj)
        # Symmetric schedule: only build (di ≥ dj) pairs with dk ≤ order.
        # mul! dispatches to diagonal/symmetric/asymmetric kernels at runtime,
        # combining the (di,dj) and (dj,di) contributions in one pass.
        # Schedule size: (order+1)(order+2)/2 valid pairs ≈ half of full square.
        mul = MulSchedule2D[]
        for di in 0:order
            for dj in 0:min(di, order - di)   # dj ≤ di  AND  dk = di+dj ≤ order
                sched = build_mul_schedule_2d(polymap, exp_to_idx, nv, di, dj, off, Nd)
                push!(mul, sched)
            end
        end
        
        # Create composition plan (placeholder)
        comp_plan = CompPlan(Int[])

        # Pre-allocate per-thread coefficient buffer pools (Float64 only)
        pools = [DescPool(N) for _ in 1:Threads.nthreads()]

        desc = PSDesc(nv, order, N, Nd, off, polymap, exp_to_idx, mul, comp_plan, pools)
        _init_desc_pools!(pools, desc)   # phase-2: populate CTPS wrappers now that desc exists
        DESC_CACHE[key] = desc
        return desc
    end
end

# Build 2D k-map schedule for degree pair (di ≥ dj) → dk = di + dj
#
# Only called for the upper triangular (di ≥ dj) with dk = di+dj ≤ order.
# For deglex ordering with uniform max order, every (i_local, j_local) pair
# produces a valid product monomial, so no validity checks are needed.
#
# k_local[j_local, i_local] = (global k index in dk-block) - k_start
#   Julia column-major → iterating j_local in the inner loop is stride-1.
#
# Symmetry property (when di == dj): k_local is symmetric, i.e.
#   k_local[j, i] == k_local[i, j]  (addition of exponents is commutative)
# This lets mul! use a triangular loop for diagonal degree pairs.
function build_mul_schedule_2d(polymap::PolyMap, exp_to_idx::Dict,
                                nv::Int,
                                di::Int, dj::Int,
                                off::Vector{Int}, Nd::Vector{Int})
    dk        = di + dj
    Ni        = Nd[di + 1]
    Nj        = Nd[dj + 1]
    i_start   = off[di + 1]
    j_start   = off[dj + 1]
    k_start   = off[dk + 1]
    K = nv + 1

    k_local = Matrix{Int32}(undef, Nj, Ni)   # (j, i) — j is inner index → sequential

    @inbounds for i_local in 1:Ni
        i = i_start + i_local - 1
        for j_local in 1:Nj
            j = j_start + j_local - 1
            exp_k = SVector{K,UInt8}(
                UInt8(polymap.map[i, v] + polymap.map[j, v]) for v in 1:K)
            k = exp_to_idx[exp_k]               # always valid: dk ≤ order
            k_local[j_local, i_local] = Int32(k)   # 1-based absolute index
        end
    end

    return MulSchedule2D(k_local,
                         Int32(i_start), Int32(j_start), Int32(k_start),
                         Int32(Ni), Int32(Nj),
                         UInt8(di), UInt8(dj))
end

# CTPS — two fields only; the descriptor is NOT stored per-instance.
# Removing the `desc::PSDesc` field:
#   • Eliminates Enzyme's "constant stored into differentiable struct" error
#     (PSDesc is a large const struct that Enzyme cannot differentiate through)
#   • Reduces CTPS memory footprint (no redundant pointer per instance)
#   • Matches the single-active-descriptor design: all live CTPS objects share
#     the same global descriptor set by set_descriptor!(nv, order).
struct CTPS{T}
    c           :: Vector{T}                # coefficients, length get_descriptor().N
    degree_mask :: Base.RefValue{UInt64}    # bit i set iff degree-i block is active
end

# Virtual `.desc` property — returns the active global descriptor.
# All code using `ctps.desc` works unchanged with zero per-instance storage cost.
@inline Base.getproperty(ctps::CTPS, s::Symbol) =
    s === :desc ? get_descriptor() : getfield(ctps, s)

# Compute degree mask from coefficients
function compute_degree_mask(c::Vector{T}, desc::PSDesc) where T
    mask = UInt64(0)
    order = desc.order
    @inbounds for d in 0:order
        d_start = desc.off[d + 1]
        d_end = d_start + desc.Nd[d + 1] - 1
        for i in d_start:d_end
            if !iszero(c[i])
                mask |= (UInt64(1) << d)
                break
            end
        end
    end
    return mask
end

# Update the degree mask after manual coefficient modifications
function update_degree_mask!(ctps::CTPS)
    ctps.degree_mask[] = compute_degree_mask(ctps.c, ctps.desc)
    return ctps
end

# Compute output degree mask from two input masks:
# bit dk is set iff there exist di, dj with di+dj==dk, bit di in mask1, bit dj in mask2.
# O(order²) vs O(N) for compute_degree_mask; order is typically small (≤20).
# Conservative: may have false positives if all contributions cancel.
@inline function compose_degree_mask(mask1::UInt64, mask2::UInt64, order::Int)
    result = UInt64(0)
    @inbounds for di in 0:order
        (mask1 & (UInt64(1) << di)) == 0 && continue
        @inbounds for dj in 0:(order - di)
            (mask2 & (UInt64(1) << dj)) == 0 && continue
            result |= (UInt64(1) << (di + dj))
        end
    end
    return result
end

# Fast internal constructors that reuse an existing PSDesc (no lock acquisition).
@inline function _ctps_constant(a::T, desc::PSDesc) where T
    c = Vector{T}(undef, desc.N)   # lazy: only c[1] is written
    c[1] = a
    mask = iszero(a) ? UInt64(0) : UInt64(1)
    return CTPS{T}(c, Ref(mask))
end

@inline function _ctps_zero(::Type{T}, desc::PSDesc) where T
    return CTPS{T}(Vector{T}(undef, desc.N), Ref(UInt64(0)))
end

# ── Thread-local pool: acquire / release ─────────────────────────────────────
#
# _ctps_pooled(T, desc) → (pool_idx::UInt8, CTPS{T})
#   Returns a CTPS backed by a pre-zeroed pool buffer.
#   pool_idx == 0x00 means a heap fallback was used (T ≠ Float64, pool full,
#   or more threads than pools). The caller MUST call _pool_release! with the
#   same idx when done with the CTPS.
#
# _ctps_pooled_copy(src, desc) → (pool_idx, CTPS)
#   Like _ctps_pooled but copies src's active range into the new buffer,
#   equivalent to CTPS(src) without the zeros(N) alloc.
#
# _pool_release!(idx, ctps, desc)
#   Zeros the CTPS's active range, returns the slot to the pool.
#   No-op for idx == 0x00 (heap fallback; GC handles it).

@inline function _ctps_pooled(::Type{Float64}, desc::PSDesc)
    tid = Threads.threadid()
    if tid <= length(desc._pools)
        pool = desc._pools[tid]
        sp = pool.sp
        if sp > 0
            idx = pool.avail[sp]
            pool.sp = sp - 1
            pool.refs[idx][] = UInt64(0)   # reset degree_mask in-place, no allocation
            return (idx, pool.ctps[idx]::CTPS{Float64})   # type assert: tag-check only, 0 allocs
        end
    end
    return (UInt8(0), _ctps_zero(Float64, desc))   # fallback: heap
end

@inline function _ctps_pooled(::Type{T}, desc::PSDesc) where T
    return (UInt8(0), _ctps_zero(T, desc))         # non-Float64: heap
end

@inline function _ctps_pooled_copy(src::CTPS{Float64}, desc::PSDesc)
    tid = Threads.threadid()
    if tid <= length(desc._pools)
        pool = desc._pools[tid]
        sp = pool.sp
        if sp > 0
            idx = pool.avail[sp]
            pool.sp = sp - 1
            tm = src.degree_mask[]
            pool.refs[idx][] = tm
            if tm != 0
                (s, e) = active_range_bounds(desc, tm)
                dst_buf = pool.bufs[idx]
                src_buf = src.c
                @inbounds @simd for i in s:e; dst_buf[i] = src_buf[i]; end
            end
            return (idx, pool.ctps[idx]::CTPS{Float64})   # pre-allocated CTPS, zero new allocations
        end
    end
    return (UInt8(0), CTPS(src))   # fallback: heap
end

@inline function _ctps_pooled_copy(src::CTPS{T}, desc::PSDesc) where T
    return (UInt8(0), CTPS(src))
end

@inline function _pool_release!(idx::UInt8, ctps::CTPS{Float64}, desc::PSDesc)
    idx == 0x00 && return
    tid = Threads.threadid()
    tid > length(desc._pools) && return
    pool = desc._pools[tid]
    dm = pool.refs[idx][]
    if dm != 0
        (s, e) = active_range_bounds(desc, dm)
        buf = pool.bufs[idx]
        @inbounds @simd for i in s:e; buf[i] = 0.0; end
    end
    sp = pool.sp + 1
    pool.sp = sp
    pool.avail[sp] = idx
    return nothing
end

@inline function _pool_release!(::UInt8, ::CTPS, ::PSDesc)
    return nothing   # non-Float64: let GC handle it
end


function CTPS(T::Type, nv::Int, order::Int)
    desc = set_descriptor!(nv, order)
    c = zeros(T, desc.N)
    return CTPS{T}(c, Ref(UInt64(0)))
end

# Constructor: constant CTPS
function CTPS(a::T, nv::Int, order::Int) where T
    desc = set_descriptor!(nv, order)
    c = zeros(T, desc.N)
    c[1] = a
    mask = iszero(a) ? UInt64(0) : UInt64(1)  # Degree 0 has non-zero
    return CTPS{T}(c, Ref(mask))
end

# Constructor: variable CTPS (a + δxₙ)
function CTPS(a::T, n::Int, nv::Int, order::Int) where T
    if n <= nv && n > 0
        desc = set_descriptor!(nv, order)
        c = zeros(T, desc.N)
        c[n + 1] = one(T)  # linear term for variable n
        c[1] = a           # constant term
        # Degree 0 and degree 1 have non-zeros
        mask = UInt64(0x3)  # bits 0 and 1 set
        if iszero(a)
            mask = UInt64(0x2)  # only bit 1 set
        end
        return CTPS{T}(c, Ref(mask))
    else
        error("Variable index out of range in CTPS")
    end
end

# -----------------------------------------------------------------------
# RANGE-LIMITED HELPERS
#
# `active_range_bounds(desc, mask)` returns the (start, stop) 1-based indices
# of the coefficient slice that covers all non-zero degrees in `mask`.
# Operating only on this range avoids touching zero pages for sparse CTPS.
# -----------------------------------------------------------------------
@inline function active_range_bounds(desc::PSDesc, mask::UInt64)
    mask == 0 && return (1, 0)   # empty range
    min_deg = trailing_zeros(mask) % Int
    max_deg = (63 - leading_zeros(mask)) % Int
    start = desc.off[min_deg + 1]
    stop  = desc.off[max_deg + 1] + desc.Nd[max_deg + 1] - 1
    return (start, stop)
end

# Copy constructor — range-limited: only copies active coefficient range.
# Uses undef allocation; positions outside [s,e] are garbage but degree_mask
# guarantees they are never read.
function CTPS(M::CTPS{T}) where T
    desc = M.desc
    c    = Vector{T}(undef, desc.N)   # lazy: only active range is written
    mask = M.degree_mask[]
    if mask != 0
        (s, e) = active_range_bounds(desc, mask)
        @inbounds @simd for i in s:e
            c[i] = M.c[i]
        end
    end
    return CTPS{T}(c, Ref(mask))
end

# ========== Simplified constructors using global descriptor ==========

# Constructor: zero CTPS using global descriptor
function CTPS(T::Type)
    desc = get_descriptor()
    c = zeros(T, desc.N)
    return CTPS{T}(c, Ref(UInt64(0)))
end

# Constructor: constant CTPS using global descriptor
function CTPS(a::T) where T<:Number
    desc = get_descriptor()
    c = zeros(T, desc.N)
    c[1] = a
    mask = iszero(a) ? UInt64(0) : UInt64(1)
    return CTPS{T}(c, Ref(mask))
end

# Constructor: variable CTPS using global descriptor
function CTPS(a::T, n::Int) where T<:Number
    desc = get_descriptor()
    nv = desc.nv
    if n <= nv && n > 0
        c = zeros(T, desc.N)
        c[n + 1] = one(T)  # linear term for variable n
        c[1] = a           # constant term
        mask = UInt64(0x3)  # bits 0 and 1 set
        if iszero(a)
            mask = UInt64(0x2)  # only bit 1 set
        end
        return CTPS{T}(c, Ref(mask))
    else
        error("Variable index $n out of range (must be 1 to $nv)")
    end
end

# ========== End simplified constructors ==========

function cst(ctps::CTPS{T}) where T
    return ctps.c[1]
end

function findindex(ctps::CTPS{T}, indexmap::Vector{Int}) where T
    # find the index of the indexmap in the coefficient vector
    # indexmap is a vector of length nv + 1, e.g. [0, 1, 1] for x1^1 * x2^1
    dim = ctps.desc.nv
    if length(indexmap) == dim
        # Compute total and work with conceptual [total; indexmap]
        total = Base.sum(indexmap)
        # Build cumsum incrementally: cumsum[1]=total, cumsum[i]=cumsum[i-1]-indexmap[i-1]
        # We read conceptual indexmap as: [total; indexmap[1]; indexmap[2]; ...; indexmap[dim]]
        cumsum_val = total
        result = Int(1)
        for i in dim:-1:1
            # We need cumsum[dim - i + 1]
            # cumsum[1] = total (already have)
            # cumsum[2] = total - indexmap[1]
            # cumsum[k] = total - sum(indexmap[1:k-1])
            # For iteration i (dim downto 1), we need cumsum[dim - i + 1]
            # So we need to have subtracted indexmap[1:dim-i]
            # Build cumsum by subtracting as we go backwards
            if cumsum_val == 0
                break
            end
            if cumsum_val < 0
                error("The index map has invalid component")
            end
            result += binomial(cumsum_val - 1 + i, i)
            # Prepare for next iteration: subtract indexmap[dim - i + 1]
            cumsum_val -= indexmap[dim - i + 1]
        end
        return result
    end
    if length(indexmap) != (dim + 1)
        error("Index map does not have correct length")
    end
    # Original pattern: cumsum[1]=indexmap[1], cumsum[i]=cumsum[i-1]-indexmap[i]
    # For i in dim:-1:1, we need cumsum[dim - i + 1]
    # Build cumsum values incrementally without allocating array
    # cumsum[k] = indexmap[1] - sum(indexmap[2:k])
    
    # Start: we'll build cumsum values on the fly
    # For the first iteration (i=dim), we need cumsum[1] = indexmap[1]
    cumsum_val = indexmap[1]
    result = Int(1)
    
    for i in dim:-1:1
        # At loop entry, cumsum_val = cumsum[dim - i + 1]
        if cumsum_val == 0
            break
        end
        if cumsum_val < 0 || indexmap[dim - i + 2] < 0
            error("The index map has invalid component")
        end
        result += binomial(cumsum_val - 1 + i, i)
        # Update cumsum_val for next iteration: cumsum[dim-i+2] = cumsum[dim-i+1] - indexmap[dim-i+2]
        if i > 1  # Only update if there's a next iteration
            cumsum_val -= indexmap[dim - i + 2]
        end
    end
    return result
end

# function findpower(ctps::CTPS{T, TPS_Dim, Max_TPS_Degree}, n::Int) where {T, TPS_Dim, Max_TPS_Degree}
#     if n < ctps.terms
#         return getindexmap(ctps.polymap[], n)
#     else
#         error("The index is out of range")
#     end
# end

# function redegree!(ctps::CTPS{T, TPS_Dim, Max_TPS_Degree}, degree::Int) where {T, TPS_Dim, Max_TPS_Degree}
#     ctps.degree = min(degree, Max_TPS_Degree)
#     ctps.terms = binomial(TPS_Dim + ctps.degree, ctps.degree)
#     new_map = [i <= length(ctps.map) ? ctps.map[i] : zero(T) for i in 1:ctps.terms]
#     ctps.map = new_map
# end
# function redegree(ctps::CTPS{T, TPS_Dim, Max_TPS_Degree}, degree::Int) where {T, TPS_Dim, Max_TPS_Degree}
#     degree = min(degree, Max_TPS_Degree)
#     terms = binomial(TPS_Dim + degree, degree)
#     new_map = zeros(T, terms)
#     for i in 1:ctps.terms
#         new_map[i] = ctps.map[i]
#     end
#     # new_map = [i <= length(ctps.map) ? ctps.map[i] : 0.0 for i in 1:terms]
#     # polymap = getOrCreatePolyMap(TPS_Dim, Max_TPS_Degree)
#     ctps_new = CTPS{T, TPS_Dim, Max_TPS_Degree}(degree, terms, new_map, ctps.polymap)
#     return ctps_new
# end
# function redegree(ctps::CTPS{T, TPS_Dim, Max_TPS_Degree}, degree::Int) where {T, TPS_Dim, Max_TPS_Degree}
#     degree = min(degree, Max_TPS_Degree)
#     terms = binomial(TPS_Dim + degree, degree)
#     new_map = zeros(T, terms)
#     new_map_buffer = Zygote.Buffer(new_map)
#     for i in 1:ctps.terms
#         new_map_buffer[i] = ctps.map[i]
#     end
#     for i in ctps.terms+1:terms
#         new_map_buffer[i] = zero(T)
#     end
#     new_map = copy(new_map_buffer)
#     ctps_new = CTPS{T, TPS_Dim, Max_TPS_Degree}(degree, terms, new_map, PolyMap(TPS_Dim, Max_TPS_Degree))
#     return ctps_new
# end
function assign!(ctps::CTPS{T}, a::T, n_var::Int) where T
    if n_var <= ctps.desc.nv && n_var > 0
        ctps.c[n_var + 1] = one(T)
        ctps.c[1] = a
        return nothing
    else
        error("Variable index out of range in CTPS")
    end
end

function assign!(ctps::CTPS{T}, a::T) where T
    ctps.c[1] = a
    return nothing
end

function reassign!(ctps::CTPS{T}, a::T, n_var::Int) where T
    if n_var <= ctps.desc.nv && n_var > 0
        fill!(ctps.c, zero(T))
        ctps.c[n_var + 1] = one(T)
        ctps.c[1] = a
        return nothing
    else
        error("Variable index out of range in CTPS")
    end
end



@inline function element(ctps::CTPS{T}, ind::Vector{Int}) where T
    result = findindex(ctps, ind)
    return ctps.c[result]
end

# Defining callable instance
function (ctps::CTPS{T})(args::T...) where T
    # Descriptor lookup outside the loop — avoids repeated global Ref access
    # through the virtual getproperty, whose Union{PSDesc,Nothing} return type
    # causes Enzyme to expand both branches on every loop iteration.
    desc = ctps.desc::PSDesc
    nv   = desc.nv
    if length(args) != nv
        error("Number of arguments does not match the number of variables in the CTPS")
    end

    return_value = ctps.c[1]  # Start with the constant term
    pm = desc.polymap.map     # Get the polymap matrix for exponent lookups
    @inbounds for i in 2:length(ctps.c)
        val = ctps.c[i]
        for v in 1:nv
            e = Int(pm[i, v + 1]) # Enzyme likes Int better
            e == 0 && continue # Skip multipling by 1
            val *= args[v]^e
        end 
        return_value += val
    end
    return return_value
end


# Overloaded operations
import Base: +, -, *, /, sin, cos, tan, sinh, cosh, asin, acos, sqrt, ^, inv, exp, log, copy!, show

# -----------------------------------------------------------------------
# Pretty-printing helpers
# -----------------------------------------------------------------------
const _SUPERSCRIPTS = ('⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹')
const _SUBSCRIPTS   = ('₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉')

function _int_to_subscript(n::Int)
    n < 10 && return string(_SUBSCRIPTS[n + 1])
    buf = Char[]
    while n > 0
        pushfirst!(buf, _SUBSCRIPTS[n % 10 + 1])
        n ÷= 10
    end
    return String(buf)
end

function _int_to_superscript(n::Int)
    n == 1 && return ""          # exponent 1 is implicit
    n < 10 && return string(_SUPERSCRIPTS[n + 1])
    buf = Char[]
    while n > 0
        pushfirst!(buf, _SUPERSCRIPTS[n % 10 + 1])
        n ÷= 10
    end
    return String(buf)
end

# Format a monomial string from the exponent row of PolyMap.
# map_row layout: [total_degree, e₁, e₂, ..., e_{nv-1}, 0]
#   - Columns 2..nv store e₁..e_{nv-1} explicitly.
#   - The last column (index nv+1) is always 0; e_nv is implicit:
#       e_nv = total_degree - sum(e₁..e_{nv-1})
function _monomial_str(map_row::AbstractVector, nv::Int)
    io = IOBuffer()
    td = Int(map_row[1])
    rest = td
    for v in 1:nv - 1
        e = Int(map_row[v + 1])
        rest -= e
        e == 0 && continue
        print(io, "x", _int_to_subscript(v), _int_to_superscript(e))
    end
    # last variable: exponent is implicit
    if rest > 0
        print(io, "x", _int_to_subscript(nv), _int_to_superscript(rest))
    end
    return String(take!(io))
end

_is_negative(c::Real) = c < zero(c)
_is_negative(_) = false          # complex / other: always print with +

# Complex coefficients need parentheses to avoid `+ 1.0 + 2.0im x₁` ambiguity.
_needs_parens(::Real)    = false
_needs_parens(_)         = true

"""
    show(io, ctps)

Pretty-print a `CTPS` as a multivariate polynomial. Only non-zero coefficients
in active degree blocks (tracked by `degree_mask`) are shown.

Example output:
```
CTPS{Float64}: nv=2, order=3
  1.0
  + 2.0 x₁
  - 3.0 x₂
  + 1.5 x₁² x₂
```
"""
# Colors cycled per degree order when the IO supports color.
# Degree 0 (constant) is printed in normal/default color; degrees 1+ alternate.
const _DEGREE_COLORS = (:normal, :cyan, :green, :yellow, :magenta, :light_blue, :light_red)

@inline function _cprint(io::IO, use_color::Bool, color::Symbol, args...)
    use_color ? printstyled(io, args...; color) : print(io, args...)
end

function Base.show(io::IO, ctps::CTPS{T}) where T
    desc       = ctps.desc
    mask       = ctps.degree_mask[]
    line_width = 80
    use_color  = get(io, :color, false)

    print(io, "CTPS{", T, "}: nv=", desc.nv, ", order=", desc.order)

    first_term = true
    col = 0

    for d in 0:desc.order
        ((mask >> d) & 1 == 0) && continue
        color = _DEGREE_COLORS[mod1(d + 1, length(_DEGREE_COLORS))]
        s = desc.off[d + 1]
        e = s + desc.Nd[d + 1] - 1
        degree_started = false
        for i in s:e
            c = ctps.c[i]
            iszero(c) && continue
            mono = _monomial_str(@view(desc.polymap.map[i, :]), desc.nv)

            if first_term
                frag = if isempty(mono)
                    string(c)
                elseif _is_negative(c)
                    string("-", -c, " ", mono)
                elseif _needs_parens(c)
                    string("(", c, ") ", mono)
                else
                    string(c, " ", mono)
                end
                print(io, "\n  ")
                _cprint(io, use_color, color, frag)
                col = 2 + length(frag)
                first_term = false
                degree_started = true
            else
                frag = if isempty(mono)
                    if _is_negative(c)
                        string("- ", -c)
                    elseif _needs_parens(c)
                        string("+ (", c, ")")
                    else
                        string("+ ", c)
                    end
                else
                    if _is_negative(c)
                        string("- ", -c, " ", mono)
                    elseif _needs_parens(c)
                        string("+ (", c, ") ", mono)
                    else
                        string("+ ", c, " ", mono)
                    end
                end
                sep = "  "
                if !degree_started || col + length(sep) + length(frag) > line_width
                    print(io, "\n  ")
                    _cprint(io, use_color, color, frag)
                    col = 2 + length(frag)
                else
                    print(io, sep)
                    _cprint(io, use_color, color, frag)
                    col += length(sep) + length(frag)
                end
                degree_started = true
            end
        end
    end

    first_term && print(io, "\n  0")
end


function add!(result::CTPS{T}, ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    m1 = ctps1.degree_mask[]; m2 = ctps2.degree_mask[]
    m_out = m1 | m2
    if m_out != 0
        only1 = m1 & ~m2    # degrees active in ctps1 only  → copy from ctps1
        only2 = m2 & ~m1    # degrees active in ctps2 only  → copy from ctps2
        both  = m1  &  m2   # degrees active in both        → add
        if only1 != 0
            (s, e) = active_range_bounds(ctps1.desc, only1)
            @inbounds @simd for i in s:e; result.c[i] = ctps1.c[i]; end
        end
        if only2 != 0
            (s, e) = active_range_bounds(ctps1.desc, only2)
            @inbounds @simd for i in s:e; result.c[i] = ctps2.c[i]; end
        end
        if both != 0
            (s, e) = active_range_bounds(ctps1.desc, both)
            @inbounds @simd for i in s:e; result.c[i] = ctps1.c[i] + ctps2.c[i]; end
        end
    end
    result.degree_mask[] = m_out
    return nothing
end

function add!(result::CTPS{T}, ctps1::CTPS{T}, a::T) where T
    m1 = ctps1.degree_mask[]
    if m1 != 0
        (s, e) = active_range_bounds(ctps1.desc, m1)
        @inbounds @simd for i in s:e
            result.c[i] = ctps1.c[i]
        end
    end
    # c[1] is valid only if bit-0 is in m1 (lazy-zero: otherwise garbage)
    c0 = (m1 & UInt64(1) != 0) ? ctps1.c[1] : zero(T)
    result.c[1] = c0 + a
    result.degree_mask[] = (m1 & ~UInt64(1)) | (iszero(result.c[1]) ? UInt64(0) : UInt64(1))
    return nothing
end

function addto!(ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    m2 = ctps2.degree_mask[]
    m2 == 0 && return nothing
    m1 = ctps1.degree_mask[]
    new_bits   = m2 & ~m1  # degrees only in ctps2 → first write (=)
    accum_bits = m2 &  m1  # degrees in both        → accumulate (+=)
    if new_bits != 0
        (s, e) = active_range_bounds(ctps1.desc, new_bits)
        @inbounds @simd for i in s:e; ctps1.c[i] = ctps2.c[i]; end
    end
    if accum_bits != 0
        (s, e) = active_range_bounds(ctps1.desc, accum_bits)
        @inbounds @simd for i in s:e; ctps1.c[i] += ctps2.c[i]; end
    end
    ctps1.degree_mask[] |= m2
    return nothing
end

function sub!(result::CTPS{T}, ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    m1 = ctps1.degree_mask[]; m2 = ctps2.degree_mask[]
    m_out = m1 | m2
    if m_out != 0
        only1 = m1 & ~m2    # active only in ctps1 → copy from ctps1
        only2 = m2 & ~m1    # active only in ctps2 → negate from ctps2
        both  = m1  &  m2   # active in both       → subtract
        if only1 != 0
            (s, e) = active_range_bounds(ctps1.desc, only1)
            @inbounds @simd for i in s:e; result.c[i] = ctps1.c[i]; end
        end
        if only2 != 0
            (s, e) = active_range_bounds(ctps1.desc, only2)
            @inbounds @simd for i in s:e; result.c[i] = -ctps2.c[i]; end
        end
        if both != 0
            (s, e) = active_range_bounds(ctps1.desc, both)
            @inbounds @simd for i in s:e; result.c[i] = ctps1.c[i] - ctps2.c[i]; end
        end
    end
    result.degree_mask[] = m_out
    return nothing
end

function subfrom!(ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    m2 = ctps2.degree_mask[]
    m2 == 0 && return nothing
    m1 = ctps1.degree_mask[]
    new_bits   = m2 & ~m1  # degrees only in ctps2 → first write (= -ctps2)
    accum_bits = m2 &  m1  # degrees in both        → subtract (-=)
    if new_bits != 0
        (s, e) = active_range_bounds(ctps1.desc, new_bits)
        @inbounds @simd for i in s:e; ctps1.c[i] = -ctps2.c[i]; end
    end
    if accum_bits != 0
        (s, e) = active_range_bounds(ctps1.desc, accum_bits)
        @inbounds @simd for i in s:e; ctps1.c[i] -= ctps2.c[i]; end
    end
    ctps1.degree_mask[] |= m2
    return nothing
end

function scale!(ctps::CTPS{T}, a::T) where T
    mask = ctps.degree_mask[]
    if mask != 0
        (s, e) = active_range_bounds(ctps.desc, mask)
        @inbounds @simd for i in s:e
            ctps.c[i] *= a
        end
    end
    return nothing
end

# 3-arg scale: dest = src * a  (range-limited copy + multiply)
function scale!(dest::CTPS{T}, src::CTPS{T}, a::T) where T
    sm = src.degree_mask[]
    dm = dest.degree_mask[]
    # Zero out degrees in dest that src doesn't cover
    extra = dm & ~sm
    if extra != 0
        (s, e) = active_range_bounds(dest.desc, extra)
        @inbounds @simd for i in s:e; dest.c[i] = zero(T); end
    end
    if sm != 0
        (s, e) = active_range_bounds(src.desc, sm)
        @inbounds @simd for i in s:e
            dest.c[i] = src.c[i] * a
        end
    end
    dest.degree_mask[] = iszero(a) ? UInt64(0) : sm
    return nothing
end

# result = a * c1 + b * c2  (single range-limited pass, zero allocations)
# The canonical in-place form of the linear combination `a*c1 + b*c2 → result`.
# Used in the rotation step: nx1 = cos_μ*x1 + sin_μ*pmx.
function scaleadd!(result::CTPS{T}, a::T, c1::CTPS{T}, b::T, c2::CTPS{T}) where T
    m1   = c1.degree_mask[]
    m2   = c2.degree_mask[]
    ma   = iszero(a) ? UInt64(0) : m1   # effective mask for a*c1
    mb   = iszero(b) ? UInt64(0) : m2   # effective mask for b*c2
    mout = ma | mb
    dm   = result.degree_mask[]
    # Zero degrees present in dest but not in the output
    extra = dm & ~mout
    if extra != 0
        (s, e) = active_range_bounds(result.desc, extra)
        @inbounds @simd for i in s:e; result.c[i] = zero(T); end
    end
    # Split mout into sub-ranges to avoid reading garbage from inactive source
    onlya   = ma & ~mb
    onlyb   = mb & ~ma
    both_ab = ma  &  mb
    if onlya != 0
        (s, e) = active_range_bounds(result.desc, onlya)
        @inbounds @simd for i in s:e; result.c[i] = a * c1.c[i]; end
    end
    if onlyb != 0
        (s, e) = active_range_bounds(result.desc, onlyb)
        @inbounds @simd for i in s:e; result.c[i] = b * c2.c[i]; end
    end
    if both_ab != 0
        (s, e) = active_range_bounds(result.desc, both_ab)
        @inbounds @simd for i in s:e; result.c[i] = a * c1.c[i] + b * c2.c[i]; end
    end
    result.degree_mask[] = mout
    return nothing
end

# ── PSWorkspace ─────────────────────────────────────────────────────────────
#
# Pre-allocated pool of CTPS objects for zero-allocation user-level code.
# Usage pattern:
#
#   ws  = PSWorkspace(desc, 16)    # pre-allocate 16 CTPS slots
#   t1  = borrow!(ws)               # obtain a zero CTPS from the pool
#   mul!(t1, x[1], x[1])            # t1 = x1^2, no heap alloc
#   ...                              # use t1 in further in-place ops
#   release!(ws, t1)                 # return slot to pool (active range zeroed)
#
# Notes:
#   • CTPS slots are type Float64 only (same as DescPool).  For other element
#     types fall back to heap via _ctps_zero.
#   • `borrow!` returns a CTPS{Float64} backed by a pre-allocated buffer.
#   • `release!` zeros only the active degree range — O(active_range), not O(N).
#   • The workspace is NOT thread-safe: create one per thread or protect access.
mutable struct PSWorkspace
    desc     :: PSDesc
    bufs     :: Vector{CTPS{Float64}}   # pre-allocated CTPS objects
    avail    :: Vector{Int}             # stack of available indices
    sp       :: Int                     # stack pointer (sp==length(bufs) → all free)
    id_to_idx :: Dict{UInt64, Int}      # objectid(ctps.c) → slot index, O(1) release
end

function PSWorkspace(desc::PSDesc, n::Int = 32)
    bufs  = [CTPS{Float64}(zeros(Float64, desc.N), Ref(UInt64(0)))
             for _ in 1:n]
    avail = collect(1:n)
    id_to_idx = Dict{UInt64, Int}(objectid(bufs[i].c) => i for i in 1:n)
    return PSWorkspace(desc, bufs, avail, n, id_to_idx)
end

"""    borrow!(ws::PSWorkspace) -> CTPS{Float64}

Obtain a zero CTPS from the workspace without heap allocation.
Must be paired with `release!(ws, ctps)` when done."""
@inline function borrow!(ws::PSWorkspace)
    ws.sp == 0 && error("PSWorkspace exhausted — increase n at construction")
    idx  = ws.avail[ws.sp]
    ws.sp -= 1
    return ws.bufs[idx]
end

"""    release!(ws::PSWorkspace, ctps::CTPS{Float64})

Return a borrowed CTPS slot to the workspace.
Zeros only the active degree range before returning — O(active_range)."""
@inline function release!(ws::PSWorkspace, ctps::CTPS{Float64})
    dm = ctps.degree_mask[]
    if dm != 0
        (s, e) = active_range_bounds(ws.desc, dm)
        buf = ctps.c
        @inbounds @simd for i in s:e; buf[i] = 0.0; end
        ctps.degree_mask[] = UInt64(0)
    end
    idx = ws.id_to_idx[objectid(ctps.c)]
    ws.sp += 1
    ws.avail[ws.sp] = idx
    return nothing
end

function copy!(dest::CTPS{T}, src::CTPS{T}) where T
    # Zero out any degrees in dest that src doesn't have, then copy active range
    src_mask  = src.degree_mask[]
    dest_mask = dest.degree_mask[]
    extra_mask = dest_mask & ~src_mask
    if extra_mask != 0
        (s, e) = active_range_bounds(dest.desc, extra_mask)
        @inbounds @simd for i in s:e
            dest.c[i] = zero(T)
        end
    end
    if src_mask != 0
        (s, e) = active_range_bounds(src.desc, src_mask)
        @inbounds @simd for i in s:e
            dest.c[i] = src.c[i]
        end
    end
    dest.degree_mask[] = src_mask
    return nothing
end

function zero!(ctps::CTPS{T}) where T
    fill!(ctps.c, zero(T))
    ctps.degree_mask[] = UInt64(0)
    return nothing
end

# Zero only the currently-active degree range, then clear the mask.
# O(active_range) instead of O(N).  Used by in-place math functions to reset
# a workspace slot before writing — free for already-zero workspace slots
# (degree_mask == 0 → no-op).
@inline function _zero_active!(ctps::CTPS{T}) where T
    dm = ctps.degree_mask[]
    if dm != 0
        (s, e) = active_range_bounds(ctps.desc, dm)
        @inbounds @simd for i in s:e; ctps.c[i] = zero(T); end
        ctps.degree_mask[] = UInt64(0)
    end
end

# Range-limited accumulation helper:  sum += term * scale.
# With lazy-zero (undef) allocations, sum.c may be uninitialised for degrees
# not yet written.  On first touch of a degree block we use = (initialise)
# rather than += (accumulate) to avoid reading garbage.
@inline function _add_scaled!(sum::CTPS{T}, term::CTPS{T}, scale::T) where T
    tm = term.degree_mask[]
    (iszero(scale) || tm == 0) && return
    sm        = sum.degree_mask[]
    desc      = sum.desc
    new_bits  = tm & ~sm    # degrees not yet written to sum → must initialise
    accum_bits = tm &  sm   # degrees already written    → safe to accumulate
    if new_bits != 0 && accum_bits != 0
        # Mixed: zero the new blocks first, then accumulate the full tm range.
        (s2, e2) = active_range_bounds(desc, new_bits)
        @inbounds @simd for j in s2:e2; sum.c[j] = zero(T); end
        (s, e) = active_range_bounds(desc, tm)
        @inbounds @simd for j in s:e; sum.c[j] += term.c[j] * scale; end
    elseif new_bits != 0
        (s, e) = active_range_bounds(desc, new_bits)
        @inbounds @simd for j in s:e; sum.c[j] = term.c[j] * scale; end
    else
        (s, e) = active_range_bounds(desc, accum_bits)
        @inbounds @simd for j in s:e; sum.c[j] += term.c[j] * scale; end
    end
    sum.degree_mask[] = sm | tm
    return nothing
end

# In-place multiplication: result = ctps1 * ctps2
#
# Symmetric schedule kernel — desc.mul contains only (di ≥ dj) entries.
# For each entry three dispatch paths are used:
#
#   diagonal  (di == dj): triangular loop — ~50% fewer FLOPs than full square.
#     For j < i: cr[k] += c1[i]*c2[j] + c1[j]*c2[i]  (both contributions)
#     For j == i: cr[k] += c1[i]*c2[i]
#
#   symmetric (di > dj, both masks set): single pass for both directions.
#     cr[k] += c1[di][i]*c2[dj][j] + c1[dj][j]*c2[di][i]
#     — one L-table lookup per (i,j) pair instead of two separate passes.
#
#   asymmetric (di > dj, one direction): standard forward or reverse pass.
#
# k_local commutativity: for any di,dj pair (including di==dj), the output index
#   for (i in di-block, j in dj-block) equals the index for (j in dj-block, i in
#   di-block), because exp[i]+exp[j] == exp[j]+exp[i] (exponent addition commutes).
function mul!(result::CTPS{T}, ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    desc  = ctps1.desc
    order = desc.order
    c1    = ctps1.c
    c2    = ctps2.c
    cr    = result.c
    mask1 = ctps1.degree_mask[]
    mask2 = ctps2.degree_mask[]

    # Range-limited zero fill for the output degree band.
    if mask1 != 0 && mask2 != 0
        dk_min = (trailing_zeros(mask1) + trailing_zeros(mask2)) % Int
        dk_max = min((63 - leading_zeros(mask1)) % Int +
                     (63 - leading_zeros(mask2)) % Int, order)
        out_s = desc.off[dk_min + 1]
        out_e = desc.off[dk_max + 1] + desc.Nd[dk_max + 1] - 1
        @inbounds @simd for i in out_s:out_e
            cr[i] = zero(T)
        end

        @inbounds for sched in desc.mul
            # desc.mul only contains (di ≥ dj) entries — no empty sentinels.
            di = UInt32(sched.di)
            dj = UInt32(sched.dj)
            has_fwd = (mask1 >> di) & UInt64(1) != 0 && (mask2 >> dj) & UInt64(1) != 0
            has_rev = (di != dj) &&
                      (mask1 >> dj) & UInt64(1) != 0 && (mask2 >> di) & UInt64(1) != 0
            (!has_fwd && !has_rev) && continue

            Ni     = Int(sched.Ni)
            Nj     = Int(sched.Nj)
            i_base = Int(sched.i_start) - 1   # 0-based → c[i_base + i_local]
            j_base = Int(sched.j_start) - 1   # 0-based → c[j_base + j_local]
            k_mat  = sched.k_local             # Matrix{Int32}(Nj × Ni), 1-based absolute

            if di == dj
                # ── Diagonal: triangular loop ──────────────────────────────
                # Handles c1[di]*c2[di] without any double-counting.
                # Off-diagonal (j < i): combines (i,j) and (j,i) contributions.
                @inbounds @fastmath for i_local in 1:Ni
                    ai = c1[i_base + i_local]
                    bi = c2[i_base + i_local]
                    (iszero(ai) && iszero(bi)) && continue
                    @inbounds @fastmath for j_local in 1:i_local-1
                        kk = k_mat[j_local, i_local]
                        cr[kk] += ai * c2[j_base + j_local] +
                                  c1[j_base + j_local] * bi
                    end
                    # Self-product (j == i): no symmetry factor
                    cr[k_mat[i_local, i_local]] += ai * bi
                end

            elseif has_fwd && has_rev
                # ── Symmetric: one pass for both (di,dj) and (dj,di) ───────
                # cr[k] += c1[di][i]*c2[dj][j] + c1[dj][j]*c2[di][i]
                @inbounds @fastmath for i_local in 1:Ni
                    ai = c1[i_base + i_local]   # c1 in di-block
                    bi = c2[i_base + i_local]   # c2 in di-block
                    (iszero(ai) && iszero(bi)) && continue
                    @inbounds @fastmath for j_local in 1:Nj
                        kk = k_mat[j_local, i_local]
                        cr[kk] += ai * c2[j_base + j_local] +
                                  c1[j_base + j_local] * bi
                    end
                end

            elseif has_fwd
                # ── Forward only: c1[di] * c2[dj] ──────────────────────────
                @inbounds @fastmath for i_local in 1:Ni
                    ai = c1[i_base + i_local]
                    iszero(ai) && continue
                    @inbounds @fastmath for j_local in 1:Nj
                        cr[k_mat[j_local, i_local]] +=
                            ai * c2[j_base + j_local]
                    end
                end

            else  # has_rev only
                # ── Reverse only: c1[dj] * c2[di] ──────────────────────────
                @inbounds @fastmath for i_local in 1:Ni
                    bi = c2[i_base + i_local]   # c2 in di-block
                    iszero(bi) && continue
                    @inbounds @fastmath for j_local in 1:Nj
                        cr[k_mat[j_local, i_local]] +=
                            c1[j_base + j_local] * bi
                    end
                end
            end
        end
    end

    result.degree_mask[] = compose_degree_mask(mask1, mask2, order)
    return nothing
end

# + (range-limited: only touches active degree range)
function +(ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    if ctps1.desc !== ctps2.desc
        if ctps1.desc.nv != ctps2.desc.nv || ctps1.desc.order != ctps2.desc.order
            error("Cannot add CTPS with different descriptors: (nv=$(ctps1.desc.nv), order=$(ctps1.desc.order)) vs (nv=$(ctps2.desc.nv), order=$(ctps2.desc.order))")
        end
    end
    c = Vector{T}(undef, ctps1.desc.N)
    result = CTPS{T}(c, Ref(UInt64(0)))
    add!(result, ctps1, ctps2)
    return result
end

function +(ctps::CTPS{T}, a::Number) where T
    ctps_new = CTPS(ctps)   # range-limited undef copy
    m = ctps.degree_mask[]
    c0 = (m & UInt64(1) != 0) ? ctps.c[1] : zero(T)
    ctps_new.c[1] = c0 + T(a)
    ctps_new.degree_mask[] = (m & ~UInt64(1)) | (iszero(ctps_new.c[1]) ? UInt64(0) : UInt64(1))
    return ctps_new
end

function +(a::Number, ctps::CTPS{T}) where T
    return ctps + a
end

# - (range-limited)
function -(ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    if ctps1.desc !== ctps2.desc
        if ctps1.desc.nv != ctps2.desc.nv || ctps1.desc.order != ctps2.desc.order
            error("Cannot subtract CTPS with different descriptors: (nv=$(ctps1.desc.nv), order=$(ctps1.desc.order)) vs (nv=$(ctps2.desc.nv), order=$(ctps2.desc.order))")
        end
    end
    c = Vector{T}(undef, ctps1.desc.N)
    result = CTPS{T}(c, Ref(UInt64(0)))
    sub!(result, ctps1, ctps2)
    return result
end

function -(ctps::CTPS{T}, a::Number) where T
    ctps_new = CTPS(ctps)
    m = ctps.degree_mask[]
    c0 = (m & UInt64(1) != 0) ? ctps.c[1] : zero(T)
    ctps_new.c[1] = c0 - T(a)
    ctps_new.degree_mask[] = (m & ~UInt64(1)) | (iszero(ctps_new.c[1]) ? UInt64(0) : UInt64(1))
    return ctps_new
end

function -(a::Number, ctps::CTPS{T}) where T
    mask = ctps.degree_mask[]
    desc = ctps.desc
    c = Vector{T}(undef, desc.N)   # lazy: only active range written
    if mask != 0
        (s, e) = active_range_bounds(desc, mask)
        @inbounds @simd for i in s:e
            c[i] = -ctps.c[i]
        end
    end
    c0 = (mask & UInt64(1) != 0) ? -ctps.c[1] : zero(T)
    c[1] = c0 + T(a)
    m_out = (mask & ~UInt64(1)) | (iszero(c[1]) ? UInt64(0) : UInt64(1))
    return CTPS{T}(c, Ref(m_out))
end

function -(ctps::CTPS{T}) where T
    ctps_new = CTPS(ctps)   # range-limited copy
    mask = ctps.degree_mask[]
    if mask != 0
        (s, e) = active_range_bounds(ctps.desc, mask)
        @inbounds @simd for i in s:e
            ctps_new.c[i] = -ctps_new.c[i]
        end
    end
    ctps_new.degree_mask[] = mask
    return ctps_new
end

# * (allocating wrapper — delegates to mul! to avoid code duplication)
function *(ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    # Check descriptor compatibility
    if ctps1.desc !== ctps2.desc
        if ctps1.desc.nv != ctps2.desc.nv || ctps1.desc.order != ctps2.desc.order
            error("Cannot multiply CTPS with different descriptors: (nv=$(ctps1.desc.nv), order=$(ctps1.desc.order)) vs (nv=$(ctps2.desc.nv), order=$(ctps2.desc.order))")
        end
    end
    result = _ctps_zero(T, ctps1.desc)
    mul!(result, ctps1, ctps2)
    return result
end

function *(ctps::CTPS{T}, a::Number) where T
    ctps_new = CTPS(ctps)   # range-limited copy
    scale!(ctps_new, T(a))  # range-limited scale
    return ctps_new
end

function *(a::Number, ctps::CTPS{T}) where T
    return ctps * a
end

# /
function inv(ctps::CTPS{T}) where T
    if cst(ctps) == zero(T)
        error("Divide by zero in CTPS")
    end
    desc  = ctps.desc
    c0    = cst(ctps)
    inv_c0 = one(T) / c0

    temp = CTPS(ctps)
    temp.c[1] -= c0
    temp.degree_mask[] &= ~UInt64(1)

    neg_temp_over_c0 = CTPS(temp)
    scale!(neg_temp_over_c0, -inv_c0)

    term      = _ctps_constant(inv_c0, desc)
    term_next = _ctps_zero(T, desc)      # pre-allocated; swapped each iteration
    sum       = _ctps_constant(inv_c0, desc)   # heap-allocated (returned)

    for i in 1:desc.order
        mul!(term_next, term, neg_temp_over_c0)
        term, term_next = term_next, term  # swap bindings — zero-cost, no copy
        addto!(sum, term)
    end
    return sum
end

function /(ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    if cst(ctps2) == zero(T)
        error("Divide by zero in CTPS")
    end
    return ctps1 * inv(ctps2)
end

function /(ctps::CTPS{T}, a::T) where T
    if a == zero(T)
        error("Divide by zero in CTPS")
    end
    ctps_new = CTPS(ctps)     # range-limited copy
    scale!(ctps_new, one(T)/a) # range-limited scale
    return ctps_new
end

function /(a::T, ctps::CTPS{T}) where T
    if cst(ctps) == zero(T)
        error("Divide by zero in CTPS")
    end
    return a * inv(ctps)
end

# exponential (zero loop allocations)
# Heap-only (no pool borrows) so that Enzyme.jl can differentiate through this
# function without hitting "constant memory written with active data" errors.
# Performance-critical code should use exp!(result, ctps) instead.
function exp(ctps::CTPS{T}) where T
    a0   = cst(ctps)
    desc = ctps.desc
    # Fast path: polynomial is identically zero → exp(0) = 1 exactly.
    # NOTE: do NOT shortcut on `a0 == 0` alone — there may be non-zero higher terms.
    ctps.degree_mask[] == 0 && return _ctps_constant(one(T), desc)

    temp = CTPS(ctps)          # heap copy
    temp.c[1] -= a0
    temp.degree_mask[] &= ~UInt64(1)

    term      = _ctps_constant(one(T), desc)
    term_next = _ctps_zero(T, desc)      # pre-allocated; swapped each iteration
    sum       = _ctps_constant(one(T), desc)

    for i in 1:desc.order
        mul!(term_next, term, temp)
        term, term_next = term_next, term  # swap bindings — zero-cost, no copy
        _add_scaled!(sum, term, T(1.0 / factorial(i)))
    end
    scale!(sum, T(Base.exp(a0)))
    return sum
end

function exp!(result::CTPS{T}, ctps::CTPS{T}) where T
    a0 = cst(ctps)
    desc = ctps.desc
    # Fast path: polynomial is identically zero → exp(0) = 1 exactly.
    # NOTE: do NOT shortcut on `a0 == 0` alone — there may be non-zero higher terms.
    if ctps.degree_mask[] == 0
        _zero_active!(result)
        result.c[1] = one(T)
        result.degree_mask[] = UInt64(1)
        return result
    end

    _zero_active!(result)
    result.c[1] = one(T)
    result.degree_mask[] = UInt64(1)

    (idx_temp, temp)      = _ctps_pooled_copy(ctps, desc)
    temp.c[1] -= a0
    temp.degree_mask[] &= ~UInt64(1)

    (idx_term, term)      = _ctps_pooled(T, desc)
    term.c[1] = one(T)
    term.degree_mask[] = UInt64(1)

    (idx_tn,   term_next) = _ctps_pooled(T, desc)

    for i in 1:desc.order
        mul!(term_next, term, temp)
        copy!(term, term_next)
        _add_scaled!(result, term, T(1.0 / factorial(i)))
    end

    scale!(result, T(Base.exp(a0)))
    _pool_release!(idx_temp, temp,      desc)
    _pool_release!(idx_term, term,      desc)
    _pool_release!(idx_tn,   term_next, desc)
    return result
end

# logarithm (zero loop allocations)
function log(ctps::CTPS{T}) where T
    a0 = cst(ctps)
    a0 == zero(T) && error("Log of zero in CTPS")
    desc   = ctps.desc
    inv_a0 = one(T) / a0

    temp = CTPS(ctps)
    temp.c[1] -= a0
    temp.degree_mask[] &= ~UInt64(1)

    term = CTPS(temp)
    scale!(term, inv_a0)

    neg_temp_over_a0 = CTPS(temp)
    scale!(neg_temp_over_a0, -inv_a0)

    sum       = CTPS(term)
    term_next = _ctps_zero(T, desc)      # pre-allocated; swapped each iteration

    for i in 2:desc.order
        mul!(term_next, term, neg_temp_over_a0)
        term, term_next = term_next, term  # swap bindings — zero-cost, no copy
        _add_scaled!(sum, term, T(1.0 / i))
    end
    sum.c[1] = Base.log(a0)
    sum.degree_mask[] |= UInt64(1)
    return sum
end

function log!(result::CTPS{T}, ctps::CTPS{T}) where T
    a0 = cst(ctps)
    if a0 == zero(T)
        error("Log of zero in CTPS")
    end
    desc   = ctps.desc
    inv_a0 = one(T) / a0

    (idx_temp, temp) = _ctps_pooled_copy(ctps, desc)
    temp.c[1] -= a0
    temp.degree_mask[] &= ~UInt64(1)

    (idx_term, term) = _ctps_pooled_copy(temp, desc)
    scale!(term, inv_a0)

    (idx_tn, term_next) = _ctps_pooled(T, desc)

    (idx_ntoa, neg_temp_over_a0) = _ctps_pooled_copy(temp, desc)
    scale!(neg_temp_over_a0, -inv_a0)

    copy!(result, term)  # result = term (first-order Taylor term)

    for i in 2:desc.order
        mul!(term_next, term, neg_temp_over_a0)
        copy!(term, term_next)
        _add_scaled!(result, term, T(1.0 / i))
    end

    result.c[1] = Base.log(a0)
    result.degree_mask[] |= UInt64(1)
    _pool_release!(idx_temp, temp,             desc)
    _pool_release!(idx_term, term,             desc)
    _pool_release!(idx_tn,   term_next,        desc)
    _pool_release!(idx_ntoa, neg_temp_over_a0, desc)
    return result
end

# square root (minimal allocations)
function sqrt(ctps::CTPS{T}) where T
    a0_val = cst(ctps)
    T <: Real && a0_val < zero(T) && error("Square root of negative number in CTPS")
    a0   = Base.sqrt(a0_val)
    desc = ctps.desc

    temp = CTPS(ctps)
    temp.c[1] -= a0_val
    temp.degree_mask[] &= ~UInt64(1)

    term = CTPS(temp)
    scale!(term, one(T) / a0)

    neg_temp_over_a0 = CTPS(temp)
    scale!(neg_temp_over_a0, -one(T) / a0_val)

    sum      = CTPS(term)
    scale!(sum, T(0.5))
    term_buf = _ctps_zero(T, desc)      # pre-allocated; swapped each iteration

    for i in 2:desc.order
        mul!(term_buf, term, neg_temp_over_a0)
        term, term_buf = term_buf, term  # swap bindings — zero-cost, no copy
        _add_scaled!(sum, term,
            T(doublefactorial(2 * i - 3)) / T(doublefactorial(2 * i)))
    end
    sum.c[1] = a0
    sum.degree_mask[] |= UInt64(1)
    return sum
end

function sqrt!(result::CTPS{T}, ctps::CTPS{T}) where T
    a0_val = cst(ctps)
    if T <: Real && a0_val < zero(T)
        error("Square root of negative number in CTPS")
    end
    a0   = Base.sqrt(a0_val)
    desc = ctps.desc

    (idx_temp, temp) = _ctps_pooled_copy(ctps, desc)
    temp.c[1] -= a0_val
    temp.degree_mask[] &= ~UInt64(1)

    (idx_term, term) = _ctps_pooled_copy(temp, desc)
    scale!(term, one(T) / a0)

    (idx_tn,   temp_mul) = _ctps_pooled(T, desc)

    (idx_ntoa, neg_temp_over_a0) = _ctps_pooled_copy(temp, desc)
    scale!(neg_temp_over_a0, -one(T) / a0_val)

    copy!(result, term)
    scale!(result, T(0.5))

    for i in 2:desc.order
        mul!(temp_mul, term, neg_temp_over_a0)
        copy!(term, temp_mul)
        _add_scaled!(result, term,
            T(doublefactorial(2 * i - 3)) / T(doublefactorial(2 * i)))
    end

    result.c[1] = a0
    result.degree_mask[] |= UInt64(1)
    _pool_release!(idx_temp, temp,             desc)
    _pool_release!(idx_term, term,             desc)
    _pool_release!(idx_tn,   temp_mul,         desc)
    _pool_release!(idx_ntoa, neg_temp_over_a0, desc)
    return result
end

# power
function pow(ctps::CTPS{T}, b::Int) where T
    desc = ctps.desc
    b == 1 && return CTPS(ctps)
    b == 0 && return _ctps_constant(one(T), desc)
    b < 0  && return inv(pow(ctps, -b))

    # Fast paths for common small exponents (1 or 2 mul! calls, minimal allocs)
    if b == 2
        r = _ctps_zero(T, desc)
        mul!(r, ctps, ctps)
        return r
    end
    if b == 3
        (idx_t, t) = _ctps_pooled(T, desc)
        r = _ctps_zero(T, desc)
        mul!(t, ctps, ctps)
        mul!(r, t, ctps)
        _pool_release!(idx_t, t, desc)
        return r
    end
    if b == 4
        (idx_t, t) = _ctps_pooled(T, desc)
        r = _ctps_zero(T, desc)
        mul!(t, ctps, ctps)
        mul!(r, t, t)
        _pool_release!(idx_t, t, desc)
        return r
    end

    # General: binary exponentiation — O(log b) mul! calls for all c₀
    (idx_base, base) = _ctps_pooled_copy(ctps, desc)
    (idx_buf,  buf)  = _ctps_pooled(T, desc)
    acc = _ctps_constant(one(T), desc)   # heap-allocated (returned)
    n = b
    while n > 0
        if n & 1 == 1
            mul!(buf, acc, base)
            copy!(acc, buf)
        end
        n >>= 1
        if n > 0
            mul!(buf, base, base)
            copy!(base, buf)
        end
    end
    _pool_release!(idx_base, base, desc)
    _pool_release!(idx_buf,  buf,  desc)
    return acc
end

function ^(ctps::CTPS{T}, b::Int) where T
    return pow(ctps, b)
end

# In-place power: result = ctps^b  (b ≥ 0; uses pool for temporaries)
function pow!(result::CTPS{T}, ctps::CTPS{T}, b::Int) where T
    desc = ctps.desc
    if b == 0
        _zero_active!(result)
        result.c[1] = one(T)
        result.degree_mask[] = UInt64(1)
        return result
    end
    b == 1 && (copy!(result, ctps); return result)
    b < 0  && error("pow!(result, ctps, b) with b < 0 not supported; use inv(pow(ctps,-b))")
    if b == 2
        mul!(result, ctps, ctps)
        return result
    end
    if b == 3
        (idx_t, t) = _ctps_pooled(T, desc)
        mul!(t, ctps, ctps)
        mul!(result, t, ctps)
        _pool_release!(idx_t, t, desc)
        return result
    end
    if b == 4
        (idx_t, t) = _ctps_pooled(T, desc)
        mul!(t, ctps, ctps)
        mul!(result, t, t)
        _pool_release!(idx_t, t, desc)
        return result
    end
    # General: binary exponentiation — identity starts in result, base in pool
    (idx_base, base) = _ctps_pooled_copy(ctps, desc)
    (idx_buf,  buf)  = _ctps_pooled(T, desc)
    _zero_active!(result)
    result.c[1] = one(T)
    result.degree_mask[] = UInt64(1)  # result = 1
    n = b
    while n > 0
        if n & 1 == 1
            mul!(buf, result, base)
            copy!(result, buf)
        end
        n >>= 1
        if n > 0
            mul!(buf, base, base)
            copy!(base, buf)
        end
    end
    _pool_release!(idx_base, base, desc)
    _pool_release!(idx_buf,  buf,  desc)
    return result
end

# sin (pool-backed temporaries)
function sin(ctps::CTPS{T}) where T
    a0     = cst(ctps)
    sin_a0 = Base.sin(a0)
    cos_a0 = Base.cos(a0)
    desc   = ctps.desc

    temp = CTPS(ctps)
    temp.c[1] -= a0
    temp.degree_mask[] &= ~UInt64(1)

    term      = _ctps_constant(one(T), desc)
    term_next = _ctps_zero(T, desc)      # pre-allocated; swapped each iteration
    sum       = _ctps_zero(T, desc)

    is_odd = true
    for i in 1:desc.order
        mul!(term_next, term, temp)
        term, term_next = term_next, term  # swap bindings — zero-cost, no copy
        coeff = is_odd ?
            cos_a0 * T((-1)^((i-1)÷2)) / T(factorial(i)) :
            sin_a0 * T((-1)^(i÷2))     / T(factorial(i))
        _add_scaled!(sum, term, coeff)
        is_odd = !is_odd
    end
    sum.c[1] = sin_a0
    sum.degree_mask[] |= UInt64(1)
    return sum
end

function sin!(result::CTPS{T}, ctps::CTPS{T}) where T
    a0 = cst(ctps)
    sin_a0 = Base.sin(a0)
    cos_a0 = Base.cos(a0)
    desc = ctps.desc

    _zero_active!(result)   # O(active_range) reset; no-op for fresh workspace slots

    (idx_temp, temp)      = _ctps_pooled_copy(ctps, desc)
    temp.c[1] -= a0
    temp.degree_mask[] &= ~UInt64(1)

    (idx_term, term)      = _ctps_pooled(T, desc)
    term.c[1] = one(T)
    term.degree_mask[] = UInt64(1)

    (idx_tn,   term_next) = _ctps_pooled(T, desc)

    is_odd = true
    for i in 1:desc.order
        mul!(term_next, term, temp)
        copy!(term, term_next)
        coeff = if is_odd
            cos_a0 * T((-1) ^ ((i - 1) ÷ 2)) / T(factorial(i))
        else
            sin_a0 * T((-1) ^ (i ÷ 2)) / T(factorial(i))
        end
        _add_scaled!(result, term, coeff)
        is_odd = !is_odd
    end

    result.c[1] = sin_a0   # explicit = since degree-0 was not touched by the loop
    result.degree_mask[] |= UInt64(1)
    _pool_release!(idx_temp, temp,      desc)
    _pool_release!(idx_term, term,      desc)
    _pool_release!(idx_tn,   term_next, desc)
    return result
end

function cos(ctps::CTPS{T}) where T
    a0     = cst(ctps)
    sin_a0 = Base.sin(a0)
    cos_a0 = Base.cos(a0)
    desc   = ctps.desc

    temp = CTPS(ctps)
    temp.c[1] -= a0
    temp.degree_mask[] &= ~UInt64(1)

    term      = _ctps_constant(one(T), desc)
    term_next = _ctps_zero(T, desc)      # pre-allocated; swapped each iteration
    sum       = _ctps_zero(T, desc)

    is_odd = true
    for i in 1:desc.order
        mul!(term_next, term, temp)
        term, term_next = term_next, term  # swap bindings — zero-cost, no copy
        coeff = is_odd ?
            sin_a0 * T((-1)^((i+1)÷2)) / T(factorial(i)) :
            cos_a0 * T((-1)^(i÷2))     / T(factorial(i))
        _add_scaled!(sum, term, coeff)
        is_odd = !is_odd
    end

    sum.c[1] = cos_a0
    sum.degree_mask[] |= UInt64(1)
    return sum
end

function cos!(result::CTPS{T}, ctps::CTPS{T}) where T
    a0 = cst(ctps)
    sin_a0 = Base.sin(a0)
    cos_a0 = Base.cos(a0)
    desc = ctps.desc

    _zero_active!(result)

    (idx_temp, temp)      = _ctps_pooled_copy(ctps, desc)
    temp.c[1] -= a0
    temp.degree_mask[] &= ~UInt64(1)

    (idx_term, term)      = _ctps_pooled(T, desc)
    term.c[1] = one(T)
    term.degree_mask[] = UInt64(1)

    (idx_tn,   term_next) = _ctps_pooled(T, desc)

    is_odd = true
    for i in 1:desc.order
        mul!(term_next, term, temp)
        copy!(term, term_next)
        coeff = if is_odd
            sin_a0 * T((-1) ^ ((i + 1) ÷ 2)) / T(factorial(i))
        else
            cos_a0 * T((-1) ^ (i ÷ 2)) / T(factorial(i))
        end
        _add_scaled!(result, term, coeff)
        is_odd = !is_odd
    end

    result.c[1] = cos_a0
    result.degree_mask[] |= UInt64(1)
    _pool_release!(idx_temp, temp,      desc)
    _pool_release!(idx_term, term,      desc)
    _pool_release!(idx_tn,   term_next, desc)
    return result
end

# arcsin
function asin(ctps::CTPS{T}) where T
    temp = CTPS(ctps)
    a0 = cst(ctps)
    arcsin_a0 = asin(a0)
    cos_y0 = sqrt(one(T) - a0 ^ 2)
    temp = temp - a0
    temp1 = CTPS(temp)
    # Newton's method on TPSA doubles order-of-accuracy each step;
    # ⌈log₂(order+2)⌉ + 2 iterations are sufficient (vs the previous order+10).
    niters = max(3, ceil(Int, log2(ctps.desc.order + 2)))
    for i in 1:niters
        temp1 = temp1 - sin(temp1) + (temp + a0*(1.0-cos(temp1)))/cos_y0
    end
    return temp1 + arcsin_a0
end

# arccos
function acos(ctps::CTPS{T}) where T
    return T(pi / 2) - asin(ctps)
end

# tangent: shares one power-series loop for both sin and cos
function tan(ctps::CTPS{T}) where T
    a0 = cst(ctps)
    sin_a0 = Base.sin(a0)
    cos_a0 = Base.cos(a0)
    desc = ctps.desc

    temp = CTPS(ctps)
    temp.c[1] -= a0
    temp.degree_mask[] &= ~UInt64(1)

    term      = _ctps_constant(one(T), desc)
    term_next = _ctps_zero(T, desc)      # pre-allocated; swapped each iteration
    sin_sum   = _ctps_zero(T, desc)
    cos_sum   = _ctps_zero(T, desc)

    is_odd = true
    for i in 1:desc.order
        mul!(term_next, term, temp)
        term, term_next = term_next, term  # swap bindings — zero-cost, no copy
        inv_fac   = T(1.0 / factorial(i))
        sin_coeff = is_odd ? cos_a0 * T((-1) ^ ((i - 1) ÷ 2)) * inv_fac :
                             sin_a0 * T((-1) ^ (i ÷ 2))        * inv_fac
        cos_coeff = is_odd ? sin_a0 * T((-1) ^ ((i + 1) ÷ 2)) * inv_fac :
                             cos_a0 * T((-1) ^ (i ÷ 2))        * inv_fac
        _add_scaled!(sin_sum, term, sin_coeff)
        _add_scaled!(cos_sum, term, cos_coeff)
        is_odd = !is_odd
    end

    sin_sum.c[1] += sin_a0
    sin_sum.degree_mask[] |= UInt64(1)
    cos_sum.c[1] += cos_a0
    cos_sum.degree_mask[] |= UInt64(1)
    return sin_sum / cos_sum
end

# hyperbolic sin
function sinh(ctps::CTPS{T}) where T
    a0 = cst(ctps)
    sinh_a0 = Base.sinh(a0)
    cosh_a0 = Base.cosh(a0)
    desc = ctps.desc

    temp = CTPS(ctps)
    temp.c[1] -= a0
    temp.degree_mask[] &= ~UInt64(1)

    term      = _ctps_constant(one(T), desc)
    term_next = _ctps_zero(T, desc)      # pre-allocated; swapped each iteration
    sum       = _ctps_zero(T, desc)      # heap-allocated (returned)

    is_odd = true
    for i in 1:desc.order
        mul!(term_next, term, temp)
        term, term_next = term_next, term  # swap bindings — zero-cost, no copy
        coeff = is_odd ? cosh_a0 / T(factorial(i)) : sinh_a0 / T(factorial(i))
        _add_scaled!(sum, term, coeff)
        is_odd = !is_odd
    end

    sum.c[1] = sinh_a0
    sum.degree_mask[] |= UInt64(1)
    return sum
end

function sinh!(result::CTPS{T}, ctps::CTPS{T}) where T
    a0 = cst(ctps)
    sinh_a0 = Base.sinh(a0)
    cosh_a0 = Base.cosh(a0)
    desc = ctps.desc

    _zero_active!(result)

    (idx_temp, temp)      = _ctps_pooled_copy(ctps, desc)
    temp.c[1] -= a0
    temp.degree_mask[] &= ~UInt64(1)

    (idx_term, term)      = _ctps_pooled(T, desc)
    term.c[1] = one(T)
    term.degree_mask[] = UInt64(1)

    (idx_tn,   term_next) = _ctps_pooled(T, desc)

    is_odd = true
    for i in 1:desc.order
        mul!(term_next, term, temp)
        copy!(term, term_next)
        coeff = is_odd ? cosh_a0 / T(factorial(i)) : sinh_a0 / T(factorial(i))
        _add_scaled!(result, term, coeff)
        is_odd = !is_odd
    end

    result.c[1] = sinh_a0
    result.degree_mask[] |= UInt64(1)
    _pool_release!(idx_temp, temp,      desc)
    _pool_release!(idx_term, term,      desc)
    _pool_release!(idx_tn,   term_next, desc)
    return result
end

# hyperbolic cos
function cosh(ctps::CTPS{T}) where T
    a0 = cst(ctps)
    sinh_a0 = Base.sinh(a0)
    cosh_a0 = Base.cosh(a0)
    desc = ctps.desc

    temp = CTPS(ctps)
    temp.c[1] -= a0
    temp.degree_mask[] &= ~UInt64(1)

    term      = _ctps_constant(one(T), desc)
    term_next = _ctps_zero(T, desc)      # pre-allocated; swapped each iteration
    sum       = _ctps_zero(T, desc)      # heap-allocated (returned)

    is_odd = true
    for i in 1:desc.order
        mul!(term_next, term, temp)
        term, term_next = term_next, term  # swap bindings — zero-cost, no copy
        coeff = is_odd ? sinh_a0 / T(factorial(i)) : cosh_a0 / T(factorial(i))
        _add_scaled!(sum, term, coeff)
        is_odd = !is_odd
    end

    sum.c[1] = cosh_a0
    sum.degree_mask[] |= UInt64(1)
    return sum
end

function cosh!(result::CTPS{T}, ctps::CTPS{T}) where T
    a0 = cst(ctps)
    sinh_a0 = Base.sinh(a0)
    cosh_a0 = Base.cosh(a0)
    desc = ctps.desc

    _zero_active!(result)

    (idx_temp, temp)      = _ctps_pooled_copy(ctps, desc)
    temp.c[1] -= a0
    temp.degree_mask[] &= ~UInt64(1)

    (idx_term, term)      = _ctps_pooled(T, desc)
    term.c[1] = one(T)
    term.degree_mask[] = UInt64(1)

    (idx_tn,   term_next) = _ctps_pooled(T, desc)

    is_odd = true
    for i in 1:desc.order
        mul!(term_next, term, temp)
        copy!(term, term_next)
        coeff = is_odd ? sinh_a0 / T(factorial(i)) : cosh_a0 / T(factorial(i))
        _add_scaled!(result, term, coeff)
        is_odd = !is_odd
    end

    result.c[1] = cosh_a0
    result.degree_mask[] |= UInt64(1)
    _pool_release!(idx_temp, temp,      desc)
    _pool_release!(idx_term, term,      desc)
    _pool_release!(idx_tn,   term_next, desc)
    return result
end
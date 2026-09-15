
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

# Composition plan — precomputed parent-monomial tree for compose!/compose.
#
# For each monomial index i (2..N), the image under a substitution map g is:
#   mono_image[i] = g[par_var[i]] * mono_image[par_idx[i]]
# so every monomial image costs exactly one mul! call.
# The root mono_image[1] = CTPS(1.0) (the constant-1 monomial).
#
# Parent choice: first variable (lowest index) with a non-zero exponent in α.
# This yields a balanced tree and guarantees that par_idx[i] < i (deglex order),
# so a single forward pass through [2..N] builds all images correctly.
#
# n_children[i] = number of j > i with par_idx[j] == i.
# Available for traversal planning; ordinary composition uses depth-first reuse.
struct CompPlan
    par_idx    :: Vector{Int32}   # parent monomial index (1-based); element 1 unused
    par_var    :: Vector{Int8}    # which variable to multiply (1-based); element 1 unused
    n_children :: Vector{Int32}   # number of monomials j with par_idx[j] == i
    first_child::Vector{Int32}    # descriptor-owned, read-only traversal links
    next_sibling::Vector{Int32}
end

# Thread-local pool of pre-allocated Float64 coefficient buffers.
# Eliminates the dominant zeros(N) allocation inside math function temporaries.
# Each thread owns one pool per descriptor; acquire/release are lock-free.
const CTPS_POOL_SIZE = 32

# Upper bound on the index-table footprint of one descriptor. The multiplication
# schedules hold about binomial(2nv + order, order)/2 Int32 entries, so very high
# orders are practical only for a few variables (PSDesc(4, 63) would need ~22 GB).
# Raise deliberately when the memory is really available:
#     PolySeries.MAX_DESCRIPTOR_BYTES[] = 8 * 1024^3
const MAX_DESCRIPTOR_BYTES = Ref{Int}(2 * 1024^3)

# Bytes needed by the index tables and one thread's coefficient pool for a
# descriptor with per-degree sizes `Nd`. Float64 arithmetic: the schedule count
# can exceed typemax(Int) long before the limit check rejects it.
function descriptor_footprint_bytes(nv::Int, order::Int, N::Int, Nd::Vector{Int})
    sched_entries = 0.0
    for di in 0:order, dj in 0:min(di, order - di)
        sched_entries += Float64(Nd[di + 1]) * Float64(Nd[dj + 1])
    end
    return 4 * sched_entries +                    # MulSchedule2D k_local (Int32)
           sizeof(Int) * (order + 1) +           # degree-row schedule offsets
           Float64(N) * (nv + 1) +                # PolyMap exponent table (UInt8)
           Float64(N) * 48 +                      # exp_to_idx Dict and CompPlan
           Float64(N) * 8 * CTPS_POOL_SIZE        # one thread's Float64 pool
end

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
            pool.ctps[i] = CTPS{Float64}(pool.bufs[i], desc, pool.refs[i])
        end
    end
end

# Descriptor fields are const: their bindings cannot be reassigned. The boxed
# representation gives stable identity and avoids Enzyme issues with returning
# large immutable metadata structs. Index tables are read-only after construction.
mutable struct PSDesc
    const id::Int                       # stable context identity
    const nv::Int                       # number of variables
    const order::Int                    # maximum order
    const N::Int                        # total number of coefficients
    const Nd::Vector{Int}               # size per degree
    const off::Vector{Int}              # start offset per degree (1-based)
    const polymap::PolyMap              # index mapping (index → exponent)
    const exp_to_idx::Dict              # reverse map: SVector{nv+1,UInt8} → Int (concrete per instance)
    const mul::Vector{MulSchedule2D}    # 2D k-map multiplication schedules indexed as (di,dj)
    const mul_offsets::Vector{Int}     # first schedule for each di; dj is the offset
    const comp_plan::CompPlan           # composition build plan
    const _pools::Vector{DescPool}      # per-thread coefficient buffer pools (Float64 only)
end

# Publish a new immutable snapshot only when a descriptor is created. Readers
# need no cache lock and never race with a resize of the vector they are using.
mutable struct DescriptorRegistry
    @atomic entries::Vector{PSDesc}
end
# Intentionally a typed, non-const binding: Enzyme may capture the contents of
# a const global registry when compiling a derivative, including its current
# entries snapshot. A runtime global load keeps later descriptor IDs visible.
# The registry object itself is never reassigned; entries are published atomically.
DESCRIPTOR_REGISTRY::DescriptorRegistry = DescriptorRegistry(PSDesc[])

@inline function _descriptor_by_id(id::Int)
    registry = DESCRIPTOR_REGISTRY
    entries = @atomic :acquire registry.entries
    return entries[id]
end

# Thread-safe cache for PSDesc instances
const DESC_CACHE = Dict{Tuple{Int,Int}, PSDesc}()

# Defaults are local to a task and are consulted only by convenience constructors.
const DESCRIPTOR_TLS_KEY = gensym(:PolySeries_descriptor)

"""
    set_descriptor!(nv::Int, order::Int)

Set this task's default descriptor for new CTPS objects. Existing polynomials
retain their descriptors. Initialize the default in each task that needs it,
or pass a descriptor explicitly to CTPS constructors.

# Arguments
- `nv::Int`: Number of variables (1–127)
- `order::Int`: Maximum order (0–63)

Descriptors with more than `typemax(Int32)` coefficients are rejected before
allocation. Invalid requests leave the current default unchanged.

# Example
```julia
using PolySeries
set_descriptor!(3, 4)  # 3 variables, order 4
x = CTPS(0.0, 1)       # Create variable x
y = CTPS(0.0, 2)       # Create variable y
```
"""
function set_descriptor!(nv::Int, order::Int)
    desc = PSDesc(nv, order)
    task_local_storage(DESCRIPTOR_TLS_KEY, desc)
    return desc
end

"""
    get_descriptor()

Get this task's default descriptor. Throws an error if not set.
"""
function get_descriptor()
    desc = get(task_local_storage(), DESCRIPTOR_TLS_KEY, nothing)
    desc === nothing && error("No descriptor set for this task. Call set_descriptor!(nv, order) first or pass a PSDesc to CTPS.")
    return desc::PSDesc
end

"""
    clear_descriptor!()

Clear this task's default descriptor. Existing polynomials remain usable.
"""
function clear_descriptor!()
    delete!(task_local_storage(), DESCRIPTOR_TLS_KEY)
    return nothing
end
const DESC_CACHE_LOCK = ReentrantLock()

# Constructor with caching (thread-safe)
function PSDesc(nv::Int, order::Int)
    1 <= nv <= typemax(Int8) ||
        throw(ArgumentError("Number of variables must be between 1 and $(typemax(Int8)) (composition variable indices use Int8)"))
    0 <= order <= 63 ||
        throw(ArgumentError("Polynomial order must be between 0 and 63 (degree masks use UInt64)"))
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
        # Check the schedule/index representation before allocating any tables.
        # BigInt is used only on cache misses, so even enormous requests fail
        # with a useful error instead of overflowing an intermediate Int.
        count = binomial(big(nv + order), order)
        count <= typemax(Int32) ||
            throw(ArgumentError("Descriptor exceeds the $(typemax(Int32))-coefficient limit of Int32 indices"))
        N = Int(count)
        
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

        # Reject descriptors whose index tables would not fit before touching
        # any large allocation; the Int32 check above only bounds N.
        footprint = descriptor_footprint_bytes(nv, order, N, Nd)
        footprint <= MAX_DESCRIPTOR_BYTES[] || throw(ArgumentError(
            "PSDesc($nv, $order) needs about $(Base.format_bytes(round(Int, footprint))) of index " *
            "tables, above the $(Base.format_bytes(MAX_DESCRIPTOR_BYTES[])) limit in " *
            "PolySeries.MAX_DESCRIPTOR_BYTES[]. Reduce the order or raise the limit deliberately."))

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
        # Each di row contains dj=0:min(di, order-di), in ascending order.
        mul = MulSchedule2D[]
        mul_offsets = Vector{Int}(undef, order + 1)
        for di in 0:order
            mul_offsets[di + 1] = length(mul) + 1
            for dj in 0:min(di, order - di)   # dj ≤ di  AND  dk = di+dj ≤ order
                sched = build_mul_schedule_2d(polymap, exp_to_idx, nv, di, dj, off, Nd)
                push!(mul, sched)
            end
        end
        
        # Build composition plan (parent-monomial tree for compose!/compose)
        comp_plan = build_comp_plan(polymap, exp_to_idx, nv, N)

        # Pre-allocate per-thread coefficient buffer pools (Float64 only)
        pools = [DescPool(N) for _ in 1:Threads.nthreads()]

        entries = @atomic :acquire DESCRIPTOR_REGISTRY.entries
        desc = PSDesc(length(entries) + 1, nv, order, N, Nd, off, polymap, exp_to_idx, mul, mul_offsets, comp_plan, pools)
        _init_desc_pools!(pools, desc)   # phase-2: populate CTPS wrappers now that desc exists
        @atomic :release DESCRIPTOR_REGISTRY.entries = [entries; desc]
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

# Build the composition plan: for each monomial index i (2..N) find the
# lowest-index variable v with αᵥ > 0, record par_idx[i] (the parent monomial
# after decrementing αᵥ) and par_var[i] = v.
#
# The polymap stores ALL variable exponents: map[i, 1] = total degree,
# map[i, v+1] = αᵥ for v = 1..nv  (last column = αₙᵥ, NOT always 0).
function build_comp_plan(polymap::PolyMap, exp_to_idx::Dict, nv::Int, N::Int)
    K          = nv + 1
    par_idx    = zeros(Int32, N)
    par_var    = zeros(Int8,  N)
    n_children = zeros(Int32, N)

    @inbounds for i in 2:N
        # Find first variable v (1-based) with αᵥ > 0
        v_found = 0
        for v in 1:nv
            if polymap.map[i, v + 1] > 0
                v_found = v
                break
            end
        end
        # v_found ≥ 1 is guaranteed: total degree ≥ 1 for all i ≥ 2

        # Build parent exponent key: decrement total (col 1) and αᵥ (col v+1)
        par_exp = SVector{K, UInt8}(
            col == 1           ? UInt8(polymap.map[i, 1]           - 1) :
            col == v_found + 1 ? UInt8(polymap.map[i, v_found + 1] - 1) :
            polymap.map[i, col]
            for col in 1:K)

        pi = Int(exp_to_idx[par_exp])
        par_idx[i]    = Int32(pi)
        par_var[i]    = Int8(v_found)
        n_children[pi] += Int32(1)
    end

    first_child = zeros(Int32,N)
    next_sibling = zeros(Int32,N)
    for i in N:-1:2
        parent = Int(par_idx[i])
        next_sibling[i] = first_child[parent]
        first_child[parent] = Int32(i)
    end
    return CompPlan(par_idx, par_var, n_children, first_child, next_sibling)
end

# A permanent context ID fixes each polynomial's descriptor independently of
# defaults. Numeric metadata avoids Enzyme's mixed-activity errors when constant
# descriptor references are stored in differentiable CTPS objects.
struct CTPS{T}
    c           :: Vector{T}                # coefficients, length desc.N
    descriptor_id :: Int
    degree_mask :: Base.RefValue{UInt64}    # bit i set iff degree-i block is active

    function CTPS{T}(c::Vector{T}, desc::PSDesc, mask::Base.RefValue{UInt64}) where T
        length(c) == desc.N || throw(DimensionMismatch("Coefficient buffer length must match descriptor size $(desc.N)"))
        new{T}(c, desc.id, mask)
    end
end

@inline Base.getproperty(ctps::CTPS, s::Symbol) =
    s === :desc ? _descriptor_by_id(getfield(ctps, :descriptor_id)) : getfield(ctps, s)
Base.propertynames(::CTPS, private::Bool=false) =
    private ? (:c, :desc, :degree_mask, :descriptor_id) : (:c, :desc, :degree_mask)

# Preserve the old raw-buffer constructor, capturing the default once.
CTPS{T}(c::Vector{T}, mask::Base.RefValue{UInt64}) where T =
    CTPS{T}(c, get_descriptor(), mask)

@inline function _check_descriptors(a::CTPS, b::CTPS)
    a.descriptor_id == b.descriptor_id ||
        throw(DimensionMismatch("CTPS operands must have the same number of variables and order"))
    return nothing
end

# A numerical zero can have a nonzero derivative. During Enzyme AD, retain
# initialized zero coefficients and execute their arithmetic; only structural
# zeros (inactive degree blocks) may be skipped. Outside AD this helper reduces
# to iszero, preserving the sparse fast paths and lazy-storage invariant.
@inline _prunable_zero(x) = !within_autodiff() && iszero(x)

# Compute degree mask from fully initialized coefficients.
function compute_degree_mask(c::Vector{T}, desc::PSDesc) where T
    mask = UInt64(0)
    order = desc.order
    @inbounds for d in 0:order
        d_start = desc.off[d + 1]
        d_end = d_start + desc.Nd[d + 1] - 1
        for i in d_start:d_end
            if !_prunable_zero(c[i])
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
    # Shifting mask2 by each active degree of mask1 sets every reachable sum;
    # bits above `order` fall off the end or are cut by the final mask. This is
    # popcount(mask1) shifts instead of an (order+1)(order+2)/2 double loop,
    # which mattered once the schedule scan itself had been removed from mul!.
    result = UInt64(0)
    m1 = mask1
    while m1 != 0
        di = trailing_zeros(m1)
        m1 &= m1 - UInt64(1)
        result |= mask2 << di
    end
    return result & (typemax(UInt64) >> (63 - order))
end

# Fast internal constructors that reuse an existing PSDesc (no lock acquisition).
@inline function _ctps_constant(a::T, desc::PSDesc) where T
    c = within_autodiff() ? zeros(T, desc.N) : Vector{T}(undef, desc.N)
    c[1] = a
    mask = _prunable_zero(a) ? UInt64(0) : UInt64(1)
    return CTPS{T}(c, desc, Ref(mask))
end

@inline function _ctps_zero(::Type{T}, desc::PSDesc) where T
    c = within_autodiff() ? zeros(T, desc.N) : Vector{T}(undef, desc.N)
    return CTPS{T}(c, desc, Ref(UInt64(0)))
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
    # Differentiation needs typed, independently owned primal/shadow storage.
    # The pool's cached objects are metadata and must not enter Enzyme's tape.
    within_autodiff() && return (UInt8(0), _ctps_zero(Float64, desc))
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
    within_autodiff() && return (UInt8(0), CTPS(src))
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
                dst_buf = pool.bufs[idx]
                src_buf = src.c
                for (s, e) in active_ranges(desc, tm)
                    @inbounds @simd for i in s:e; dst_buf[i] = src_buf[i]; end
                end
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


"""
    CTPS(T::Type, nv::Int, order::Int)
    CTPS(a::Number, nv::Int, order::Int)
    CTPS(a::Number, n::Int, nv::Int, order::Int)

Legacy convenience constructors. Each one first calls
[`set_descriptor!`](@ref)`(nv, order)`, replacing this task's default
descriptor as a side effect, and then builds the zero polynomial, the constant
`a`, or the variable `a + δxₙ` exactly like the explicit-descriptor forms.
Prefer `CTPS(…, desc::PSDesc)` in new code. Invalid variable indices and
`order == 0` variables raise `ArgumentError`.
"""
CTPS(T::Type, nv::Int, order::Int) = CTPS(T, set_descriptor!(nv, order))
CTPS(a::T, nv::Int, order::Int) where {T<:Number} = CTPS(a, set_descriptor!(nv, order))
CTPS(a::T, n::Int, nv::Int, order::Int) where {T<:Number} = CTPS(a, n, set_descriptor!(nv, order))

# -----------------------------------------------------------------------
# RANGE-LIMITED HELPERS
#
# `active_range_bounds(desc, mask)` returns the (start, stop) 1-based indices
# of the coefficient slice that covers all non-zero degrees in `mask`.
# Operating only on this range avoids touching zero pages for sparse CTPS.
# This bounding interval is safe for clearing discarded storage. Coefficient
# reads and arithmetic must use active_ranges to respect inactive gaps.
# -----------------------------------------------------------------------
@inline function active_range_bounds(desc::PSDesc, mask::UInt64)
    mask == 0 && return (1, 0)   # empty range
    min_deg = trailing_zeros(mask) % Int
    max_deg = (63 - leading_zeros(mask)) % Int
    start = desc.off[min_deg + 1]
    stop  = desc.off[max_deg + 1] + desc.Nd[max_deg + 1] - 1
    return (start, stop)
end

# Iterate maximal runs of active degrees without allocating. Dense masks take
# one coefficient loop, just like active_range_bounds; gaps take separate loops
# so inactive (possibly uninitialized) coefficients are never read or overwritten.
struct ActiveRanges
    desc::PSDesc
    mask::UInt64
end

@inline active_ranges(desc::PSDesc, mask::UInt64) = ActiveRanges(desc, mask)
Base.IteratorSize(::Type{ActiveRanges}) = Base.SizeUnknown()
Base.eltype(::Type{ActiveRanges}) = Tuple{Int,Int}

@inline function Base.iterate(ranges::ActiveRanges, mask::UInt64=ranges.mask)
    iszero(mask) && return nothing
    first_degree = trailing_zeros(mask)
    # Adding the lowest set bit carries through the first run of ones. UInt64
    # wraparound gives zero for a run ending at bit 63; trailing_zeros(0) is 64.
    after_run = mask + (UInt64(1) << first_degree)
    past_degree = trailing_zeros(after_run)
    desc = ranges.desc
    start = desc.off[first_degree + 1]
    stop = desc.off[past_degree] + desc.Nd[past_degree] - 1
    remaining = mask & after_run
    return ((start, stop), remaining)
end

# Copy constructor — ordinarily copies only active coefficient runs. During AD,
# it materializes every coefficient through `_coefficient` so structural zeros
# retain tangent paths without reading uninitialized inactive storage.
function CTPS(M::CTPS{T}) where T
    desc = M.desc
    if within_autodiff()
        # Materialize structural zeros through the differentiable accessor.
        # Reading M.c directly would touch uninitialized inactive storage;
        # copying only active blocks would discard tangents at zero values.
        c = Vector{T}(undef, desc.N)
        mask = M.degree_mask[]
        for degree in 0:desc.order
            s = desc.off[degree + 1]
            e = s + desc.Nd[degree + 1] - 1
            @inbounds for i in s:e
                c[i] = _coefficient(M.c, mask, i, degree)
            end
        end
        full_mask = typemax(UInt64) >> (63 - desc.order)
        return CTPS{T}(c, desc, Ref(full_mask))
    end
    c    = Vector{T}(undef, desc.N)   # lazy: only active range is written
    mask = M.degree_mask[]
    if mask != 0
        for (s, e) in active_ranges(desc, mask)
            @inbounds @simd for i in s:e
                c[i] = M.c[i]
            end
        end
    end
    return CTPS{T}(c, desc, Ref(mask))
end

# Arithmetic must not treat a prebuilt input's sparsity as derivative activity.
# A dense AD copy uses the accessor rules to preserve coefficient sensitivities
# without reading inactive storage. Already dense inputs need no extra copy.
@inline function _ad_input(p::CTPS)
    if within_autodiff() && p.degree_mask[] != typemax(UInt64) >> (63 - p.desc.order)
        return CTPS(p)
    end
    return p
end

# Convenience constructors capture this task's default at construction time.
CTPS(T::Type) = CTPS(T, get_descriptor())
CTPS(a::Number) = CTPS(a, get_descriptor())
CTPS(a::Number, n::Int) = CTPS(a, n, get_descriptor())

"""    CTPS(T::Type, desc::PSDesc)

Create a zero polynomial with an explicit descriptor, without changing defaults.
"""
function CTPS(::Type{T}, desc::PSDesc) where T
    return CTPS{T}(zeros(T, desc.N), desc, Ref(UInt64(0)))
end

"""    CTPS(a::Number, desc::PSDesc)

Create a constant polynomial with an explicit descriptor.
"""
function CTPS(a::T, desc::PSDesc) where T<:Number
    c = zeros(T, desc.N)
    c[1] = a
    mask = _prunable_zero(a) ? UInt64(0) : UInt64(1)
    return CTPS{T}(c, desc, Ref(mask))
end

"""    CTPS(a::Number, n::Int, desc::PSDesc)

Create the variable `a + δxₙ` with an explicit descriptor.
"""
function CTPS(a::T, n::Int, desc::PSDesc) where T<:Number
    1 <= n <= desc.nv || throw(ArgumentError("Variable index must be between 1 and $(desc.nv)"))
    desc.order >= 1 || throw(ArgumentError("A variable requires order >= 1"))
    c = zeros(T, desc.N)
    c[1] = a
    c[n + 1] = one(T)
    mask = _prunable_zero(a) ? UInt64(2) : UInt64(3)
    return CTPS{T}(c, desc, Ref(mask))
end

# ========== End simplified constructors ==========

# Coefficient storage is lazy: a degree block whose bit is clear in
# `degree_mask` is mathematically zero but its memory is uninitialized, so every
# read must go through the mask. `cst` feeds the expansion point of sqrt, inv,
# sin, asin, ... — reading `c[1]` directly returned garbage whenever the constant
# term was exactly zero (e.g. `asin(px / d2)` about the reference orbit).
# During AD, constructors/arithmetic retain initialized numerical zeros in the
# mask; a clear bit still denotes a structural zero with no coefficient to read.
# A shared read boundary lets differentiation rules use coefficient activity
# independently of the primal's lazy-storage mask.
@inline function _coefficient(coeffs::Vector{T}, mask::UInt64, index::Int, degree::Int) where T
    return (mask & (UInt64(1) << degree)) != 0 ? coeffs[index] : zero(T)
end

@inline cst(ctps::CTPS) = _coefficient(ctps.c, ctps.degree_mask[], 1, 0)

function findindex(ctps::CTPS, indexmap::Vector{Int})
    dim = ctps.desc.nv
    n = length(indexmap)
    (n == dim || n == dim + 1) ||
        throw(ArgumentError("Exponent vector must have length $dim or $(dim + 1), got $n"))

    prefixed = n == dim + 1
    degree = prefixed ? indexmap[1] : 0
    if prefixed
        0 <= degree <= ctps.desc.order ||
            throw(ArgumentError("Total degree must be between 0 and $(ctps.desc.order), got $degree"))
    end

    first_exponent = prefixed ? 2 : 1
    exponent_sum = 0
    @inbounds for j in first_exponent:n
        exponent = indexmap[j]
        exponent >= 0 || throw(ArgumentError("Exponents must be nonnegative, got $exponent at position $j"))
        limit = prefixed ? degree : ctps.desc.order
        exponent <= limit - exponent_sum ||
            throw(ArgumentError(prefixed ?
                "Degree prefix $degree does not equal the sum of the exponents" :
                "Total degree exceeds descriptor order $(ctps.desc.order)"))
        exponent_sum += exponent
    end
    if prefixed
        exponent_sum == degree ||
            throw(ArgumentError("Degree prefix $degree does not equal the sum of the exponents ($exponent_sum)"))
    else
        degree = exponent_sum
    end

    # Rank the validated exponent vector in the descriptor's graded ordering.
    cumsum_val = degree
    result = Int(1)
    for i in dim:-1:1
        cumsum_val == 0 && break
        result += binomial(cumsum_val - 1 + i, i)
        if i > 1
            cumsum_val -= indexmap[dim - i + first_exponent]
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
# function assign!(ctps::CTPS{T}, a::T, n_var::Int) where T
#     if n_var <= ctps.desc.nv && n_var > 0
#         ctps.c[n_var + 1] = one(T)
#         ctps.c[1] = a
#         return nothing
#     else
#         error("Variable index out of range in CTPS")
#     end
# end

# function assign!(ctps::CTPS{T}, a::T) where T
#     ctps.c[1] = a
#     return nothing
# end

# function reassign!(ctps::CTPS{T}, a::T, n_var::Int) where T
#     if n_var <= ctps.desc.nv && n_var > 0
#         fill!(ctps.c, zero(T))
#         ctps.c[n_var + 1] = one(T)
#         ctps.c[1] = a
#         return nothing
#     else
#         error("Variable index out of range in CTPS")
#     end
# end



@inline function element(ctps::CTPS{T}, ind::Vector{Int}) where T
    result = findindex(ctps, ind)
    degree = Int(ctps.desc.polymap.map[result, 1])
    return _coefficient(ctps.c, ctps.degree_mask[], result, degree)
end

# Defining callable instance — evaluate the CTPS polynomial at numerical args.
#
# Efficiency improvements over a naive loop:
#   • degree_mask limits the active coefficient range (skips trailing zero blocks)
#   • per-term zero check avoids nv exponent lookups for zero coefficients
#   • descriptor lookup hoisted outside the loop
function (ctps::CTPS{T})(args::T...) where T
    desc = ctps.desc::PSDesc   # hoisted: avoids repeated Union{PSDesc,Nothing} access
    nv   = desc.nv
    length(args) == nv || throw(ArgumentError(
        "expected $nv evaluation arguments (one per variable), got $(length(args))"))

    source_mask = ctps.degree_mask[]
    mask = within_autodiff() ? typemax(UInt64) >> (63 - desc.order) : source_mask
    mask == 0 && return zero(T)   # identically-zero polynomial fast path

    pm         = desc.polymap.map
    return_value = cst(ctps)

    for (lo, hi) in active_ranges(desc, mask & ~UInt64(1))
        @inbounds for i in lo:hi
            val = within_autodiff() ? _coefficient(ctps.c, source_mask, i, Int(pm[i, 1])) : ctps.c[i]
            _prunable_zero(val) && continue       # skip zero coefficients early
            for v in 1:nv
                e = Int(pm[i, v + 1])     # Enzyme prefers Int
                e == 0 && continue
                val *= args[v]^e
            end
            return_value += val
        end
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

# Colors cycled per degree order when the IO supports color.
# Degree 0 (constant) is printed in normal/default color; degrees 1+ alternate.
const _DEGREE_COLORS = (:normal, :cyan, :green, :yellow, :magenta, :light_blue, :light_red)

@inline function _cprint(io::IO, use_color::Bool, color::Symbol, args...)
    use_color ? printstyled(io, args...; color) : print(io, args...)
end

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
    ctps1 = _ad_input(ctps1)
    ctps2 = _ad_input(ctps2)
    _check_descriptors(result, ctps1)
    _check_descriptors(ctps1, ctps2)
    m1 = ctps1.degree_mask[]; m2 = ctps2.degree_mask[]
    m_out = m1 | m2
    if m_out != 0
        only1 = m1 & ~m2    # degrees active in ctps1 only  → copy from ctps1
        only2 = m2 & ~m1    # degrees active in ctps2 only  → copy from ctps2
        both  = m1  &  m2   # degrees active in both        → add
        if only1 != 0
            for (s, e) in active_ranges(ctps1.desc, only1)
                @inbounds @simd for i in s:e; result.c[i] = ctps1.c[i]; end
            end
        end
        if only2 != 0
            for (s, e) in active_ranges(ctps1.desc, only2)
                @inbounds @simd for i in s:e; result.c[i] = ctps2.c[i]; end
            end
        end
        if both != 0
            for (s, e) in active_ranges(ctps1.desc, both)
                @inbounds @simd for i in s:e; result.c[i] = ctps1.c[i] + ctps2.c[i]; end
            end
        end
    end
    result.degree_mask[] = m_out
    return nothing
end

function add!(result::CTPS{T}, ctps1::CTPS{T}, a::T) where T
    ctps1 = _ad_input(ctps1)
    _check_descriptors(result, ctps1)
    m1 = ctps1.degree_mask[]
    if m1 != 0
        for (s, e) in active_ranges(ctps1.desc, m1)
            @inbounds @simd for i in s:e
                result.c[i] = ctps1.c[i]
            end
        end
    end
    # c[1] is valid only if bit-0 is in m1 (lazy-zero: otherwise garbage)
    c0 = (m1 & UInt64(1) != 0) ? ctps1.c[1] : zero(T)
    result.c[1] = c0 + a
    result.degree_mask[] = (m1 & ~UInt64(1)) | (_prunable_zero(result.c[1]) ? UInt64(0) : UInt64(1))
    return nothing
end

function addto!(ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    within_autodiff() && return add!(ctps1, ctps1, ctps2)
    _check_descriptors(ctps1, ctps2)
    m2 = ctps2.degree_mask[]
    m2 == 0 && return nothing
    m1 = ctps1.degree_mask[]
    new_bits   = m2 & ~m1  # degrees only in ctps2 → first write (=)
    accum_bits = m2 &  m1  # degrees in both        → accumulate (+=)
    if new_bits != 0
        for (s, e) in active_ranges(ctps1.desc, new_bits)
            @inbounds @simd for i in s:e; ctps1.c[i] = ctps2.c[i]; end
        end
    end
    if accum_bits != 0
        for (s, e) in active_ranges(ctps1.desc, accum_bits)
            @inbounds @simd for i in s:e; ctps1.c[i] += ctps2.c[i]; end
        end
    end
    ctps1.degree_mask[] |= m2
    return nothing
end

function sub!(result::CTPS{T}, ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    ctps1 = _ad_input(ctps1)
    ctps2 = _ad_input(ctps2)
    _check_descriptors(result, ctps1)
    _check_descriptors(ctps1, ctps2)
    m1 = ctps1.degree_mask[]; m2 = ctps2.degree_mask[]
    m_out = m1 | m2
    if m_out != 0
        only1 = m1 & ~m2    # active only in ctps1 → copy from ctps1
        only2 = m2 & ~m1    # active only in ctps2 → negate from ctps2
        both  = m1  &  m2   # active in both       → subtract
        if only1 != 0
            for (s, e) in active_ranges(ctps1.desc, only1)
                @inbounds @simd for i in s:e; result.c[i] = ctps1.c[i]; end
            end
        end
        if only2 != 0
            for (s, e) in active_ranges(ctps1.desc, only2)
                @inbounds @simd for i in s:e; result.c[i] = -ctps2.c[i]; end
            end
        end
        if both != 0
            for (s, e) in active_ranges(ctps1.desc, both)
                @inbounds @simd for i in s:e; result.c[i] = ctps1.c[i] - ctps2.c[i]; end
            end
        end
    end
    result.degree_mask[] = m_out
    return nothing
end

function subfrom!(ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    within_autodiff() && return sub!(ctps1, ctps1, ctps2)
    _check_descriptors(ctps1, ctps2)
    m2 = ctps2.degree_mask[]
    m2 == 0 && return nothing
    m1 = ctps1.degree_mask[]
    new_bits   = m2 & ~m1  # degrees only in ctps2 → first write (= -ctps2)
    accum_bits = m2 &  m1  # degrees in both        → subtract (-=)
    if new_bits != 0
        for (s, e) in active_ranges(ctps1.desc, new_bits)
            @inbounds @simd for i in s:e; ctps1.c[i] = -ctps2.c[i]; end
        end
    end
    if accum_bits != 0
        for (s, e) in active_ranges(ctps1.desc, accum_bits)
            @inbounds @simd for i in s:e; ctps1.c[i] -= ctps2.c[i]; end
        end
    end
    ctps1.degree_mask[] |= m2
    return nothing
end

function scale!(ctps::CTPS{T}, a::T) where T
    within_autodiff() && return scale!(ctps, ctps, a)
    mask = ctps.degree_mask[]
    if mask != 0
        for (s, e) in active_ranges(ctps.desc, mask)
            @inbounds @simd for i in s:e
                ctps.c[i] *= a
            end
        end
    end
    return nothing
end

# 3-arg scale: dest = src * a  (range-limited copy + multiply)
function scale!(dest::CTPS{T}, src::CTPS{T}, a::T) where T
    src = _ad_input(src)
    _check_descriptors(dest, src)
    sm = src.degree_mask[]
    dm = dest.degree_mask[]
    # Zero out degrees in dest that src doesn't cover
    extra = dm & ~sm
    if extra != 0
        for (s, e) in active_ranges(dest.desc, extra)
            @inbounds @simd for i in s:e; dest.c[i] = zero(T); end
        end
    end
    if sm != 0
        for (s, e) in active_ranges(src.desc, sm)
            @inbounds @simd for i in s:e
                dest.c[i] = src.c[i] * a
            end
        end
    end
    dest.degree_mask[] = _prunable_zero(a) ? UInt64(0) : sm
    return nothing
end

# result = a * c1 + b * c2  (single range-limited pass, zero allocations)
# The canonical in-place form of the linear combination `a*c1 + b*c2 → result`.
# Used in the rotation step: nx1 = cos_μ*x1 + sin_μ*pmx.
function scaleadd!(result::CTPS{T}, a::T, c1::CTPS{T}, b::T, c2::CTPS{T}) where T
    c1 = _ad_input(c1)
    c2 = _ad_input(c2)
    _check_descriptors(result, c1)
    _check_descriptors(c1, c2)
    m1   = c1.degree_mask[]
    m2   = c2.degree_mask[]
    ma   = _prunable_zero(a) ? UInt64(0) : m1   # effective mask for a*c1
    mb   = _prunable_zero(b) ? UInt64(0) : m2   # effective mask for b*c2
    mout = ma | mb
    dm   = result.degree_mask[]
    # Zero degrees present in dest but not in the output
    extra = dm & ~mout
    if extra != 0
        for (s, e) in active_ranges(result.desc, extra)
            @inbounds @simd for i in s:e; result.c[i] = zero(T); end
        end
    end
    # Split mout into sub-ranges to avoid reading garbage from inactive source
    onlya   = ma & ~mb
    onlyb   = mb & ~ma
    both_ab = ma  &  mb
    if onlya != 0
        for (s, e) in active_ranges(result.desc, onlya)
            @inbounds @simd for i in s:e; result.c[i] = a * c1.c[i]; end
        end
    end
    if onlyb != 0
        for (s, e) in active_ranges(result.desc, onlyb)
            @inbounds @simd for i in s:e; result.c[i] = b * c2.c[i]; end
        end
    end
    if both_ab != 0
        for (s, e) in active_ranges(result.desc, both_ab)
            @inbounds @simd for i in s:e; result.c[i] = a * c1.c[i] + b * c2.c[i]; end
        end
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
#   • Slots default to Float64; pass T as the third constructor argument for
#     a workspace of CTPS{T} objects (e.g. ComplexF64 or Float32).
#   • `borrow!` returns a CTPS{T} backed by a pre-allocated buffer.
#   • `release!` zeros only the active degree range — O(active_range), not O(N).
#   • The workspace is NOT thread-safe: create one per concurrent task or protect access.
mutable struct PSWorkspace{T}
    desc     :: PSDesc
    bufs     :: Vector{CTPS{T}}         # pre-allocated CTPS objects
    avail    :: Vector{Int}             # stack of available indices
    sp       :: Int                     # stack pointer (sp==length(bufs) → all free)
    id_to_idx :: Dict{UInt64, Int}      # objectid(ctps.c) → slot index, O(1) release
    inuse    :: BitVector               # slot currently borrowed (rejects double release)
end

function PSWorkspace(desc::PSDesc, n::Int = 32, ::Type{T} = Float64) where T
    n >= 1 || throw(ArgumentError("PSWorkspace needs at least one slot"))
    bufs  = [CTPS{T}(zeros(T, desc.N), desc, Ref(UInt64(0)))
             for _ in 1:n]
    avail = collect(1:n)
    id_to_idx = Dict{UInt64, Int}(objectid(bufs[i].c) => i for i in 1:n)
    return PSWorkspace(desc, bufs, avail, n, id_to_idx, falses(n))
end

PSWorkspace{T}(desc::PSDesc, n::Int = 32) where T = PSWorkspace(desc, n, T)

"""    borrow!(ws::PSWorkspace{T}) -> CTPS{T}

Obtain a zero CTPS from the workspace without heap allocation.
Must be paired with `release!(ws, ctps)` when done."""
@inline function borrow!(ws::PSWorkspace)
    ws.sp == 0 && error("PSWorkspace exhausted — increase n at construction")
    idx  = ws.avail[ws.sp]
    ws.sp -= 1
    @inbounds ws.inuse[idx] = true
    return ws.bufs[idx]
end

"""    release!(ws::PSWorkspace{T}, ctps::CTPS{T})

Return a borrowed CTPS slot to the workspace.
Zeros only the active degree range before returning — O(active_range)."""
@inline function release!(ws::PSWorkspace{T}, ctps::CTPS{T}) where T
    ctps.desc === ws.desc || throw(DimensionMismatch("CTPS and workspace descriptors must match"))
    idx = get(ws.id_to_idx, objectid(ctps.c), 0)
    idx != 0 || throw(ArgumentError("CTPS was not borrowed from this workspace"))
    @inbounds ws.inuse[idx] || throw(ArgumentError("CTPS slot was already released to this workspace"))
    @inbounds ws.inuse[idx] = false
    dm = ctps.degree_mask[]
    if dm != 0
        (s, e) = active_range_bounds(ws.desc, dm)
        buf = ctps.c
        @inbounds @simd for i in s:e; buf[i] = zero(T); end
        ctps.degree_mask[] = UInt64(0)
    end
    ws.sp += 1
    ws.avail[ws.sp] = idx
    return nothing
end

function copy!(dest::CTPS{T}, src::CTPS{T}) where T
    src = _ad_input(src)
    _check_descriptors(dest, src)
    # Zero out any degrees in dest that src doesn't have, then copy active range
    src_mask  = src.degree_mask[]
    dest_mask = dest.degree_mask[]
    extra_mask = dest_mask & ~src_mask
    if extra_mask != 0
        for (s, e) in active_ranges(dest.desc, extra_mask)
            @inbounds @simd for i in s:e
                dest.c[i] = zero(T)
            end
        end
    end
    if src_mask != 0
        for (s, e) in active_ranges(src.desc, src_mask)
            @inbounds @simd for i in s:e
                dest.c[i] = src.c[i]
            end
        end
    end
    dest.degree_mask[] = src_mask
    return nothing
end

function zero!(ctps::CTPS{T}) where T
    _zero_active!(ctps)
    return nothing
end

# Zero only the currently-active degree range, then clear the mask.
# O(active_range) instead of O(N).  Used by in-place math functions to reset
# a workspace slot before writing — free for already-zero workspace slots
# (degree_mask == 0 → no-op).
# As in pool/workspace release, the entire value is discarded: clearing gaps
# in the bounding interval is safe and avoids unnecessary per-run overhead.
@inline function _zero_active!(ctps::CTPS{T}) where T
    if within_autodiff()
        fill!(ctps.c, zero(T))
        ctps.degree_mask[] = UInt64(0)
        return nothing
    end
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
    within_autodiff() && return scaleadd!(sum, one(T), sum, scale, term)
    _check_descriptors(sum, term)
    tm = term.degree_mask[]
    (_prunable_zero(scale) || tm == 0) && return
    sm = sum.degree_mask[]
    desc = sum.desc
    # Initialize new blocks and accumulate shared blocks independently. A
    # bounding interval could erase an existing block between two new blocks.
    for (s, e) in active_ranges(desc, tm & ~sm)
        @inbounds @simd for j in s:e; sum.c[j] = term.c[j] * scale; end
    end
    for (s, e) in active_ranges(desc, tm & sm)
        @inbounds @simd for j in s:e; sum.c[j] += term.c[j] * scale; end
    end
    sum.degree_mask[] = sm | tm
    return nothing
end

# Sparse traversal keeps the descriptor's ascending (di,dj) order, so the
# arithmetic accumulates in exactly the same order as the dense traversal.
struct ActiveMulSchedules
    desc::PSDesc
    mask1::UInt64
    mask2::UInt64
end

# Equal masks covering degrees 0:k select a contiguous prefix of the table.
# Iterate it directly, without a view allocation or per-pair bit operations.
struct PrefixMulSchedules
    schedules::Vector{MulSchedule2D}
    stop::Int
end
Base.eltype(::Type{PrefixMulSchedules}) = MulSchedule2D
Base.length(it::PrefixMulSchedules) = it.stop
@inline function Base.iterate(it::PrefixMulSchedules, index::Int=1)
    index > it.stop && return nothing
    @inbounds return it.schedules[index], index + 1
end
Base.IteratorSize(::Type{ActiveMulSchedules}) = Base.SizeUnknown()
Base.eltype(::Type{ActiveMulSchedules}) = MulSchedule2D

@inline function Base.iterate(it::ActiveMulSchedules,
                             state=(it.mask1 | it.mask2, UInt64(0), 0))
    remaining_di, remaining_dj, di = state
    while remaining_dj == 0
        remaining_di == 0 && return nothing
        di = trailing_zeros(remaining_di)
        remaining_di &= remaining_di - UInt64(1)
        # Include either orientation, but visit the symmetric schedule once.
        partners = ((it.mask1 >> di) & 1 != 0 ? it.mask2 : UInt64(0)) |
                   ((it.mask2 >> di) & 1 != 0 ? it.mask1 : UInt64(0))
        limit = min(di, it.desc.order - di)
        remaining_dj = partners & (typemax(UInt64) >> (63 - limit))
    end
    dj = trailing_zeros(remaining_dj)
    remaining_dj &= remaining_dj - UInt64(1)
    @inbounds sched = it.desc.mul[it.desc.mul_offsets[di + 1] + dj]
    return sched, (remaining_di, remaining_dj, di)
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
    ctps1 = _ad_input(ctps1)
    ctps2 = _ad_input(ctps2)
    _check_descriptors(result, ctps1)
    _check_descriptors(ctps1, ctps2)
    aliases1 = result.c === ctps1.c
    aliases2 = result.c === ctps2.c
    if aliases1 || aliases2
        idx1 = UInt8(0)
        idx2 = UInt8(0)
        input1 = ctps1
        input2 = ctps2
        try
            if aliases1
                idx1, input1 = _ctps_pooled_copy(ctps1, ctps1.desc)
            end
            if aliases2
                if ctps2.c === ctps1.c && ctps2.degree_mask === ctps1.degree_mask
                    input2 = input1
                else
                    idx2, input2 = _ctps_pooled_copy(ctps2, ctps2.desc)
                end
            end
            return mul!(result, input1, input2)
        finally
            aliases1 && _pool_release!(idx1, input1, ctps1.desc)
            idx2 != 0 && _pool_release!(idx2, input2, ctps2.desc)
        end
    end
    desc  = ctps1.desc
    order = desc.order
    c1    = ctps1.c
    c2    = ctps2.c
    cr    = result.c
    mask1 = ctps1.degree_mask[]
    mask2 = ctps2.degree_mask[]

    if within_autodiff() && T <: Union{Float32,Float64}
        # _ad_input materializes all coefficient directions. Pool management,
        # alias handling, and masks stay outside the convolution AD boundary.
        _dense_product!(cr, c1, c2, desc)
        result.degree_mask[] = typemax(UInt64) >> (63-order)
        return result
    end

    # Range-limited zero fill for the output degree band.
    if mask1 != 0 && mask2 != 0
        dk_min = (trailing_zeros(mask1) + trailing_zeros(mask2)) % Int
        dk_max = min((63 - leading_zeros(mask1)) % Int +
                     (63 - leading_zeros(mask2)) % Int, order)
        if dk_min > dk_max
            result.degree_mask[] = UInt64(0)
            return result
        end
        out_s = desc.off[dk_min + 1]
        out_e = desc.off[dk_max + 1] + desc.Nd[dk_max + 1] - 1
        @inbounds @simd for i in out_s:out_e
            cr[i] = zero(T)
        end

        # `Val(true)` asserts that every visited schedule is active in both
        # orientations, so the kernel may skip its per-schedule mask tests.
        # That holds for the whole table when both masks are full (which
        # _ad_input guarantees during Enzyme AD) and for a prefix of the table
        # when both masks are the same contiguous run of degrees from zero.
        full_mask = typemax(UInt64) >> (63 - order)
        square = !within_autodiff() && desc.N >= 128 && c1 === c2 && mask1 == mask2 && mask1 > UInt64(3)
        if within_autodiff() || (mask1 == full_mask && mask2 == full_mask)
            _mul_full!(cr, c1, c2, desc, square)
        elseif mask1 == mask2 && (mask1 & (mask1 + UInt64(1))) == 0
            last_degree = 63 - leading_zeros(mask1)
            stop = desc.mul_offsets[last_degree + 1] + min(last_degree, order - last_degree)
            if square
                _square_schedules!(cr, c1, PrefixMulSchedules(desc.mul, stop), Val(0))
            else
                _mul_schedules!(cr, c1, c2, mask1, mask2, PrefixMulSchedules(desc.mul, stop), Val(true))
            end
        elseif mask1 & ~UInt64(3) == 0 && mask2 & ~UInt64(3) != 0
            _affine_product_add!(cr, c1, mask1, c2, mask2, desc)
        elseif mask2 & ~UInt64(3) == 0 && mask1 & ~UInt64(3) != 0
            _affine_product_add!(cr, c2, mask2, c1, mask1, desc)
        elseif mask1 & ~UInt64(7) == 0 && mask2 & ~UInt64(7) != 0 &&
               _few_coefficients(c1, mask1, desc)
            _sparse_product_add!(cr, c1, mask1, c2, mask2, desc)
        elseif mask2 & ~UInt64(7) == 0 && mask1 & ~UInt64(7) != 0 &&
               _few_coefficients(c2, mask2, desc)
            _sparse_product_add!(cr, c2, mask2, c1, mask1, desc)
        elseif square
            _square_schedules!(cr, c1, ActiveMulSchedules(desc, mask1, mask2), Val(0))
        else
            _mul_schedules!(cr, c1, c2, mask1, mask2,
                            ActiveMulSchedules(desc, mask1, mask2), Val(false))
        end
    end

    result.degree_mask[] = compose_degree_mask(mask1, mask2, order)
    return result
end

@inline function _few_coefficients(c, mask, desc)
    count = 0
    for (s,e) in active_ranges(desc, mask)
        @inbounds for i in s:e
            count += !iszero(c[i])
            count > 4 && return false
        end
    end
    return true
end

function _sparse_product_add!(cr::Vector{T}, a::Vector{T}, am::UInt64,
                              b::Vector{T}, bm::UInt64, desc::PSDesc) where T
    adegrees = am
    @inbounds while adegrees != 0
        ad = trailing_zeros(adegrees)
        adegrees &= adegrees-UInt64(1)
        abase = desc.off[ad+1]-1
        for ai in 1:desc.Nd[ad+1]
            av = a[abase+ai]
            iszero(av) && continue
            bdegrees = bm & (typemax(UInt64) >> (63-(desc.order-ad)))
            while bdegrees != 0
                bd = trailing_zeros(bdegrees)
                bdegrees &= bdegrees-UInt64(1)
                bbase = desc.off[bd+1]-1
                if ad >= bd
                    sched = desc.mul[desc.mul_offsets[ad+1]+bd]
                    for bi in 1:desc.Nd[bd+1]
                        bv = b[bbase+bi]
                        iszero(bv) || (cr[sched.k_local[bi,ai]] += av*bv)
                    end
                else
                    sched = desc.mul[desc.mul_offsets[bd+1]+ad]
                    for bi in 1:desc.Nd[bd+1]
                        bv = b[bbase+bi]
                        iszero(bv) || (cr[sched.k_local[ai,bi]] += av*bv)
                    end
                end
            end
        end
    end
    return nothing
end

# Full coefficient convolution is the AD boundary. Its arguments contain no
# pool state or numeric sparsity decisions; aliases are handled by mul!.
function _dense_product!(cr::Vector{T}, a::Vector{T}, b::Vector{T}, desc::PSDesc) where T
    fill!(cr, zero(T))
    _mul_full!(cr, a, b, desc, a === b && desc.N >= 128)
    return nothing
end

# Multiplication by a constant plus a linear form. Put each nonzero linear
# coefficient outside the long loop, instead of multiplying the inner block's
# zero entries for every coefficient of the other polynomial.
function _affine_product_add!(cr::Vector{T}, a::Vector{T}, am::UInt64,
                              b::Vector{T}, bm::UInt64, desc::PSDesc) where T
    if am & UInt64(1) != 0
        a0 = a[1]
        for (s,e) in active_ranges(desc, bm)
            @inbounds @simd for i in s:e
                cr[i] += a0 * b[i]
            end
        end
    end
    am & UInt64(2) == 0 && return nothing
    @inbounds for variable in 1:desc.nv
        av = a[variable+1]
        iszero(av) && continue
        bm & UInt64(1) != 0 && (cr[variable+1] += av * b[1])
        degrees = bm & ~UInt64(1) & ~(UInt64(1) << desc.order)
        while degrees != 0
            d = trailing_zeros(degrees)
            degrees &= degrees - UInt64(1)
            sched = desc.mul[desc.mul_offsets[d+1] + 1]
            base = Int(sched.i_start) - 1
            for i in 1:Int(sched.Ni)
                cr[sched.k_local[variable,i]] += av * b[base+i]
            end
        end
    end
    return nothing
end

# `dense == true` promises that every schedule yielded by `schedules` is active
# in both orientations (see the dispatcher in mul!); the caller owns that
# promise. Outside AD a broken promise would read uninitialized storage.
function _mul_schedules!(cr::Vector{T}, c1::Vector{T}, c2::Vector{T},
                         mask1::UInt64, mask2::UInt64, schedules,
                         ::Val{dense}) where {T, dense}
    @inbounds for sched in schedules
        # desc.mul only contains (di ≥ dj) entries — no empty sentinels.
        di = UInt32(sched.di)
        dj = UInt32(sched.dj)
        has_fwd = dense || ((mask1 >> di) & UInt64(1) != 0 && (mask2 >> dj) & UInt64(1) != 0)
        has_rev = (di != dj) &&
                  (dense || ((mask1 >> dj) & UInt64(1) != 0 && (mask2 >> di) & UInt64(1) != 0))

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
                (_prunable_zero(ai) && _prunable_zero(bi)) && continue
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
                (_prunable_zero(ai) && _prunable_zero(bi)) && continue
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
                _prunable_zero(ai) && continue
                @inbounds @fastmath for j_local in 1:Nj
                    cr[k_mat[j_local, i_local]] +=
                        ai * c2[j_base + j_local]
                end
            end

        else  # has_rev only
            # ── Reverse only: c1[dj] * c2[di] ──────────────────────────
            @inbounds @fastmath for i_local in 1:Ni
                bi = c2[i_base + i_local]   # c2 in di-block
                _prunable_zero(bi) && continue
                @inbounds @fastmath for j_local in 1:Nj
                    cr[k_mat[j_local, i_local]] +=
                        c1[j_base + j_local] * bi
                end
            end
        end
    end
    return nothing
end

# ── CTPS Composition ─────────────────────────────────────────────────────────

"""
    CompositionWorkspace(desc::PSDesc, T::Type = Float64)

Reusable storage for `compose!(result, f, g, workspace)`. Ordinary evaluation
uses one coefficient vector per tree depth: `(desc.order + 1) * desc.N`
coefficients, plus O(N) tree metadata. Unused monomial branches are skipped.
A workspace must not be shared by concurrent calls. Enzyme differentiation
uses a separate retained-image implementation instead of this scratch storage.
"""
struct CompositionWorkspace{T}
    desc::PSDesc
    images::Vector{CTPS{T}}
    first_child::Vector{Int32}
    next_sibling::Vector{Int32}
    needed::BitVector
end

function CompositionWorkspace(desc::PSDesc, ::Type{T}=Float64) where T
    images = [_ctps_zero(T, desc) for _ in 0:desc.order]
    return CompositionWorkspace(desc, images, desc.comp_plan.first_child,
                                desc.comp_plan.next_sibling, falses(desc.N))
end

@inline function _check_composition(result, f, g)
    _check_descriptors(result, f)
    length(g) == f.desc.nv || throw(DimensionMismatch(
        "compose!: expected $(f.desc.nv) substitution polynomials, got $(length(g))"))
    # The traversal clears `result` before reading `f` and the images of `g`,
    # so sharing storage would silently produce zeros.
    result.c === f.c && throw(ArgumentError("compose!: result must not alias f"))
    for substitution in g
        _check_descriptors(f, substitution)
        result.c === substitution.c &&
            throw(ArgumentError("compose!: result must not alias a member of g"))
    end
    return nothing
end

# Visit children while retaining only the path from the root. A sibling can
# overwrite the previous child's image after its entire subtree is consumed.
function _compose_children!(result::CTPS{T}, f::CTPS{T}, g,
                            ws::CompositionWorkspace{T}, parent::Int, depth::Int) where T
    i = Int(ws.first_child[parent])
    while i != 0
        if ws.needed[i]
            img = ws.images[depth + 1]
            mul!(img, g[Int(ws.desc.comp_plan.par_var[i])], ws.images[depth])
            degree = Int(ws.desc.polymap.map[i, 1])
            if (f.degree_mask[] & (UInt64(1) << degree)) != 0
                coeff = f.c[i]
                iszero(coeff) || _add_scaled!(result, img, coeff)
            end
            _compose_children!(result, f, g, ws, i, depth + 1)
        end
        i = Int(ws.next_sibling[i])
    end
    return result
end

function compose!(result::CTPS{T}, f::CTPS{T}, g::AbstractVector{<:CTPS{T}}) where T
    if within_autodiff()
        return _compose_retained!(result, f, g)
    end
    _check_composition(result, f, g)
    _is_translation(g, f.desc) && return _translate!(result, f, g)
    return compose!(result, f, g, CompositionWorkspace(f.desc, T))
end

function compose!(result::CTPS{T}, f::CTPS{T}, g::AbstractVector{<:CTPS{T}},
                  ws::CompositionWorkspace{T}) where T
    if within_autodiff()
        return _compose_retained!(result, f, g)
    end
    _check_composition(result, f, g)
    ws.desc === f.desc || throw(DimensionMismatch("Composition workspace descriptor must match inputs"))
    _is_translation(g, f.desc) && return _translate!(result, f, g)
    fill!(ws.needed, false)
    # Read only active degrees; their inactive neighbours may contain poison
    # or uninitialized references. Mark ancestors in one reverse-index pass.
    for (s, e) in active_ranges(f.desc, f.degree_mask[])
        for i in s:e
            ws.needed[i] = !iszero(f.c[i])
        end
    end
    for i in f.desc.N:-1:2
        ws.needed[i] && (ws.needed[Int(f.desc.comp_plan.par_idx[i])] = true)
    end
    _zero_active!(result)
    ws.needed[1] || return result
    root = ws.images[1]
    root.c[1] = one(T)
    root.degree_mask[] = UInt64(1)
    if (f.degree_mask[] & UInt64(1)) != 0
        result.c[1] = f.c[1]
        iszero(f.c[1]) || (result.degree_mask[] = UInt64(1))
    end
    return _compose_children!(result, f, g, ws, 1, 1)
end

# Differentiation retains all images: reverse mode may need primal values
# after the forward traversal. Do not apply numeric coefficient pruning here,
# since a zero-valued coefficient can have a nonzero derivative.
function _compose_retained!(result::CTPS{T}, f::CTPS{T}, g::AbstractVector{<:CTPS{T}}) where T
    _check_composition(result, f, g)
    desc  = f.desc
    nv    = desc.nv
    N     = desc.N
    source_mask = f.degree_mask[]
    fm    = within_autodiff() ? typemax(UInt64) >> (63 - desc.order) : source_mask
    plan  = desc.comp_plan

    _zero_active!(result)
    fm == 0 && return result     # f is the zero polynomial

    # Constant term: result += f.c[1]
    if (fm & UInt64(1)) != 0 && !_prunable_zero(cst(f))
        result.c[1]          = cst(f)
        result.degree_mask[] |= UInt64(1)
    end

    par_idx  = plan.par_idx
    par_var  = plan.par_var

    # AD includes every degree: an inactive primal coefficient can still have
    # a tangent. Ordinary execution needs only the degrees present in f.
    max_deg_f = (63 - leading_zeros(fm)) % Int
    N_eff     = desc.off[max_deg_f + 1] + desc.Nd[max_deg_f + 1] - 1

    # Typed container — required for Enzyme (must not be Vector{Any}).
    # par_idx[i] < i (deglex ordering) guarantees mono_img[pi] is always
    # written before being read as a parent.
    mono_img = Vector{CTPS{T}}(undef, N_eff)
    mono_img[1] = _ctps_constant(one(T), desc)

    @inbounds for i in 2:N_eff
        pv  = Int(par_var[i])
        pi  = Int(par_idx[i])
        img = _ctps_zero(T, desc)
        mul!(img, g[pv], mono_img[pi])
        mono_img[i] = img

        # The accessor supplies zero for inactive primal storage while its AD
        # rule preserves coefficient tangents independently of the primal mask.
        d = Int(desc.polymap.map[i, 1])
        ((fm >> d) & UInt64(1)) == 0 && continue

        coeff = _coefficient(f.c, source_mask, i, d)
        _prunable_zero(coeff) || _add_scaled!(result, img, coeff)
    end

    return result
end

"""
    compose(f::CTPS{T}, g::AbstractVector{<:CTPS{T}}) -> CTPS{T}

Compose `f` with the substitution map `g`:

    h(x) = f(g[1](x), g[2](x), …, g[nv](x))

Uses a depth-first parent-monomial traversal, with at most one multiplication
per needed monomial and O(N * order) coefficient storage. Pass a reusable
`CompositionWorkspace` as the fourth argument to `compose!` to reuse storage.
Enzyme uses a retained-image path with O(N²) worst-case coefficient storage.
The result must not alias `f` or any member of `g`.

# Example
```julia
set_descriptor!(2, 3)
x = CTPS(0.0, 1);  y = CTPS(0.0, 2)
f  = x^2 + y              # f(x,y) = x² + y
g1 = x + y;  g2 = x - y   # substitution map g: (x,y) ↦ (x+y, x-y)
h  = compose(f, [g1, g2]) # h(x,y) = (x+y)² + (x-y) = x²+2xy+y²+x-y
```
"""
function compose(f::CTPS{T}, g::AbstractVector{<:CTPS{T}}) where T
    result = _ctps_zero(T, f.desc)
    compose!(result, f, g)
    return result
end

# + (range-limited: only touches active degree range)
function +(ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    _check_descriptors(ctps1, ctps2)
    desc = ctps1.desc
    c = Vector{T}(undef, desc.N)
    result = CTPS{T}(c, desc, Ref(UInt64(0)))
    add!(result, ctps1, ctps2)
    return result
end

function +(ctps::CTPS{T}, a::Number) where T
    ctps = _ad_input(ctps)
    ctps_new = CTPS(ctps)   # range-limited undef copy
    m = ctps.degree_mask[]
    c0 = (m & UInt64(1) != 0) ? ctps.c[1] : zero(T)
    ctps_new.c[1] = c0 + T(a)
    ctps_new.degree_mask[] = (m & ~UInt64(1)) | (_prunable_zero(ctps_new.c[1]) ? UInt64(0) : UInt64(1))
    return ctps_new
end

function +(a::Number, ctps::CTPS{T}) where T
    return ctps + a
end

# - (range-limited)
function -(ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    _check_descriptors(ctps1, ctps2)
    desc = ctps1.desc
    c = Vector{T}(undef, desc.N)
    result = CTPS{T}(c, desc, Ref(UInt64(0)))
    sub!(result, ctps1, ctps2)
    return result
end

function -(ctps::CTPS{T}, a::Number) where T
    ctps = _ad_input(ctps)
    ctps_new = CTPS(ctps)
    m = ctps.degree_mask[]
    c0 = (m & UInt64(1) != 0) ? ctps.c[1] : zero(T)
    ctps_new.c[1] = c0 - T(a)
    ctps_new.degree_mask[] = (m & ~UInt64(1)) | (_prunable_zero(ctps_new.c[1]) ? UInt64(0) : UInt64(1))
    return ctps_new
end

function -(a::Number, ctps::CTPS{T}) where T
    ctps = _ad_input(ctps)
    mask = ctps.degree_mask[]
    desc = ctps.desc
    c = Vector{T}(undef, desc.N)   # lazy: only active range written
    if mask != 0
        for (s, e) in active_ranges(desc, mask)
            @inbounds @simd for i in s:e
                c[i] = -ctps.c[i]
            end
        end
    end
    c0 = (mask & UInt64(1) != 0) ? -ctps.c[1] : zero(T)
    c[1] = c0 + T(a)
    m_out = (mask & ~UInt64(1)) | (_prunable_zero(c[1]) ? UInt64(0) : UInt64(1))
    return CTPS{T}(c, desc, Ref(m_out))
end

function -(ctps::CTPS{T}) where T
    ctps = _ad_input(ctps)
    ctps_new = CTPS(ctps)   # range-limited copy
    mask = ctps.degree_mask[]
    if mask != 0
        for (s, e) in active_ranges(ctps.desc, mask)
            @inbounds @simd for i in s:e
                ctps_new.c[i] = -ctps_new.c[i]
            end
        end
    end
    ctps_new.degree_mask[] = mask
    return ctps_new
end

# * (allocating wrapper — delegates to mul! to avoid code duplication)
function *(ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    # Check descriptor compatibility
    _check_descriptors(ctps1, ctps2)
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

# Scalar division. The scalar may be any `Number` (e.g. an `Int` literal in
# `ctps / 2`) and is converted to the coefficient type, matching `+`, `-`, `*`.
# A single method per direction avoids the `(CTPS{T}, T)` / `(CTPS{T}, Number)`
# specificity ambiguity that made `ctps / 2.0` recurse.
function /(ctps::CTPS{T}, a::Number) where T
    b = T(a)
    iszero(b) && throw(DomainError(b, "division by zero"))
    ctps_new = CTPS(ctps)       # range-limited copy
    scale!(ctps_new, one(T)/b)  # range-limited scale
    return ctps_new
end

function /(a::Number, ctps::CTPS{T}) where T
    y = inv(ctps)
    scale!(y, T(a))
    return y
end

# ── Graded (Euler-operator) recurrences for elementary functions ─────────────
#
# Write f = a₀ + h with h the degrees ≥ 1, and split every series into
# homogeneous degree blocks, y = Σ_k y_k. The Euler operator E = Σ xᵢ∂ᵢ acts
# on a homogeneous block as multiplication by its degree, so an ODE in f
# becomes a recurrence over blocks that costs ONE block-convolution in total:
#
#   exp:   E y = y·E h            → k y_k = Σ_{j=1}^{k} j h_j ⊛ y_{k-j}
#   sin/cos: E s = c·E h, E c = -s·E h
#                                 → k s_k =  Σ j h_j ⊛ c_{k-j},  k c_k = -Σ j h_j ⊛ s_{k-j}
#   sinh/cosh: same with a plus sign in the second line.
#
# Accumulating powers hⁱ/i! instead (the previous implementation) performs
# Σ dᵢ·Nd[dᵢ]·Nd[dⱼ] block products against Σ Nd[dᵢ]·Nd[dⱼ] here — 2.4×
# more work at nv=2 order 6, 4.8× at nv=2 order 12 — and pays mul!'s
# per-call cost `order` times over. Only blocks of h that are active are
# visited, and a block of the result is written only when some active h_j
# can reach it, so sparsity in the argument is preserved in the result.
#
# `⊛` below is the block product taken from the multiplication schedules:
# the (deg_a ≥ deg_b) schedule maps (i in the deg_a block, j in the deg_b
# block) to the output index of the product monomial in either orientation.

# out[deg_a + deg_b block] += w · (a's deg_a block) ⊛ (b's deg_b block)
@inline function _block_product_add!(cr::Vector{T}, a::Vector{T}, deg_a::Int,
                                     b::Vector{T}, deg_b::Int, w::T, desc::PSDesc) where T
    if deg_a >= deg_b
        @inbounds sched = desc.mul[desc.mul_offsets[deg_a + 1] + deg_b]
        Ni, Nj = Int(sched.Ni), Int(sched.Nj)
        i_base, j_base = Int(sched.i_start) - 1, Int(sched.j_start) - 1
        k_mat = sched.k_local
        @inbounds @fastmath for i_local in 1:Ni          # a in the di block
            ai = w * a[i_base + i_local]
            _prunable_zero(ai) && continue
            for j_local in 1:Nj
                cr[k_mat[j_local, i_local]] += ai * b[j_base + j_local]
            end
        end
    else
        @inbounds sched = desc.mul[desc.mul_offsets[deg_b + 1] + deg_a]
        Ni, Nj = Int(sched.Ni), Int(sched.Nj)
        i_base, j_base = Int(sched.i_start) - 1, Int(sched.j_start) - 1
        k_mat = sched.k_local
        @inbounds @fastmath for i_local in 1:Ni          # b in the di block
            bw = w * b[i_base + i_local]
            _prunable_zero(bw) && continue
            for j_local in 1:Nj
                cr[k_mat[j_local, i_local]] += a[j_base + j_local] * bw
            end
        end
    end
    return nothing
end

@inline function _zero_block!(c::Vector{T}, desc::PSDesc, degree::Int) where T
    s = desc.off[degree + 1]
    e = s + desc.Nd[degree + 1] - 1
    @inbounds @simd for i in s:e
        c[i] = zero(T)
    end
    return nothing
end

# Active degrees j of h (1 ≤ j ≤ k) for which the partner block k-j is present
# in `reach`; zero means block k of the result cannot be reached. During AD
# every block is computed, since a numerically zero block can carry a tangent.
@inline function _reaching_degrees(hmask::UInt64, reach::UInt64, k::Int)
    candidates = hmask & (typemax(UInt64) >> (63 - k)) & ~UInt64(1)
    within_autodiff() && return candidates
    contrib = UInt64(0)
    m = candidates
    while m != 0
        j = trailing_zeros(m)
        m &= m - UInt64(1)
        (reach >> (k - j)) & UInt64(1) != 0 && (contrib |= UInt64(1) << j)
    end
    return contrib
end

# y = exp(f).  `y` must not share storage with `f`; block 0 of y is written
# here and blocks 1..order are produced in increasing degree.
function _exp_series!(y::CTPS{T}, f::CTPS{T}) where T
    desc  = f.desc
    order = desc.order
    fc, yc = f.c, y.c
    hmask = f.degree_mask[] & ~UInt64(1)
    yc[1] = Base.exp(cst(f))
    reach = UInt64(1)
    for k in 1:order
        contrib = _reaching_degrees(hmask, reach, k)
        contrib == 0 && continue
        _zero_block!(yc, desc, k)
        inv_k = one(T) / T(k)
        while contrib != 0
            j = trailing_zeros(contrib)
            contrib &= contrib - UInt64(1)
            _block_product_add!(yc, fc, j, yc, k - j, T(j) * inv_k, desc)
        end
        reach |= UInt64(1) << k
    end
    y.degree_mask[] = reach
    return y
end

# Explicit AD boundary: the derivative is multiplication by the primal exp.
# All coefficient slots are initialized before entering this primitive.
function _dense_exp!(yc::Vector{T}, fc::Vector{T}, desc::PSDesc) where T
    mask = typemax(UInt64) >> (63-desc.order)
    f = CTPS{T}(fc,desc,Ref(mask))
    y = CTPS{T}(yc,desc,Ref(mask))
    _exp_series!(y,f)
    return nothing
end

@inline function _exp_dispatch!(y::CTPS{T}, f::CTPS{T}) where T
    if within_autodiff() && T <: Union{Float32,Float64}
        _dense_exp!(y.c,f.c,f.desc)
        y.degree_mask[] = typemax(UInt64) >> (63-f.desc.order)
        return y
    end
    return _exp_series!(y,f)
end

# (s, c) = (sin(f), cos(f)) or (sinh(f), cosh(f)) when `hyperbolic`. Neither
# output may share storage with `f` or with the other. Each recurrence reads
# its partner only up to degree order-1, so when a single function is wanted
# the companion's top block — by far the largest — is skipped (`s_top`/`c_top`
# false); the companion's mask then omits that degree.
# Evaluate both scalar centers before borrowing scratch storage or writing an
# output. Passing the values into the recurrence avoids evaluating them twice.
@inline function _sincos_centers(a0::T, hyperbolic::Bool) where T
    s0 = T(hyperbolic ? Base.sinh(a0) : Base.sin(a0))
    c0 = T(hyperbolic ? Base.cosh(a0) : Base.cos(a0))
    return s0, c0
end

function _sincos_series!(s::CTPS{T}, c::CTPS{T}, f::CTPS{T}, hyperbolic::Bool,
                         s0::T, c0::T;
                         s_top::Bool=true, c_top::Bool=true) where T
    desc  = f.desc
    order = desc.order
    fc, sc, cc = f.c, s.c, c.c
    hmask = f.degree_mask[] & ~UInt64(1)
    sc[1] = s0
    cc[1] = c0
    sign  = hyperbolic ? one(T) : -one(T)
    reach = UInt64(1)
    for k in 1:order
        contrib = _reaching_degrees(hmask, reach, k)
        contrib == 0 && continue
        do_s = s_top || k < order
        do_c = c_top || k < order
        do_s && _zero_block!(sc, desc, k)
        do_c && _zero_block!(cc, desc, k)
        inv_k = one(T) / T(k)
        while contrib != 0
            j = trailing_zeros(contrib)
            contrib &= contrib - UInt64(1)
            w = T(j) * inv_k
            do_s && _block_product_add!(sc, fc, j, cc, k - j, w, desc)
            do_c && _block_product_add!(cc, fc, j, sc, k - j, sign * w, desc)
        end
        reach |= UInt64(1) << k
    end
    top = UInt64(1) << order
    s.degree_mask[] = s_top ? reach : reach & ~top
    c.degree_mask[] = c_top ? reach : reach & ~top
    return nothing
end

# ── Recurrences with a division: log, sqrt, inv, /, asin ─────────────────────
#
#   (E y)·g = sign·E h          (log: g = h = f;  asin: g = √(1-f²), h = f)
#     → y_k = sign·h_k/g₀ − Σ_{j=1}^{k-1} ((k-j)/(k g₀)) g_j ⊛ y_{k-j}
#   y² = w                       (sqrt; w_k supplied as hsign·h_k)
#     → y_k = hsign·h_k/(2y₀) − (1/(2y₀)) Σ_{j=1}^{k-1} y_j ⊛ y_{k-j}
#       and the symmetric sum takes each unordered pair once.
#   y·f = 1                      (inv)   → y_k = −(1/a₀) Σ_{j=1}^{k} h_j ⊛ y_{k-j}
#   y·b = a                      (÷)     → y_k = (a_k − Σ_{j=1}^{k} b_j ⊛ y_{k-j})/b₀
#
# Block k of the result is written from its explicit term first (an elementwise
# read-then-write at the same index, so that term may come from a series that
# shares storage with the result), then the products are accumulated.

# Write block k of `yc` as w·hc[block k] when the block of h is active, else zero.
@inline function _seed_block!(yc::Vector{T}, hc::Vector{T}, hmask::UInt64, k::Int,
                              w::T, desc::PSDesc) where T
    s = desc.off[k + 1]
    e = s + desc.Nd[k + 1] - 1
    if (hmask >> k) & UInt64(1) != 0
        @inbounds @simd for i in s:e
            yc[i] = w * hc[i]
        end
        return true
    end
    @inbounds @simd for i in s:e
        yc[i] = zero(T)
    end
    return false
end

# Solve (E y)·g = sign·E h with y₀ given. `y` may share storage with h (each h
# block is consumed by _seed_block! before anything else touches it) but not
# with g.
function _euler_divide_series!(y::CTPS{T}, y0::T, hc::Vector{T}, hmask::UInt64,
                               gc::Vector{T}, gmask::UInt64, g0::T, sign::T,
                               desc::PSDesc) where T
    order = desc.order
    yc = y.c
    hm = hmask & ~UInt64(1)
    gm = gmask & ~UInt64(1)
    yc[1] = y0
    inv_g0 = one(T) / g0
    reach = UInt64(1)
    for k in 1:order
        # g_j with 1 ≤ j ≤ k-1 paired with an already reached y_{k-j}
        contrib = _reaching_degrees(gm, reach, k) & ~(UInt64(1) << k)
        has_h = (hm >> k) & UInt64(1) != 0
        (contrib == 0 && !has_h && !within_autodiff()) && continue
        _seed_block!(yc, hc, hm, k, sign * inv_g0, desc)
        inv_kg0 = inv_g0 / T(k)
        while contrib != 0
            j = trailing_zeros(contrib)
            contrib &= contrib - UInt64(1)
            _block_product_add!(yc, gc, j, yc, k - j, -T(k - j) * inv_kg0, desc)
        end
        reach |= UInt64(1) << k
    end
    y.degree_mask[] = reach
    return y
end

# y = √(y₀² + hsign·h) with the root fixed by y₀. `y` may share storage with h.
function _sqrt_series!(y::CTPS{T}, y0::T, hc::Vector{T}, hmask::UInt64, hsign::T,
                       desc::PSDesc) where T
    order = desc.order
    yc = y.c
    hm = hmask & ~UInt64(1)
    yc[1] = y0
    inv_2y0 = one(T) / (y0 + y0)
    reach = UInt64(1)
    for k in 1:order
        # pairs (j, k-j) of reached y blocks, 1 ≤ j ≤ k-1
        contrib = _reaching_degrees(reach & ~UInt64(1), reach, k) & ~(UInt64(1) << k)
        has_h = (hm >> k) & UInt64(1) != 0
        (contrib == 0 && !has_h && !within_autodiff()) && continue
        _seed_block!(yc, hc, hm, k, hsign * inv_2y0, desc)
        while contrib != 0
            j = trailing_zeros(contrib)
            contrib &= contrib - UInt64(1)
            2j > k && break                       # ascending j: the rest are mirrors
            w = 2j == k ? -inv_2y0 : -(inv_2y0 + inv_2y0)
            _block_product_add!(yc, yc, j, yc, k - j, w, desc)
        end
        reach |= UInt64(1) << k
    end
    y.degree_mask[] = reach
    return y
end

# y = 1/f. `y` must not share storage with f.
function _inv_series!(y::CTPS{T}, f::CTPS{T}) where T
    desc  = f.desc
    order = desc.order
    fc, yc = f.c, y.c
    hm = f.degree_mask[] & ~UInt64(1)
    a0 = cst(f)
    inv_a0 = one(T) / a0
    yc[1] = inv_a0
    reach = UInt64(1)
    for k in 1:order
        contrib = _reaching_degrees(hm, reach, k)
        contrib == 0 && continue
        _zero_block!(yc, desc, k)
        while contrib != 0
            j = trailing_zeros(contrib)
            contrib &= contrib - UInt64(1)
            _block_product_add!(yc, fc, j, yc, k - j, -inv_a0, desc)
        end
        reach |= UInt64(1) << k
    end
    y.degree_mask[] = reach
    return y
end

# y = a/b. `y` may share storage with a, not with b.
function _div_series!(y::CTPS{T}, a::CTPS{T}, b::CTPS{T}) where T
    desc  = a.desc
    order = desc.order
    ac, bc, yc = a.c, b.c, y.c
    am = a.degree_mask[]
    bm = b.degree_mask[] & ~UInt64(1)
    b0 = cst(b)
    inv_b0 = one(T) / b0
    yc[1] = cst(a) * inv_b0
    reach = UInt64(1)
    for k in 1:order
        contrib = _reaching_degrees(bm, reach, k)
        has_a = (am >> k) & UInt64(1) != 0
        (contrib == 0 && !has_a && !within_autodiff()) && continue
        _seed_block!(yc, ac, am, k, inv_b0, desc)
        while contrib != 0
            j = trailing_zeros(contrib)
            contrib &= contrib - UInt64(1)
            _block_product_add!(yc, bc, j, yc, k - j, -inv_b0, desc)
        end
        reach |= UInt64(1) << k
    end
    y.degree_mask[] = reach
    return y
end

# y = asin(f) (or acos when `want_acos`), given scratch series t and g that do
# not share storage with y or f. y may share storage with f.
function _asin_series!(y::CTPS{T}, f::CTPS{T}, want_acos::Bool,
                       t::CTPS{T}, g::CTPS{T}) where T
    desc = f.desc
    a0 = cst(f)
    g0 = _asin_root(a0)                       # the branch of Base.asin's derivative
    # g = √(1 − f²): t = f², then the square-root recurrence on 1 − t with the
    # root fixed by g0 (so the branch is that of the scalar function).
    mul!(t, f, f)
    _sqrt_series!(g, g0, t.c, t.degree_mask[], -one(T), desc)
    y0   = want_acos ? Base.acos(a0) : Base.asin(a0)
    sign = want_acos ? -one(T) : one(T)
    _euler_divide_series!(y, y0, f.c, f.degree_mask[], g.c, g.degree_mask[], g0, sign, desc)
    return y
end


# exponential
function exp(ctps::CTPS{T}) where T
    ctps = _ad_input(ctps)
    y = _ctps_zero(T, ctps.desc)
    return _exp_dispatch!(y, ctps)
end

function exp!(result::CTPS{T}, ctps::CTPS{T}) where T
    ctps = _ad_input(ctps)
    _check_descriptors(result, ctps)
    desc = ctps.desc
    if result.c === ctps.c
        # The recurrence reads block k of f while writing block k of y.
        (idx, src) = _ctps_pooled_copy(ctps, desc)
        _exp_dispatch!(result, src)
        _pool_release!(idx, src, desc)
    else
        _exp_dispatch!(result, ctps)
    end
    return result
end

# sin / cos, together and separately
function _sincos_into!(s::CTPS{T}, c::CTPS{T}, ctps::CTPS{T}, hyperbolic::Bool) where T
    ctps = _ad_input(ctps)
    _check_descriptors(s, ctps)
    _check_descriptors(c, ctps)
    s.c === c.c && throw(ArgumentError("sincos!: the two outputs must not share storage"))
    desc = ctps.desc
    s0, c0 = _sincos_centers(cst(ctps), hyperbolic)
    if s.c === ctps.c || c.c === ctps.c
        (idx, src) = _ctps_pooled_copy(ctps, desc)
        try
            _sincos_series!(s, c, src, hyperbolic, s0, c0)
        finally
            _pool_release!(idx, src, desc)
        end
    else
        _sincos_series!(s, c, ctps, hyperbolic, s0, c0)
    end
    return nothing
end

"""
    sincos!(s::CTPS, c::CTPS, p::CTPS) -> nothing

Write `sin(p)` to `s` and `cos(p)` to `c` in one pass. The two series share
every intermediate, so this costs about the same as either one alone. `s` and
`c` must not share storage with each other; either may alias `p`.
"""
sincos!(s::CTPS{T}, c::CTPS{T}, ctps::CTPS{T}) where T = _sincos_into!(s, c, ctps, false)

# Single-output forms borrow one pooled series for the companion function.
function _single_sincos!(result::CTPS{T}, ctps::CTPS{T}, want_sin::Bool, hyperbolic::Bool) where T
    ctps = _ad_input(ctps)
    _check_descriptors(result, ctps)
    desc = ctps.desc
    s0, c0 = _sincos_centers(cst(ctps), hyperbolic)
    (idx_other, other) = _ctps_pooled(T, desc)
    try
        if result.c === ctps.c
            (idx_src, src) = _ctps_pooled_copy(ctps, desc)
            try
                want_sin ? _sincos_series!(result, other, src, hyperbolic, s0, c0; c_top=false) :
                           _sincos_series!(other, result, src, hyperbolic, s0, c0; s_top=false)
            finally
                _pool_release!(idx_src, src, desc)
            end
        else
            want_sin ? _sincos_series!(result, other, ctps, hyperbolic, s0, c0; c_top=false) :
                       _sincos_series!(other, result, ctps, hyperbolic, s0, c0; s_top=false)
        end
    finally
        _pool_release!(idx_other, other, desc)
    end
    return result
end

function _single_sincos(ctps::CTPS{T}, want_sin::Bool, hyperbolic::Bool) where T
    ctps = _ad_input(ctps)
    desc = ctps.desc
    s0, c0 = _sincos_centers(cst(ctps), hyperbolic)
    result = _ctps_zero(T, desc)
    (idx_other, other) = _ctps_pooled(T, desc)   # heap under AD, pool otherwise
    try
        want_sin ? _sincos_series!(result, other, ctps, hyperbolic, s0, c0; c_top=false) :
                   _sincos_series!(other, result, ctps, hyperbolic, s0, c0; s_top=false)
    finally
        _pool_release!(idx_other, other, desc)
    end
    return result
end

sin(ctps::CTPS)  = _single_sincos(ctps, true,  false)
cos(ctps::CTPS)  = _single_sincos(ctps, false, false)
sinh(ctps::CTPS) = _single_sincos(ctps, true,  true)
cosh(ctps::CTPS) = _single_sincos(ctps, false, true)
sin!(result::CTPS{T}, ctps::CTPS{T}) where T  = _single_sincos!(result, ctps, true,  false)
cos!(result::CTPS{T}, ctps::CTPS{T}) where T  = _single_sincos!(result, ctps, false, false)
sinh!(result::CTPS{T}, ctps::CTPS{T}) where T = _single_sincos!(result, ctps, true,  true)
cosh!(result::CTPS{T}, ctps::CTPS{T}) where T = _single_sincos!(result, ctps, false, true)

# ── inverse, division, logarithm, square root ────────────────────────────────

"""
    inv!(out::CTPS, p::CTPS) -> out

Write `1/p` to `out`. The constant term of `p` must be nonzero. `out` may alias `p`.
"""
function inv!(result::CTPS{T}, ctps::CTPS{T}) where T
    ctps = _ad_input(ctps)
    _check_descriptors(result, ctps)
    a0 = cst(ctps)
    iszero(a0) && throw(DomainError(a0, "inv: the constant term of the series must be nonzero"))
    desc = ctps.desc
    if result.c === ctps.c
        (idx, src) = _ctps_pooled_copy(ctps, desc)
        _inv_series!(result, src)
        _pool_release!(idx, src, desc)
    else
        _inv_series!(result, ctps)
    end
    return result
end

function inv(ctps::CTPS{T}) where T
    ctps = _ad_input(ctps)
    a0 = cst(ctps)
    iszero(a0) && throw(DomainError(a0, "inv: the constant term of the series must be nonzero"))
    return _inv_series!(_ctps_zero(T, ctps.desc), ctps)
end

"""
    div!(out::CTPS, a::CTPS, b::CTPS) -> out

Write `a / b` to `out` with a single division recurrence (no intermediate
inverse). The constant term of `b` must be nonzero. `out` may alias `a` or `b`.
"""
function div!(result::CTPS{T}, a::CTPS{T}, b::CTPS{T}) where T
    a = _ad_input(a)
    b = _ad_input(b)
    _check_descriptors(result, a)
    _check_descriptors(a, b)
    b0 = cst(b)
    iszero(b0) && throw(DomainError(b0, "division: the constant term of the divisor must be nonzero"))
    desc = a.desc
    if result.c === b.c
        (idx, src) = _ctps_pooled_copy(b, desc)
        _div_series!(result, a, src)          # aliasing a is fine: seeded first
        _pool_release!(idx, src, desc)
    else
        _div_series!(result, a, b)
    end
    return result
end

function /(ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    ctps1 = _ad_input(ctps1)
    ctps2 = _ad_input(ctps2)
    _check_descriptors(ctps1, ctps2)
    b0 = cst(ctps2)
    iszero(b0) && throw(DomainError(b0, "division: the constant term of the divisor must be nonzero"))
    return _div_series!(_ctps_zero(T, ctps1.desc), ctps1, ctps2)
end

@inline function _log_center(a0)
    iszero(a0) && throw(DomainError(a0, "log: the constant term of the series must be nonzero"))
    return Base.log(a0)                       # DomainError for a negative real center
end

function log(ctps::CTPS{T}) where T
    ctps = _ad_input(ctps)
    a0 = cst(ctps)
    log_a0 = _log_center(a0)
    y = _ctps_zero(T, ctps.desc)
    m = ctps.degree_mask[]
    return _euler_divide_series!(y, log_a0, ctps.c, m, ctps.c, m, a0, one(T), ctps.desc)
end

function log!(result::CTPS{T}, ctps::CTPS{T}) where T
    ctps = _ad_input(ctps)
    _check_descriptors(result, ctps)
    a0 = cst(ctps)
    log_a0 = _log_center(a0)                  # validate before touching result
    desc = ctps.desc
    if result.c === ctps.c
        (idx, src) = _ctps_pooled_copy(ctps, desc)
        m = src.degree_mask[]
        _euler_divide_series!(result, log_a0, src.c, m, src.c, m, a0, one(T), desc)
        _pool_release!(idx, src, desc)
    else
        m = ctps.degree_mask[]
        _euler_divide_series!(result, log_a0, ctps.c, m, ctps.c, m, a0, one(T), desc)
    end
    return result
end

@inline function _sqrt_center(a0::T) where T
    iszero(a0) && throw(DomainError(a0, "Square root requires a nonzero expansion center"))
    T <: Real && a0 < zero(T) && throw(DomainError(a0, "sqrt: negative real expansion center; use complex coefficients"))
    return Base.sqrt(a0)
end

# The square-root recurrence consumes each block of the argument before it
# writes the same block of the result, so the output may alias the input.
function sqrt(ctps::CTPS{T}) where T
    ctps = _ad_input(ctps)
    y0 = _sqrt_center(cst(ctps))
    return _sqrt_series!(_ctps_zero(T, ctps.desc), y0, ctps.c, ctps.degree_mask[], one(T), ctps.desc)
end

function sqrt!(result::CTPS{T}, ctps::CTPS{T}) where T
    ctps = _ad_input(ctps)
    _check_descriptors(result, ctps)
    y0 = _sqrt_center(cst(ctps))
    return _sqrt_series!(result, y0, ctps.c, ctps.degree_mask[], one(T), ctps.desc)
end

# Factor before rounding: subtracting the rounded square loses relative
# accuracy near the real branch points at ±1.
@inline _asin_root(a0) = Base.sqrt((one(a0) - a0) * (one(a0) + a0))
@inline function _asin_root(a0::Complex)
    a, b = reim(a0)
    # sqrt(1-z)*sqrt(1+z) selects the derivative branch of Base.asin.
    # Construct the imaginary parts explicitly: complex subtraction can lose
    # the -0.0 in 1-(a+0.0im), selecting the other side of the cut.
    return Base.sqrt(Complex(one(a) - a, -b)) *
           Base.sqrt(Complex(one(a) + a, b))
end

@inline function _asin_center(a0::T) where T
    T <: Real && abs(a0) >= one(T) && throw(DomainError(a0, "asin/acos: |constant term| must be < 1 for real coefficients"))
    iszero(_asin_root(a0)) && throw(DomainError(a0, "Inverse trigonometric series require a nonsingular expansion center"))
    return nothing
end

# Two scratch series (f² and √(1−f²)); the result may alias the argument.
function _asin_into!(result::CTPS{T}, ctps::CTPS{T}, want_acos::Bool) where T
    ctps = _ad_input(ctps)
    _check_descriptors(result, ctps)
    _asin_center(cst(ctps))                   # validate before borrowing or writing
    desc = ctps.desc
    (idx_t, t) = _ctps_pooled(T, desc)
    (idx_g, g) = _ctps_pooled(T, desc)
    _asin_series!(result, ctps, want_acos, t, g)
    _pool_release!(idx_g, g, desc)
    _pool_release!(idx_t, t, desc)
    return result
end

asin(ctps::CTPS{T}) where T = _asin_into!(_ctps_zero(T, ctps.desc), ctps, false)
acos(ctps::CTPS{T}) where T = _asin_into!(_ctps_zero(T, ctps.desc), ctps, true)
asin!(result::CTPS{T}, ctps::CTPS{T}) where T = _asin_into!(result, ctps, false)
acos!(result::CTPS{T}, ctps::CTPS{T}) where T = _asin_into!(result, ctps, true)

# ── tangent: one shared sin/cos pass, then one division recurrence ───────────

"""
    tan!(out::CTPS, p::CTPS) -> out

Write `tan(p)` to `out`. `out` may alias `p`.
"""
function tan!(result::CTPS{T}, ctps::CTPS{T}) where T
    ctps = _ad_input(ctps)
    _check_descriptors(result, ctps)
    desc = ctps.desc
    s0, c0 = _sincos_centers(cst(ctps), false)
    (idx_s, s) = _ctps_pooled(T, desc)
    try
        (idx_c, c) = _ctps_pooled(T, desc)
        try
            _sincos_series!(s, c, ctps, false, s0, c0) # read ctps before touching result
            _div_series!(result, s, c)
        finally
            _pool_release!(idx_c, c, desc)
        end
    finally
        _pool_release!(idx_s, s, desc)
    end
    return result
end

function tan(ctps::CTPS{T}) where T
    ctps = _ad_input(ctps)
    return tan!(_ctps_zero(T, ctps.desc), ctps)
end

# power
function pow(ctps::CTPS{T}, b::Int) where T
    desc = ctps.desc
    b == 1 && return CTPS(ctps)
    b == 0 && return _ctps_constant(one(T), desc)
    if b < 0
        reciprocal = inv(ctps)
        # Invert first to avoid overflowing a positive intermediate power.
        # -(b + 1) also remains representable for typemin(Int).
        return b == typemin(Int) ? pow(reciprocal, -(b + 1)) * reciprocal :
                                  pow(reciprocal, -b)
    end

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
    _check_descriptors(result, ctps)
    desc = ctps.desc
    if b == 0
        _zero_active!(result)
        result.c[1] = one(T)
        result.degree_mask[] = UInt64(1)
        return result
    end
    b == 1 && (copy!(result, ctps); return result)
    b < 0  && throw(ArgumentError("pow!(result, ctps, b) requires b >= 0; use inv(pow(ctps, -b)) for negative exponents"))
    # The square/cube shortcuts multiply directly into result. Preserve the
    # base when output shares its coefficient buffer, including another wrapper.
    if (b == 2 || b == 3) && result.c === ctps.c
        idx, input = _ctps_pooled_copy(ctps, desc)
        try
            return pow!(result, input, b)
        finally
            _pool_release!(idx, input, desc)
        end
    end
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

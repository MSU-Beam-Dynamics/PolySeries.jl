"""
TPSA Math Function Benchmark & Accuracy Verification
=====================================================
Tests all math functions for:
  1. Taylor coefficient accuracy  (vs analytical reference)
  2. Pointwise polynomial accuracy (vs Base function at nearby points)
  3. Performance: allocating vs in-place, across orders
"""

using PolySeries
using BenchmarkTools
using Printf

# ─── helpers ─────────────────────────────────────────────────────────────────

"""Return list of Taylor coefficients of f(a0+t) at t=0 up to degree `order`."""
function ref_exp(a0, order)
    e = exp(a0)
    [e / factorial(k) for k in 0:order]
end
function ref_log(a0, order)
    coeffs = Vector{Float64}(undef, order + 1)
    coeffs[1] = log(a0)
    for k in 1:order
        coeffs[k + 1] = (-1)^(k + 1) / (k * a0^k)
    end
    coeffs
end
function ref_sqrt(a0, order)
    coeffs = Vector{Float64}(undef, order + 1)
    coeffs[1] = sqrt(a0)
    binom = 1.0
    for k in 1:order
        binom *= (0.5 - (k - 1)) / k   # C(1/2, k)
        coeffs[k + 1] = binom * a0^(0.5 - k)
    end
    coeffs
end
function ref_inv(a0, order)
    [(-1)^k / a0^(k + 1) for k in 0:order]
end
function ref_sin(a0, order)
    sa, ca = sin(a0), cos(a0)
    cycle  = [sa, ca, -sa, -ca]
    [cycle[mod(k, 4) + 1] / factorial(k) for k in 0:order]
end
function ref_cos(a0, order)
    sa, ca = sin(a0), cos(a0)
    cycle  = [ca, -sa, -ca, sa]
    [cycle[mod(k, 4) + 1] / factorial(k) for k in 0:order]
end
function ref_sinh(a0, order)
    sa, ca = sinh(a0), cosh(a0)
    [iseven(k) ? sa / factorial(k) : ca / factorial(k) for k in 0:order]
end
function ref_cosh(a0, order)
    sa, ca = sinh(a0), cosh(a0)
    [iseven(k) ? ca / factorial(k) : sa / factorial(k) for k in 0:order]
end
function ref_asin(a0, order)
    c0      = sqrt(1.0 - a0^2)
    inv_c0  = 1.0 / c0
    inv_2c0 = inv_c0 / 2.0
    B = Vector{Float64}(undef, order + 1)
    B[1] = c0
    for n in 1:order
        R = n == 1 ? -2.0 * a0 : (n == 2 ? -1.0 : 0.0)
        s = sum(B[j+1] * B[n-j+1] for j in 1:n-1; init = 0.0)
        B[n+1] = (R - s) * inv_2c0
    end
    coeffs    = Vector{Float64}(undef, order + 1)
    coeffs[1] = asin(a0)   # h^0 term
    if order >= 1
        coeffs[2] = inv_c0   # h^1 term
        A = Vector{Float64}(undef, order)
        A[1] = inv_c0
        for n in 1:order-1
            s = sum(B[j+1] * (n - j + 1) * A[n-j+1] for j in 1:n; init = 0.0)
            A[n+1] = -s * inv_c0 / (n + 1)
            coeffs[n+2] = A[n+1]
        end
    end
    coeffs
end
function ref_acos(a0, order)
    c = ref_asin(a0, order)
    c[1] = acos(a0)   # π/2 - asin(a0)
    for k in 2:order+1
        c[k] = -c[k]
    end
    c
end

"""Evaluate a 1-variable CTPS polynomial at a0+h: sum c[k] * h^k."""
function polyval(y::CTPS, h::Float64)
    order = y.desc.order
    val   = 0.0
    hk    = 1.0
    for k in 0:order
        val += element(y, [k]) * hk
        hk  *= h
    end
    val
end

"""Check Taylor coefficients against a reference vector. Returns max abs error."""
function check_coeffs(y::CTPS, ref::Vector{Float64})
    order = min(y.desc.order, length(ref) - 1)
    maxerr = 0.0
    for k in 0:order
        got = real(element(y, [k]))
        err = abs(got - ref[k + 1])
        maxerr = max(maxerr, err)
    end
    maxerr
end

"""Check polyval(y, h) against basefn(a0+h) for several small h values."""
function check_pointwise(y::CTPS, basefn, a0::Float64; npts::Int = 5)
    hs     = [0.001 * i for i in 1:npts]
    maxerr = 0.0
    for h in hs
        got = real(polyval(y, h))
        ref = basefn(a0 + h)
        err = abs(ref) > 1e-12 ? abs(got - ref) / abs(ref) : abs(got - ref)
        maxerr = max(maxerr, err)
    end
    maxerr
end

# ─── accuracy test runner ─────────────────────────────────────────────────────

struct AccuracyResult
    name      :: String
    coeff_err :: Float64   # max coefficient error  (NaN if not checked)
    point_err :: Float64   # max pointwise rel error
end

function run_accuracy(order::Int = 8)
    a0 = 0.5   # expand at 0.5 — valid for log/sqrt/inv/asin/acos (|a0|<1)

    # (name, allocated fn, base fn for pointwise, ref coeff fn or nothing)
    tests = [
        ("exp",   PolySeries.exp,  Base.exp,  (a,o) -> ref_exp(a, o)),
        ("log",   PolySeries.log,  Base.log,  (a,o) -> ref_log(a, o)),
        ("sqrt",  PolySeries.sqrt, Base.sqrt, (a,o) -> ref_sqrt(a, o)),
        ("inv",   inv,       x -> 1/x,  (a,o) -> ref_inv(a, o)),
        ("sin",   PolySeries.sin,  Base.sin,  (a,o) -> ref_sin(a, o)),
        ("cos",   PolySeries.cos,  Base.cos,  (a,o) -> ref_cos(a, o)),
        ("tan",   PolySeries.tan,  Base.tan,  nothing),
        ("sinh",  PolySeries.sinh, Base.sinh, (a,o) -> ref_sinh(a, o)),
        ("cosh",  PolySeries.cosh, Base.cosh, (a,o) -> ref_cosh(a, o)),
        ("asin",  PolySeries.asin, Base.asin, (a,o) -> ref_asin(a, o)),
        ("acos",  PolySeries.acos, Base.acos, (a,o) -> ref_acos(a, o)),
    ]

    # In-place variants (result must match allocating version)
    inplace_tests = [
        ("exp!",   exp!,   PolySeries.exp,  (a,o) -> ref_exp(a, o)),
        ("log!",   log!,   PolySeries.log,  (a,o) -> ref_log(a, o)),
        ("sqrt!",  sqrt!,  PolySeries.sqrt, (a,o) -> ref_sqrt(a, o)),
        ("sin!",   sin!,   PolySeries.sin,  (a,o) -> ref_sin(a, o)),
        ("cos!",   cos!,   PolySeries.cos,  (a,o) -> ref_cos(a, o)),
        ("sinh!",  sinh!,  PolySeries.sinh, (a,o) -> ref_sinh(a, o)),
        ("cosh!",  cosh!,  PolySeries.cosh, (a,o) -> ref_cosh(a, o)),
        ("asin!",  asin!,  PolySeries.asin, (a,o) -> ref_asin(a, o)),
        ("acos!",  acos!,  PolySeries.acos, (a,o) -> ref_acos(a, o)),
    ]

    results = AccuracyResult[]

    # Allocating functions
    for (name, fn, basefn, reffn) in tests
        set_descriptor!(1, order)
        x = CTPS(a0, 1)
        y = fn(x)
        ref = isnothing(reffn) ? nothing : reffn(a0, order)
        ce  = isnothing(ref)   ? NaN     : check_coeffs(y, ref)
        pe  = check_pointwise(y, basefn, a0)
        push!(results, AccuracyResult(name, ce, pe))
    end

    # In-place functions
    for (name, fn!, basefn, reffn) in inplace_tests
        set_descriptor!(1, order)
        x = CTPS(a0, 1)
        r = CTPS(Float64)
        fn!(r, x)
        ref = reffn(a0, order)
        ce  = check_coeffs(r, ref)
        pe  = check_pointwise(r, basefn, a0)
        push!(results, AccuracyResult(name, ce, pe))
    end

    results
end

# ─── performance test runner ─────────────────────────────────────────────────

struct PerfResult
    name     :: String
    nv       :: Int
    order    :: Int
    N        :: Int
    alloc_ms :: Float64
    inpl_ms  :: Float64   # NaN if no in-place version
    alloc_n  :: Int
    inpl_n   :: Int       # -1 if no in-place
end

function run_performance(configs; n_samples::Int = 150)
    fn_list = [
        ("exp",   PolySeries.exp,  exp!,   true ),
        ("log",   PolySeries.log,  log!,   true ),
        ("sqrt",  PolySeries.sqrt, sqrt!,  true ),
        ("sin",   PolySeries.sin,  sin!,   true ),
        ("cos",   PolySeries.cos,  cos!,   true ),
        ("tan",   PolySeries.tan,  nothing, false),
        ("sinh",  PolySeries.sinh, sinh!,  true ),
        ("cosh",  PolySeries.cosh, cosh!,  true ),
        ("asin",  PolySeries.asin, asin!,  true ),
        ("acos",  PolySeries.acos, acos!,  true ),
        ("inv",   inv,       nothing, false),
    ]

    results = PerfResult[]
    for (nv, order) in configs
        for (name, fn, fn!, has_ip) in fn_list
            set_descriptor!(nv, order)
            N = get_descriptor().N
            x = CTPS(0.5, 1)
            r = CTPS(Float64)

            ta = @benchmark $fn($x)        samples=n_samples evals=10
            alloc_ms = median(ta).time / 1e6
            alloc_n  = ta.allocs

            if has_ip
                ti      = @benchmark $fn!($r, $x) samples=n_samples evals=10
                inpl_ms = median(ti).time / 1e6
                inpl_n  = ti.allocs
            else
                inpl_ms = NaN
                inpl_n  = -1
            end

            push!(results, PerfResult(name, nv, order, N,
                                      alloc_ms, inpl_ms, alloc_n, inpl_n))
        end
    end
    results
end

# ─── main ────────────────────────────────────────────────────────────────────

function main()
    println("="^78)
    println("TPSA Math Function Benchmark & Accuracy Verification")
    println("="^78)

    # ── 1. Accuracy ──────────────────────────────────────────────────────────
    println("\n── 1. ACCURACY  (nv=1, order=8, a₀=0.5) ────────────────────────────────")
    acc = run_accuracy(8)

    @printf("  %-8s  %18s  %18s  %s\n",
            "Function", "Max coeff err", "Max pointwise rerr", "Status")
    println("  " * "─"^62)
    all_pass = true
    for r in acc
        ce_str = isnan(r.coeff_err) ? "   (no ref formula)" : @sprintf("%18.3e", r.coeff_err)
        pe_str = @sprintf("%18.3e", r.point_err)
        pass   = (isnan(r.coeff_err) || r.coeff_err < 1e-10) && r.point_err < 1e-6
        all_pass = all_pass && pass
        @printf("  %-8s  %s  %s  %s\n", r.name, ce_str, pe_str,
                pass ? "PASS" : "FAIL ←")
    end
    println("  " * "─"^62)
    println(all_pass ? "  All checks PASSED ✓" : "  Some checks FAILED ✗")

    # ── 2. Performance across configs ────────────────────────────────────────
    configs = [(2, 6), (2, 10), (4, 6), (4, 8)]
    println("\n\n── 2. PERFORMANCE (times in ms, ratio = alloc/inpl) ─────────────────────")
    perf = run_performance(configs, n_samples=150)

    for (nv, order) in configs
        set_descriptor!(nv, order)
        N = get_descriptor().N
        println("\n  nv=$nv, order=$order, N=$N")
        @printf("  %-8s  %10s  %8s  %10s  %8s  %6s\n",
                "Function", "alloc(ms)", "allocs", "inpl(ms)", "allocs", "ratio")
        println("  " * "─"^60)
        for r in perf
            r.nv == nv && r.order == order || continue
            inpl_s  = isnan(r.inpl_ms) ? "         —" : @sprintf("%10.4f", r.inpl_ms)
            inpl_ns = r.inpl_n < 0     ? "       —" : @sprintf("%8d", r.inpl_n)
            ratio_s = isnan(r.inpl_ms) ? "     —" :
                      @sprintf("%6.2f", r.alloc_ms / r.inpl_ms)
            @printf("  %-8s  %10.4f  %8d  %s  %s  %s\n",
                    r.name, r.alloc_ms, r.alloc_n, inpl_s, inpl_ns, ratio_s)
        end
    end

    # ── 3. Order scaling for all functions (nv=2) ────────────────────────────
    println("\n\n── 3. ORDER SCALING  (nv=2, alloc time in ms) ───────────────────────────")
    scale_fns = [
        ("exp",  PolySeries.exp),  ("log",  PolySeries.log),  ("sqrt", PolySeries.sqrt),
        ("sin",  PolySeries.sin),  ("cos",  PolySeries.cos),  ("tan",  PolySeries.tan),
        ("sinh", PolySeries.sinh), ("cosh", PolySeries.cosh),
        ("asin", PolySeries.asin), ("acos", PolySeries.acos),
        ("inv",  inv),
    ]
    orders = [4, 6, 8, 10, 12]

    # Header
    hdr = @sprintf("  %-8s", "Function")
    for o in orders
        set_descriptor!(2, o); N = get_descriptor().N
        hdr *= @sprintf("  %12s", "o$o(N=$N)")
    end
    println(hdr)
    println("  " * "─"^(8 + length(orders) * 14))

    for (name, fn) in scale_fns
        row = @sprintf("  %-8s", name)
        for o in orders
            set_descriptor!(2, o)
            x = CTPS(0.5, 1)
            t = median(@benchmark $fn($x) samples=100 evals=10).time / 1e6
            row *= @sprintf("  %12.4f", t)
        end
        println(row)
    end

    println("\n" * "="^78)
    println("Benchmark complete.")
    println("="^78)
end

main()

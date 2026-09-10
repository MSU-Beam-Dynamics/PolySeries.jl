# Changelog

## [0.1.0] — unreleased

First registered release.

### Changed
- `mul!(out, a, b)` is now a method of `LinearAlgebra.mul!` and returns `out`
  (previously a package-local function returning `nothing`), so
  `using PolySeries, LinearAlgebra` no longer produces an export conflict.
- Error types are now part of the API contract: `DomainError` for singular
  expansion centres (division by a zero constant term, `log` at zero, real
  `sqrt` of a negative constant, real `asin`/`acos` at `|a| ≥ 1`),
  `ArgumentError` for invalid arguments, `DimensionMismatch` for descriptor
  mismatches and substitution maps of the wrong length. Several of these were
  `ErrorException` before.
- The legacy constructors `CTPS(T, nv, order)`, `CTPS(a, nv, order)` and
  `CTPS(a, n, nv, order)` validate their arguments like the explicit-descriptor
  forms (an `order == 0` variable used to hit a `BoundsError`).
- Minimum supported Julia version lowered to 1.10 (LTS).

### Performance
- `mul!` visits only the multiplication schedules whose degree pairs are
  active in the operands (previously it scanned all of them); sparse
  products are 1.3–2.2× faster and the output degree mask is computed in
  O(active degrees) instead of O(order²).
- All elementary functions (`exp`, `log`, `sqrt`, `inv`, `/`, `sin`, `cos`,
  `tan`, `asin`, `acos`, `sinh`, `cosh`) use degree-block (Euler operator)
  recurrences: one block convolution per function (three for `tan`, `asin`,
  `acos`) instead of one series multiplication per order — 2–5× less
  arithmetic on dense arguments and far less per-call overhead on sparse
  ones. Results keep the sparsity structure of the argument (an even
  argument yields only even degrees). `asin!`/`acos!` no longer allocate.

### Added
- `sincos!(s, c, p)` computes both trigonometric series in one pass;
  `tan!`, `inv!` and `div!` complete the in-place API.
- `@tpsa` supports `/`, `tan`, `asin`, `acos`, and `Number` (including
  complex) scalars.
- `PSDesc` rejects descriptors whose index tables would exceed
  `PolySeries.MAX_DESCRIPTOR_BYTES[]` (default 2 GiB) instead of attempting
  the allocation.
- `release!` rejects releasing the same workspace slot twice.
- `compose!` rejects a result that shares storage with `f` or a member of `g`.
- Reference tests for `mul!`/`pow!` against an independent convolution, and
  multivariate analytic identities for every math function.

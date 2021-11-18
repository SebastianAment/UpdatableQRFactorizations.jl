# UpdatableQRFactorizations.jl
[![codecov](https://codecov.io/gh/SebastianAment/UpdatableQRFactorizations.jl/branch/master/graph/badge.svg?token=HPB1TBAYAU)](https://codecov.io/gh/SebastianAment/UpdatableQRFactorizations.jl)

This package contains implementations of efficient representations and updating algorithms for QR factorizations.
Notably, the implementations can scale to very high dimensions and moderately large numbers of columns even if memory is not abundant.

## Basic Usage 
The following example highlights the basic usage of the package.
```julia
using UpdatableQRFactorizations
n, m = 32, 8
A = randn(n, m)
F = UpdatableQR(A)
```

The following snippet shows how to add a column to the factorization.
```julia
b = randn(n)
add_column!(F, b)
println(Matrix(F) ≈ [A b])
```

Lastly, we can also efficiently update the factorization upon removal of a column.
```julia
remove_column!(F)
println(Matrix(F) ≈ A)
```

## Documentation
### Initialization
The `UpdatableGivensQR` structure is central to the package and can also be referred to as `UpdatableQR`, or `UQR`.
It holds storage for Givens rotations and the `R` matrix of a QR factorization and allows for efficient addition and deletion of columns. 
See `add_column!` and `remove_column!`.
There are two primary ways of initializing such a structure.
First,
```julia
UpdatableQR(A::AbstractMatrix, r::Int = size(A, 1))
```
computes the QR factorization of `A` and pre-allocates enough memory for up to `r` columns total. 
In cases where `n` is large, and in particular, if an `n` by `n` matrix does not fit in memory, 
reducing the maximum allowed rank `r` is necessary to use this structure.
Note also that there is a mutating constructor 
```julia 
UpdatableQR!(A::AbstractMatrix, r::Int = size(A, 1)),
```
which overwrites the input matrix `A` with the `R` matrix but allocates extra space for the representation of `Q`.
Second,
```julia
UpdatableQR(n::Int, r::Int)
```
initializes an empty QR factorization of size `n` by `0` with enough space to add `r` columns total.

### Column Addition
To add columns to the factorization, use
```julia
add_column!(F::UpdatableGivensQR, x::AbstractVector, k::Int = size(F, 2) + 1).
```
Given an existing QR factorization `F` of a matrix, `add_column!` computes the factorization of
the same matrix after a new column `x` has been added as the `k`ᵗʰ column.
Computational complexity: O(nm) where size(F) = (n, m).
This overwrites the existing factorization.
The package also supports the addition of multiple columns with a single call with performance benefits:
```julia
add_column!(F::UpdatableGivensQR, X::AbstractMatrix, k::AbstractArray{Int} = (size(F, 2) + 1) : (size(F, 2) + size(X, 2)))
```
By default, both functions append the columns to the factorization. 
If a different column ordering is desired, the index at which the columns should be added can be passed as the third argument.

### Column Removal
To remove columns from a factorization, use
```julia
remove_column!(F::UpdatableGivensQR, k::Int = size(F, 2))
```
Updates the existing QR factorization `F` of a matrix to the factorization of
the same matrix after its kᵗʰ column has been deleted.
Computational complexity: O(m^2) where size(F) = (n, m).

## On Efficiency

First, note that the goal of this package is primarily to allow for the efficient updating of QR factorizations.
It is hard to beat `stdlib`'s `qr`, which links to LAPACK, for a factorization from scratch.
For a representative performance on a dual core 13'' MacBook Pro from 2017 for a moderately sized problem, see the following:
```julia
n, m = 1_000, 128;
A = randn(n, m);
b = randn(n);

@time F = UpdatableQR(A, 2m);
  0.018699 seconds (10 allocations: 10.241 MiB)
  
@time add_column!(F, b);
  0.000635 seconds (1 allocation: 1.141 KiB)
  
@time qr(A);
  0.006914 seconds (7 allocations: 1.047 MiB)

@time qr!(A);
  0.003037 seconds (5 allocations: 72.188 KiB)
```
Notably, `qr` is three times faster than `UpdatableQR` in constructing a factorization from scratch, but is significantly 
slower than the addition of a single column, which is not supported by `qr` and would require a second factorization from scratch.

## On Scalability
The implementation contained herein represents Q implicitly by keeping track of the Givens rotations that constitute it,
allowing a scaling to very high `n` and moderate `m`.
This is in contrast to an existing implementation of updatable QR factorizations in [GeneralQP.jl](https://github.com/oxfordcontrol/GeneralQP.jl/blob/master/src/linear_algebra.jl
),
which keeps the `n` by `n` Q matrix densely in memory and updates it upon column addition, prohibiting scaling to problems where `n` is large regardless of the number of columns `m`.


In fact, on the same laptop as before, we can compute an updatable QR factorization in 1,000,000 dimensions in under a minute:
```julia
n, m = 1_000_000, 128;
A = randn(n, m);
@time F = UpdatableQR(A, 2m);
39.086453 seconds (10 allocations: 10.490 GiB, 0.74% gc time)
```
It is also notable that contructing the QR factorization from scratch with `stdlib`'s `qr` is again much faster
```julia
@time qr(A);
  3.363778 seconds (112.75 k allocations: 983.465 MiB, 1.77% gc time, 1.34% compilation time).
 
@time qr!(A);
  3.058636 seconds (5 allocations: 72.188 KiB)
```
However, subsequent addition of single columns beats calculating a factorization from scratch, even with the very efficient `qr`
```julia
@time add_column!(F, b);
  0.599013 seconds (1 allocation: 1.141 KiB)
```
Future work on this package could improve on the performance difference between `qr` and `UpdatableQR` for a factorization from scratch.

## Limitations
At this point, UpdatableQRFactorizations.jl only supports factorizations of matrices with full column rank. 
Further, no specialization for sparse matrices has been implemented, which might significantly improve performance in cases where sparsity is present. 

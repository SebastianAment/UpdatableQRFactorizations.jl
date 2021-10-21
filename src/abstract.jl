############################# abstract QR type #################################
abstract type AbstractQR{T} <: Factorization{T} end
# AbstractQR assumes existence of n, m, Q, R fields
Base.size(F::AbstractQR) = (F.n, F.m)
Base.size(F::AbstractQR, i::Int) = i > 2 ? 1 : size(F)[i]
Base.eltype(F::AbstractQR{T}) where T = T

Base.:\(F::AbstractQR, x::AbstractVecOrMat) = ldiv!(F, copy(x))

# some helpers
"""
```
    number_of_rotations(n::Int, m::Int)
```
Computes the number of Givens rotations that are necessary to compute the QR
factorization of a general matrix of size n by m.
"""
number_of_rotations(n::Int, m::Int) = n*m - (m*(m+1)) ÷ 2
number_of_rotations(A::AbstractMatOrFac) = number_of_rotations(size(A)...)
# min_nm = min(m, m) # IDEA: for overdetermined systems

"""
```
    allocate_rotations(T::DataType, n::Int, m::Int)
```
Allocates a vector of Givens rotations of a length that is necessary to compute
the QR factorization of a general matrix of size n by m.
"""
function allocate_rotations(T::DataType, n::Int, m::Int)
    Vector{Givens{T}}(undef, number_of_rotations(n, m))
end
function allocate_rotations(A::AbstractMatOrFac)
    allocate_rotations(eltype(A), size(A)...)
end

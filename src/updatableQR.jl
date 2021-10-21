"""
```
    UpdatableGivensQR <: AbstractQR <: Factorization
```
Holds storage for Givens rotations and the R matrix for a QR factorization.
Allows for efficient addition and deletion of columns.
See `add_column!` and `remove_column!`.
"""
mutable struct UpdatableGivensQR{T, QFT, RFT<:AbstractMatrix{T}, QX, PT} <: AbstractQR{T}
    rotations_full::QFT
    rot_index::Int # need to track it as varying numbers of deletions can create an unpredictible number of rotations
    R_full::RFT
    Qx::QX # holds space for Q'*x when adding columns
    perm_full::PT # allows efficient insertion of column at arbitrary indices, while only appending columns in memory and keeping track of the permutations
    n::Int
    m::Int
end
const UGQR = UpdatableGivensQR
const UpdatableQR = UpdatableGivensQR
const UQR = UpdatableQR

function UpdatableGivensQR(A::AbstractMatrix, r::Int = size(A, 1))
    UpdatableGivensQR!(copy(A), r)
end

function UpdatableGivensQR!(A::AbstractMatrix, r::Int = size(A, 1))
    n, m = size(A)
    T = eltype(A)
    rotations_full = allocate_rotations(T, n, r)
    rot_index = number_of_rotations(n, m)
    rotations = @view rotations_full[1:rot_index]
    R_full = zeros(T, r, r)
    F = GivensQR!(A, rotations)
    @. R_full[1:m, 1:m] = F.R
    Qx = zeros(T, n)
    perm_full = collect(1:r)
    UpdatableGivensQR(rotations_full, rot_index, R_full, Qx, perm_full, n, m)
end
# copying underlying data so that future changes in F don't change GivensQR structure
GivensQR(F::UpdatableGivensQR) = GivensQR(copy(F.Q), copy(F.R), F.n, F.m)

function Base.getproperty(F::UpdatableGivensQR, s::Symbol)
    if s == :Q
        GivensQ(F.rotations, F.n, F.m)
    elseif s == :rotations
        @view F.rotations_full[1:F.rot_index]
    elseif s == :R
        R = @view F.R_full[1:F.m, 1:F.m]
        UpperTriangular(R)
    elseif s == :perm
        @view F.perm_full[1:F.m]
    else
        getfield(F, s)
    end
end
Base.eltype(F::UGQR{T}) where T = T
Base.size(F::UGQR) = (F.n, F.m)
Base.size(F::UGQR, i::Int) = i > 2 ? 1 : size(F)[i]
# Base.Matrix(F::UGQR) = Matrix(F.uqr)[:, invperm(F.perm)] F.Q * F.R
# Base.AbstractMatrix(F::UGQR) = Matrix(F)

function Base.Matrix(F::UQR)
    n, m = size(F)
    A = zeros(eltype(F), n, m)
    @. A[1:m, 1:m] = F.R
    lmul!(F.Q, A) # avoids every explicitly creating F.Q
    ip = invperm(F.perm) # this allocates, IDEA: could pre-allocate
    for c in eachrow(A) # A[:, invperm(F.perm)]
        permute!(c, ip)
    end
    return A
end

# overwrites the first m elements of x with the solution
function LinearAlgebra.ldiv!(F::UQR, x::AbstractVecOrMat)
    ldiv!(F.Q, x)
    xm = typeof(x) <: AbstractVector ? view(x, 1:F.m) : view(x, 1:F.m, :)
    ldiv!(F.R, xm)
    p = invperm(F.perm)
    for xi in eachcol(xm)
        permute!(xi, p)
    end
    return xm
end

"""
```
    add_column!(F::UpdatableGivensQR, x::AbstractVector, k::Int = size(F, 2) + 1)
```
Given an existing QR factorization `F` of a matrix, computes the factorization of
the same matrix after a new column `x` has been added as the `k`ᵗʰ column.
WARNING: Overwrites existing factorization.
"""
function add_column!(F::UpdatableGivensQR, x::AbstractVector, k::Int = size(F, 2) + 1)
    length(x) == size(F, 1) || throw(DimensionMismatch("length of input not equal to first dimension of UpdatableQR factorization"))
    for (i, p) in enumerate(F.perm)
        if p ≥ k # incrementing indices above k
            F.perm[i] = p+1
        end
    end
    F.perm_full[F.m+1] = k
    @. F.Qx = x # so that input doesn't get mutated
    F.rot_index = append_column!(F.Qx, F.rotations_full, F.rot_index, F.R_full, F.m)
    F.m += 1
    return F
end

# x is column to be appended to factorization
# rotations is a vector of Givens rotations,
# R is matrix where R[1:m, 1:m] has already been filled with the factorization of the previously added columns
function append_column!(x::AbstractVector, rotations::AbstractVector{<:Givens},
                        rot_index::Int, R::AbstractMatrix, m::Int)
    n = length(x)
    size(R, 2) > m || throw("cannot add column to factorization with maximum rank $(size(R, 2))")
    Q = GivensQ(@view(rotations[1:rot_index]), n, m)
    Qx = lmul!(Q', x) # overwriting x
    for i in n:-1:m+2 # zero out the "spike"
        G, r = givens(Qx[i-1], Qx[i], i-1, i)
        lmul!(G, Qx)
        rot_index += 1
        rotations[rot_index] = G
    end
    @. R[1:m+1, m+1] = Qx[1:m+1] # only necessary if these are not already equal in memory
    return rot_index
end


"""
remove_column!(F::UpdatableGivensQR, k::Int = size(F, 2))

Updates the existing QR factorization F of a matrix to the factorization of
the same matrix after its kth column has been deleted.
"""
function remove_column!(F::UpdatableGivensQR, k::Int = size(F, 2))
    i = findfirst(==(k), F.perm)
    isnothing(i) && throw(DimensionMismatch("index k = $k not found in F.perm, can't remove the associated column"))
    perm = F.perm
    for j in i+1:F.m # shifts all indices above k down to keep active indices adjacent in memory
        perm[j-1] = perm[j]
    end
    for (i, p) in enumerate(perm) # decrements indices above k (since we are removing it)
        if p ≥ k
            perm[i] = p-1
        end
    end
    rotations = @view F.rotations_full[F.rot_index+1:end]
    F.rot_index += remove_column!(F.R, rotations, i)
    F.m -= 1
    return F
end

# NOTE: rotation just have to be empty space for new rotations to be added
function remove_column!(R::AbstractMatrix, rotations::AbstractVector{<:Givens}, k::Int = size(F, 2))
    m = size(R, 2)
    1 <= k <= m || throw(DimensionMismatch("index $k not in range [1, $m]"))
    length(rotations) ≥ m-k || throw(DimensionMismatch("not enough allocated space in rotations vector to remove column"))
    rot_index = 0
    for i in k+1:m # zero out subdiagonal of submatrix following kth column
        Ri = @view R[:, i:end]
        G, rho = givens(R[i-1, i], R[i, i], i-1, i)
        lmul!(G, Ri)
        rot_index += 1
        rotations[rot_index] = G
    end
    @inbounds for j in k:m-1 # overwrite deleted column
        @simd for i in 1:j # guarantees we only access upper triangular part
            R[i, j] = R[i, j+1]
        end
    end
    return rot_index # this is generally m-k, important to increment the rot_index of the factorization
end

# function LinearAlgebra.ldiv!(y::AbstractVector, F::UGQR, x::AbstractVector)
#     ldiv!(y, F.uqr, x)
#     invpermute!(y, P.perm) # I believe this still allocates the invperm
#     #invpermute!!(y, P.perm) # this changes P.perm!
# end
# function LinearAlgebra.ldiv!(F::UGQR, x::AbstractVector)
#     ldiv!(y, P.uqr, x)
#     invpermute!(y, P.perm) # I believe this still allocates the invperm
#     #invpermute!!(y, P.perm) # this changes P.perm!
# end

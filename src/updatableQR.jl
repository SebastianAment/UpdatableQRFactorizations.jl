"""

    UpdatableGivensQR <: AbstractQR <: Factorization

Holds storage for Givens rotations and the R matrix for a QR factorization.
Allows for efficient addition and deletion of columns.
See `add_column!` and `remove_column!`.
"""
mutable struct UpdatableGivensQR{T, QFT, RFT<:AbstractMatrix{T}, QXT, PT} <: AbstractQR{T}
    rotations_full::QFT
    rot_index::Int # need to track it as varying numbers of deletions can create an unpredictible number of rotations
    R_full::RFT
    QX::QXT # holds space for Q'*x when adding columns
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

# memory footprint: O(nr + r^2)
function UpdatableGivensQR!(A::AbstractMatrix, r::Int = size(A, 1))
    n, m = size(A)
    n ≥ r || throw("maximum rank r = $r has to be smaller or equal to n = $n")
    r ≥ m || throw("maximum rank r = $r has to be larger or equal to m = $m")
    T = eltype(A)
    rotations_full = allocate_rotations(T, n, r)
    rot_index = number_of_rotations(n, m)
    rotations = @view rotations_full[1:rot_index]
    R_full = zeros(T, r, r)
    F = GivensQR!(A, rotations)
    @. R_full[1:m, 1:m] = F.R
    QX = zeros(T, n, r) # stores Q'*X for addition of multiple columns X
    perm_full = collect(1:r)
    UpdatableGivensQR(rotations_full, rot_index, R_full, QX, perm_full, n, m)
end
# constructor that pre-allocates memory without starting a qr factorization
UpdatableGivensQR(n::Int, r::Int) = UpdatableGivensQR(Float64, n, r)
function UpdatableGivensQR(T::DataType, n::Int, r::Int)
    n ≥ r || throw("maximum rank r = $r has to be smaller or equal to n = $n")
    rotations_full = allocate_rotations(T, n, r)
    rot_index = 0
    R_full = zeros(T, r, r)
    QX = zeros(T, n, r)
    perm_full = collect(1:r)
    m = 0
    UpdatableGivensQR(rotations_full, rot_index, R_full, QX, perm_full, n, m)
end

# copying underlying data so that future changes in F don't change GivensQR structure
GivensQR(F::UpdatableGivensQR) = GivensQR(copy(F.Q), copy(F.R), F.n, F.m)
function Base.copy(F::UpdatableGivensQR)
    UpdatableGivensQR(copy(F.rotations_full), F.rot_index, copy(F.R_full),
                      copy(F.Qx), copy(F.perm_full), F.n, F.m)
end

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
Base.AbstractMatrix(F::UGQR) = Matrix(F)
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

    add_column!(F::UpdatableGivensQR, x::AbstractVector, k::Int = size(F, 2) + 1)

Given an existing QR factorization `F` of a matrix, computes the factorization of
the same matrix after a new column `x` has been added as the `k`ᵗʰ column.
Computational complexity: O(nm) where size(F) = (n, m).
WARNING: Overwrites existing factorization.
"""
function add_column!(F::UpdatableGivensQR, x::AbstractVector, k::Int = size(F, 2) + 1)
    Qx = @view F.QX[:, 1]
    @. Qx = x # so that x doesn't get overwritten
    add_column!!(F, Qx, k)
end
function add_column!(F::UpdatableGivensQR, X::AbstractMatrix, k::AbstractArray{Int} = (size(F, 2) + 1) : (size(F, 2) + size(X, 2)))
    QX = @view F.QX[:, 1:length(k)]
    @. QX = X # so that X doesn't get overwritten
    add_column!!(F, QX, k)
end

"""
Same as `add_column!` but WARNING: mutates `x`!
"""
function add_column!!(F::UpdatableGivensQR, x::AbstractVector, k::Int = size(F, 2) + 1)
    length(x) == size(F, 1) || throw(DimensionMismatch("length of input not equal to first dimension of UpdatableQR factorization"))
    add_permutation!(F, k)
    F.rot_index = append_column!(x, F.rotations_full, F.rot_index, F.R_full, F.m)
    F.m += 1
    return F
end

# adding columns in X to factorization, mutates X!
function add_column!!(F::UpdatableGivensQR, X::AbstractMatrix, k::AbstractArray = (size(F, 2) + 1) : (size(F, 2) + size(X, 2)))
    size(X, 1) == size(F, 1) || throw(DimensionMismatch("length of input not equal to first dimension of UpdatableQR factorization"))
    size(X, 2) == length(k) || throw(DimensionMismatch("length of index array k ($(length(k))) not equal to second dimension of data matrix X ($(size(X, 2)))"))
    add_permutation!(F, k)
    F.rot_index = append_column!(X, F.rotations_full, F.rot_index, F.R_full, F.m)
    F.m += length(k)
    return F
end

# X are columns to be appended to the factorization
# rotations is a vector of Givens rotations,
# R is matrix where R[1:m, 1:m] has already been filled with the factorization of the previously added columns
function append_column!(X::AbstractVecOrMat, rotations::AbstractVector{<:Givens},
                        rot_index::Int, R::AbstractMatrix, m::Int)
    n, k = size(X, 1), size(X, 2) # if X is a vector, size(X, 2) = 1
    size(R, 2) > m || throw("cannot add another column to factorization with current rank $m and maximum rank $(size(R, 2))")

    # check if there's enough space in rotations left to carry out this addition
    ensure_space_to_append_column!(rotations, rot_index, n, m, k)

    Q = GivensQ(@view(rotations[1:rot_index]), n, m)
    QX = lmul!(Q', X) # overwriting X
    for j in 1:k
        QXj = @view QX[:, j:end] # do not need to treat columns 1:j-1 further
        # for i in n:-1:m+2 # zero out the "spike"
        for i in n:-1:m+1+j # zero out the "spike"
            G, r = givens(QXj[i-1, 1], QXj[i, 1], i-1, i)
            lmul!(G, QXj)
            rot_index += 1
            rotations[rot_index] = G
        end
    end
    @. R[1:m+k, m+1:m+k] = QX[1:m+k, :] # only necessary if these are not already equal in memory
    return rot_index
end

function add_permutation!(F::UpdatableQR, k::Union{Int, AbstractVector{Int}})
    F.m + length(k) ≤ length(F.perm_full) || throw("cannot add $(length(k)) columns to factorization with current rank $(F.m) and maximum rank $(length(F.perm_full))")
    for (i, p) in enumerate(F.perm)
        for ki in k
            if p ≥ ki # incrementing indices above ki
                F.perm[i] += 1
            end
        end
    end
    F.perm_full[F.m+1:F.m+length(k)] .= k
    return F
end

# current factorization size is (n, m), number of columns to be added is k
function ensure_space_to_append_column!(rotations::AbstractVector{<:Givens},
                        rot_index::Int, n::Int, m::Int, k::Int = 1, verbose::Bool = false)
    nrot = length(rotations) - rot_index # number of rotations left to add in rotations vector
    nrot_to_add = number_of_rotations_to_append_column(n, m, k)
    if nrot < nrot_to_add
        verbose && println("INFO: adding more memory for Givens rotations in append_column!")
        append!(rotations, Vector{eltype(rotations)}(undef, nrot_to_add - nrot))
    end
    return rotations
end

"""

remove_column!(F::UpdatableGivensQR, k::Int = size(F, 2))

Updates the existing QR factorization `F` of a matrix to the factorization of
the same matrix after its kᵗʰ column has been deleted.
Computational complexity: O(m²) where size(F) = (n, m).
"""
function remove_column!(F::UpdatableGivensQR, k::Int = size(F, 2))
    i = remove_permutation!(F, k)
    F.rot_index = remove_column!(F.R, F.rotations_full, F.rot_index, i)
    F.m -= 1
    return F
end

function remove_column!(R::AbstractMatrix, rotations::AbstractVector{<:Givens},
                        rot_index::Int, k::Int = size(F, 2))
    m = size(R, 2)
    1 <= k <= m || throw(DimensionMismatch("index $k not in range [1, $m]"))

    # check if there's enough space in rotations left to carry out this removal
    ensure_space_to_remove_column!(rotations, rot_index, m, k)

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

# returns the index i of the column in R that is referenced by the kth column of the factorization
# those are not equal because of the permutation of the columns given by F.perm
function remove_permutation!(F::UpdatableQR, k::Int)
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
    return i
end

function ensure_space_to_remove_column!(rotations::AbstractVector{<:Givens},
                        rot_index::Int, m::Int, k::Int, verbose::Bool = false)
    nrot = length(rotations) - rot_index
    nrot_to_remove = number_of_rotations_to_remove_column(m, k)
    if nrot < nrot_to_remove
        verbose && println("INFO: adding more memory for Givens rotations in remove_column!")
        append!(rotations, Vector{eltype(rotations)}(undef, nrot_to_remove-nrot))
    end
    return rotations
end

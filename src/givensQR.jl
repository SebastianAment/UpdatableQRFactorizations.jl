#################### regular (non-updatable) QR structure ######################
struct GivensQR{T, QT, RT<:AbstractMatrix{T}} <: AbstractQR{T}
    Q::QT
    R::RT
    n::Int
    m::Int
end

GivensQR(A::AbstractMatrix) = GivensQR!(copy(A))
# overwrites A
GivensQR!(A::AbstractMatrix) = GivensQR!(A, allocate_rotations(A))
# overwrites A and rotations
function GivensQR!(A::AbstractMatrix, rotations::AbstractVector{<:Givens})
    n, m = size(A)
    n ≥ m || throw(DimensionMismatch("GivensQR only supports systems with full column rank."))
    rot_index = 1
    for j in 1:m
        Aj = @view A[:, j:end] # do not need to treat columns 1:j-1 further
        @inbounds for i in n:-1:j+1
            G, rho = givens(Aj[i-1, 1], Aj[i, 1], i-1, i)
            lmul!(G, Aj)
            rotations[rot_index] = G
            rot_index += 1
        end
        # Aj = @view A[:, j] # the above is more efficient if constructing an entire factorization (probably due to BLAS3)
        # rot_index = append_column!(Aj, rotations, rot_index, A, j-1)
    end
    Q = GivensQ(rotations, n, m)
    R = UpperTriangular(@view(A[1:m, 1:m]))
    return GivensQR(Q, R, n, m)
end

Base.copy(F::GivensQR) = GivensQR(copy(F.Q), copy(F.R), F.n, F.m)
function Base.Matrix(F::GivensQR)
    n, m = size(F)
    A = zeros(eltype(F), n, m)
    @. A[1:m, 1:m] = F.R
    return lmul!(F.Q, A) # avoids every explicitly creating F.Q
end

# overwrites the first m elements of x with the solution
function LinearAlgebra.ldiv!(F::GivensQR, x::AbstractVecOrMat)
    ldiv!(F.Q, x)
    xm = typeof(x) <: AbstractVector ? view(x, 1:F.m) : view(x, 1:F.m, :)
    ldiv!(F.R, xm)
    return xm
end

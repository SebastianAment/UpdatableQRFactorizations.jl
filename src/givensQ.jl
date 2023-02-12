# Givens representation of Q matrix, memory efficient compared to dense representation
struct GivensQ{T, QT<:AbstractVector{<:Givens{T}}} <: AbstractQ{T}
    rotations::QT
    n::Int
    m::Int
end

# r controls how many columns of Q we instantiate
# NOTE: this also works if rotations is empty (returns identity matrix)
function Base.Matrix(F::GivensQ, r::Int = size(F, 2))
    n, m = size(F)
    Q = Matrix(one(eltype(F))*I, n, r)
    return lmul!(F, Q)
end
Base.size(F::GivensQ) = (F.n, F.n) # could be square but also (n, m)?
Base.size(F::GivensQ, i::Int) = i > 2 ? 1 : size(F)[i]
Base.copy(F::GivensQ) = GivensQ(copy(F.rotations), F.n, F.m)
if !isdefined(LinearAlgebra, :AdjointQ) # VERSION < v"1.10-"
    Base.adjoint(F::GivensQ) = Adjoint(F)
end
# NOTE: this also works if rotations is empty (is identity operator)
function LinearAlgebra.lmul!(F::GivensQ, X::AbstractVecOrMat)
    for G in Iterators.reverse(F.rotations) # the adjoint of a GivensQ reverses the order of the rotations
        lmul!(G', X) # ... and takes their adjoint
    end
    return X
end
function LinearAlgebra.lmul!(A::AdjointQ{<:Any, <:GivensQ}, X::AbstractVecOrMat)
    F = parent(A)
    for G in F.rotations # the adjoint of a GivensQ reverses the order of the rotations
        lmul!(G, X) # ... and takes their adjoint
    end
    return X
end
LinearAlgebra.rmul!(X::AbstractVecOrMat, F::GivensQ) = lmul!(F', X')'
LinearAlgebra.rmul!(X::AbstractVecOrMat, F::AdjointQ{<:Any, <:GivensQ}) = lmul!(F', X')'

function Base.:*(F::GivensQ, G::AdjointQ{<:Any, <:GivensQ})
    T = promote_type(eltype(F), eltype(G))
    F === G' ? (one(T) * I)(F.n) : F * Matrix(G)
end
function Base.:*(F::AdjointQ{<:Any, <:GivensQ}, G::GivensQ)
    T = promote_type(eltype(F), eltype(G))
    F' === G ? (one(T) * I)(G.n) : F * Matrix(G)
end
function Base.:(==)(F::GivensQ, G::GivensQ)
    F.rotations == G.rotations && F.n == G.n && F.m == G.m
end

if VERSION < v"1.10-"
    lmul(F, X) = lmul!(F, copy(X))
    rmul(X, F) = rmul!(copy(X), F)

    Base.:*(F::GivensQ, X::AbstractVector) = lmul(F, X)
    Base.:*(F::GivensQ, X::AbstractMatrix) = lmul(F, X)
    Base.:*(F::AdjointQ{<:Any, <:GivensQ}, X::AbstractVector) = lmul(F, X)
    Base.:*(F::AdjointQ{<:Any, <:GivensQ}, X::AbstractMatrix) = lmul(F, X)

    Base.:*(X::AbstractMatrix, F::GivensQ) = rmul(X, F)
    Base.:*(X::AbstractMatrix, F::AdjointQ{<:Any, <:GivensQ}) = rmul(X, F)

    Base.:*(F::GivensQ, X::StridedVector) = lmul(F, X)
    Base.:*(F::GivensQ, X::StridedMatrix) = lmul(F, X)
    Base.:*(F::GivensQ, X::Adjoint{<:Any, <:StridedVecOrMat}) = lmul(F, X)
    Base.:*(F::Adjoint{<:Any, <:GivensQ}, X::StridedVector) = lmul(F, X)
    Base.:*(F::Adjoint{<:Any, <:GivensQ}, X::StridedMatrix) = lmul(F, X)
    Base.:*(F::Adjoint{<:Any, <:GivensQ}, X::Adjoint{<:Any, <:StridedVecOrMat}) = lmul(F, X)

    Base.:*(X::StridedMatrix, F::GivensQ) = rmul(X, F)
    Base.:*(X::StridedMatrix, F::AdjointQ{<:Any, <:GivensQ}) = rmul(X, F)

    LinearAlgebra.mul!(Y::StridedVector, F::GivensQ, X::StridedVector) =
        invoke(LinearAlgebra.mul!, Tuple{AbstractVector, GivensQ, AbstractVector}, Y, F, X)
    LinearAlgebra.mul!(Y::StridedMatrix, F::GivensQ, X::StridedMatrix) =
        invoke(LinearAlgebra.mul!, Tuple{AbstractMatrix, GivensQ, AbstractMatrix}, Y, F, X)
    LinearAlgebra.mul!(Y::StridedVector, F::AdjointQ{<:Any, <:GivensQ}, X::StridedVector) =
        invoke(LinearAlgebra.mul!, Tuple{AbstractVector, AdjointQ{<:Any, <:GivensQ}, AbstractVector}, Y, F, X)
    LinearAlgebra.mul!(Y::StridedMatrix, F::AdjointQ{<:Any, <:GivensQ}, X::StridedMatrix) =
        invoke(LinearAlgebra.mul!, Tuple{AbstractMatrix, AdjointQ{<:Any, <:GivensQ}, AbstractMatrix}, Y, F, X)

    LinearAlgebra.ldiv!(F::GivensQ, x::AbstractVector) = lmul!(F', x)
    LinearAlgebra.ldiv!(F::AdjointQ{<:Any, <:GivensQ}, x::AbstractVector) = lmul!(F', x)
    LinearAlgebra.ldiv!(F::GivensQ, X::AbstractMatrix) = lmul!(F', X)
    LinearAlgebra.ldiv!(F::AdjointQ{<:Any, <:GivensQ}, X::AbstractMatrix) = lmul!(F', X)

    function LinearAlgebra.mul!(y::AbstractVector, F::GivensQ, x::AbstractVector)
        @. y = x
        lmul!(F, y)
    end
    function LinearAlgebra.mul!(Y::AbstractMatrix, F::GivensQ, X::AbstractMatrix)
        @. Y = X
        lmul!(F, Y)
    end
    function LinearAlgebra.mul!(y::AbstractVector, F::AdjointQ{<:Any, <:GivensQ}, x::AbstractVector)
        @. y = x
        lmul!(F, y)
    end
    function LinearAlgebra.mul!(Y::AbstractMatrix, F::AdjointQ{<:Any, <:GivensQ}, X::AbstractMatrix)
        @. Y = X
        lmul!(F, Y)
    end    
end

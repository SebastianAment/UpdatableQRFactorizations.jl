module TestUpdatableQRFactorizations
using LinearAlgebra
using LinearAlgebra: Givens
using Test
using UpdatableQRFactorizations

@testset "UpdatableQRFactorizations" begin

    n, m = 32, 8 # NOTE: a test of UpdatableQR assumes n = 4m
    A = randn(n, m)
    F = GivensQR(A)
    @test F isa GivensQR

    # testing size
    @test size(F) == (n, m)
    for i in 1:3
        @test size(F, 1) == size(A, 1)
    end

    # testing copy
    CF = copy(F)
    @test CF.Q == F.Q
    @test CF.R == F.R
    @test size(CF) == size(F)

    @testset "GivensQ" begin
        Q = Matrix(F.Q)

        @test Q'Q ≈ I(n)
        @test Q*Q' ≈ I(n)

        @test F.Q'*F.Q == I(n)
        @test F.Q'*F.Q == I(n)

        x = randn(n)
        Qx = Q*x
        @test Qx ≈ F.Q*x
        @test F.Q \ Qx ≈ x
        Qtx = Q'*x
        @test Qtx ≈ F.Q'*x
        @test F.Q' \ Qtx ≈ x

        r = 4
        X = randn(r, n)
        @test Q*X' ≈ F.Q*X'
        @test X*Q ≈ X*F.Q
        @test Q'*X' ≈ F.Q'*X'
        @test X*Q' ≈ X*F.Q'
        @test Q \ (Q*X') ≈ X'
        @test Q' \ (Q'*X') ≈ X'

        x = randn(n)
        y = similar(x)
        X = randn(n, r)
        Y = similar(X)
        @test mul!(y, F.Q, x) ≈ Q*x
        @test mul!(Y, F.Q, X) ≈ Q*X
        @test mul!(y, F.Q', x) ≈ Q'*x
        @test mul!(Y, F.Q', X) ≈ Q'*X

        # GivensQ with empty rotations vector behaves like identity
        Qid = GivensQ(Vector{Givens{Float64}}(undef, 0), n, m)
        @. y = x
        @test lmul!(Qid, x) ≈ y
        @test lmul!(Qid', x) ≈ y
        @test Matrix(Qid) ≈ I(n)
    end

    @testset "GivensQR" begin
        QA = F.Q'*A
        @test UpperTriangular(QA[1:m, 1:m]) ≈ F.R
        @test Matrix(F) ≈ A

        x = randn(m)
        Ax = A*x
        @test F \ (A*x) ≈ x
        @test F \ complex(A*x) ≈ x
        r = 4
        X = randn(m, r)
        @test F \ (A*X) ≈ X
        @test F \ complex(A*X) ≈ X
    end

    @testset "UpdatableQR" begin
        F = UpdatableQR(A)
        @test Matrix(F) ≈ A

        x = randn(m)
        Ax = A*x
        @test F \ (A*x) ≈ x
        r = 4
        X = randn(m, r)
        @test F \ (A*X) ≈ X

        # testing column addition
        F = UpdatableQR(A)
        b = randn(n)
        add_column!(F, b)
        @test Matrix(F) ≈ [A b]

        # testing column addition at particular index
        x = randn(m+1)
        X = randn(m+1, r)
        for i in 1:m
            F = UpdatableQR(A)
            add_column!(F, b, i)
            B = [A[:, 1:i-1] b A[:, i:end]]
            @test B ≈ Matrix(F)
            Q = Matrix(F.Q)
            @test Q'Q ≈ I(n) # maintains orthogonality

            # testing solves
            @test F \ (B*x) ≈ x
            r = 4
            @test F \ (B*X) ≈ X
        end

        # testing column addition
        F = UpdatableQR(A)
        remove_column!(F)
        @test Matrix(F) ≈ A[:, 1:m-1]

        # testing column removal
        for i in 1:m
            F = UpdatableQR(A)
            remove_column!(F, i)
            B = @view A[:, 1:m .!= i]
            @test B ≈ Matrix(F)
            Q = Matrix(F.Q)
            @test Q'Q ≈ I(n) # maintains orthogonality
        end

        # testing empty initialization
        F = UpdatableQR(n, m)
        @test size(F) == (n, 0)
        @test Matrix(F) ≈ zeros(n, 0)

        order = reverse(1:m) # adding columns of A in reverse order (but to front of factorization)
        for i in order
            add_column!(F, A[:, i], 1)
        end
        @test Matrix(F) ≈ A

        F = UpdatableQR(n, m)
        add_column!(F, A) # adding all columns in A simultaneously
        @test Matrix(F) ≈ A

        # try out full rank QR
        F = UpdatableQR(n, n) # NOTE: assumes n = 4m
        add_column!(F, A)
        add_column!(F, A)
        add_column!(F, A)
        add_column!(F, A)
        M = Matrix(F)
        @test size(M) == (n, n)
        for i in 1:4
            Mi = @view M[:, m*(i-1)+1:m*(i-1)+m]
            @test Mi ≈ A
        end
    end

    # multiply is very efficient!
    # @time Q*x
    # @time Q*x
    # @time F.Q*x
    # @time F.Q*x

    # IDEA: testing overdetermined system, irrelevant for now
    # n, m = 128, 64
    # A = randn(m, n)
    # @time F = GivensQR(A)

    # A = randn(n, m)
    # rotations = allocate_rotations(A)
    # F = GivensQR()
    # for i in 1:m
    #     Ai = @view A[:, 1:i]
    #     append_column!()
    # end
end

end # module

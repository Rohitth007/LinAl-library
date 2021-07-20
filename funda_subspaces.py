# Gives Subspaces of a matrix A
# Did not impliment Column Space Basis Matrix
# as exchanges have to be tracked instead used A_trans
#
# Improvments:
# Find Gauss Jordan first and track exchanges to do elimination only once
# instead of thrice.
import numpy as np
from get_matrix import get_matrix
from rref import rref
from gauss_jordan import gj


def main():

    A = get_matrix()

    rank, Rb, Cb, dim_nullspace, N, dim_leftnullspace, lN = subspaces(A)

    print("\n\nRank of Matrix A = Dim(columnspace-C(A)) = Dim(rowspace-C(A_t)) =", rank)
    print('\n Rowspace Basis Matrix, Rb:\n', Rb)
    print('\n Columnspace Basis Matrix, Cb:\n', Cb)
    print('\nDim(nullspace-N(A)) = n-r = ', dim_nullspace)
    print('\n Nullspace Basis Matrix, N:\n', N)
    print('\nDim(left-nullspace-N(A_t)) = m-r = ', dim_leftnullspace)
    print('\n Left-Nullspace Basis Matrix, lN:\n', lN)


def subspaces(A, verbose=True, dtype=np.float64):
    rows, cols = A.shape

    R, pivots = rref(A, returnPivot=True, verbose=verbose)
    if verbose:
        print('\nReduced Row Echelon Form:\n', R)

    rank = len(pivots)
    dim_nullspace = cols - rank
    dim_leftnullspace = rows - rank

    Rb = rowspace_basis_matrix(R, rank)
    N = nullspace_matrix(R, dim_nullspace, pivots, dtype=dtype)

    At = np.transpose(A)
    Rt = rref(At, verbose=False)
    Cb = rowspace_basis_matrix(Rt, rank)
    lN = nullspace_matrix(Rt, dim_leftnullspace, pivots, dtype=dtype)

    # Use below to find left-null-space using Gauss Jordan
    # lN = left_nullspace_matrix(A, dim_leftnullspace)

    return rank, Rb, Cb, dim_nullspace, N, dim_leftnullspace, lN


def left_nullspace_matrix(A, dim_leftnullspace):
    rows, cols = A.shape
    if dim_leftnullspace == 0:
        lN = 0
    else:
        # Gauss-Jordan on any matrix gives the Elimination Matrix
        E = gj(A, verbose=False)
        lN = np.empty((rows, 0))
        for i in range(rows - 1, rows - dim_leftnullspace - 1, -1):
            ln = np.reshape(E[i, :], (rows, 1))
            lN = np.append(lN, ln, axis=1)
    return lN


def nullspace_matrix(R, dim_nullspace, pivots, dtype=np.float64):
    rank = len(pivots)
    if dim_nullspace == 0:
        N = 0
    else:
        rows, cols = R.shape
        N = np.empty((cols, 0))
        I = np.identity(dim_nullspace)
        for j in range(cols):
            n = np.zeros((cols, 1), dtype=dtype)
            if j not in pivots:  # free column
                for i, pivot in enumerate(pivots):
                    # First element goes to the position of the first pivot and so on
                    n[pivot] = -R[i, j]
                n[j] = 1   # The current columns position in 'n' gets a one.
                N = np.append(N, n, axis=1)
    return N


def rowspace_basis_matrix(R, rank):
    rows, cols = R.shape
    Rb = np.empty((cols, 0))
    for i in range(rank):
        b = np.reshape(R[i, :], (cols, 1))
        Rb = np.append(Rb, b, axis=1)
    return Rb


if __name__ == '__main__':
    main()

# Finds Ax=b, given A and b if solution exists
import numpy as np
from get_matrix import get_matrix
from rref import rref
from funda_subspaces import nullspace_matrix


def main():
    A = get_matrix("Input Matrix A")
    q = input("Is Matrix Orthogonal, 'y' or 'n'? ")
    b = get_matrix("Input Vector b as Matrix")

    if q:  # Orthogonal Case Easy
        xp = np.transpose(A)@b
        N = 0
    else:
        aug = np.concatenate((A, b), axis=1)
        xp, N = solution(aug)

    print("\nParticular Solution:\n", xp)
    print("\nGeneral Solution:\n", N)


def solution(aug, verbose=True):
    rows, cols = aug.shape
    cols -= 1  # Last column is b
    R_aug, pivots = rref(aug, returnPivot=True, verbose=False)
    rank = len(pivots)

    # Rank of aug becomes more than columns of A
    # if b is also independent
    try:
        N = nullspace_matrix(R_aug[:, :cols], cols - rank, pivots)
    except:
        print("b is not in C(A), try using Least Squares")
        return None

    if verbose:
        print("\nAx=b\nx = xp + xn")
        if rank == rows:
            if rank == cols:   # A is invertible
                print("Unique solution exists for every b")
                # Can also be done using Inverse
                # Can also be done by making A orthogonal.
                # Can also be done using Crammer's Rule (Computationally expensive)
            else:              # Full row rank
                print("Infinite solutions for every b")
        elif rank == cols:     # Full column rank
            print("Unique solution if b is in columnspace of A")
        else:
            print("Infinite solutions if b is in columnspace of A")

    xp = particular_solution(R_aug, pivots)
    return xp, N


def particular_solution(R_aug, pivots):
    rows, cols = R_aug.shape
    cols -= 1  # Last column is b
    rank = len(pivots)
    xp = np.zeros((cols, 1))

    # Another way of checking if b is in C(A)
    # if rank < rows:
    #     for i in range(rows - 1, rank - 1, -1):
    #         if R_aug[i, -1] != 0:
    #             return None

    i = 0
    for j in range(cols):
        if j in pivots:
            xp[j] = R_aug[i, -1]
            i += 1

    return xp


if __name__ == '__main__':
    main()

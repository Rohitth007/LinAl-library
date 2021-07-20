# 3 methods
# LU Decomposition
# Big Formula (n! terms)
# Cofactor Formula
import numpy as np
from itertools import permutations
from get_matrix import get_matrix
from LU_Decomp import lu_decomp


def main():
    A = get_matrix()

    D1 = det(A)  # default method=lu_decomp
    print(f"\nDeterminant using LU Decomposition = {D1}")

    D2 = det(A, method='cofactor_formula')
    print(f"\nDeterminant using Cofactor Formula = {D2}")

    D3 = det(A, method='permutation_formula')
    print(f"\nDeterminant using Permutation Formula = {D3}")


def det(A, method='lu_decomp'):
    rows, cols = A.shape
    if rows != cols:
        print("Given Matrix is not Square")
        exit(1)

    if method == 'lu_decomp':
        _, _, U = lu_decomp(A, verbose=False)
        det = 1
        for i in range(rows):
            det *= U[i, i]
        return det

    elif method == 'cofactor_formula':  # Recursive Method
        return _det(A)

    elif method == 'permutation_formula':  # Iterative Method
        det = 0
        perms = permutations(range(cols))
        for n, perm in enumerate(perms):
            d = 1
            for i in range(rows):
                d *= A[i, perm[i]]
            det += parity(perm, cols)*d
        return det

    else:
        print('\nMethod Unknown')
        exit(1)


# Finds the parity of the permutation (no of exchanges)
def parity(perm, n):
    perm = list(perm)
    normal = list(range(n))
    inversions = 0
    while perm:
        p = perm.pop(0)
        inversions += normal.index(p)
        normal.remove(p)
    parity = -1 if inversions % 2 == 1 else 1
    return parity


# Determinant using determinant
def _det(A):
    _, cols = A.shape
    # Base case
    if cols == 1:
        return A[0, 0]
    # Recursion
    det = 0
    for j in range(cols):
        a = np.concatenate((A[1:, :j], A[1:, j+1:]), axis=1)
        det += A[0, j]*(-1)**(2+j)*_det(a)

    return det


if __name__ == '__main__':
    main()

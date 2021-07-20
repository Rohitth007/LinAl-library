from get_matrix import get_matrix
from gauss_jordan import gj
from determinant import det
import numpy as np


def main():
    A = get_matrix("Input Matrix to be inverted")

    A_inv = inv(A)
    print("\nInverse using Gauss_Jordan Elimination:\n", A_inv)

    A_inv = inv(A, method='formula')
    print("\nInverse using Formula:\n", A_inv)


def inv(A, method='gauss_jordan'):
    rows, cols = A.shape
    inv, pivots = gj(A, returnPivot=True, verbose=False)

    if rows != cols or len(pivots) != rows:
        print('Matrix not Invertible')
        exit

    if method == 'formula':
        inv = inv_formula(A)

    return inv


def inv_formula(A):
    rows, cols = A.shape
    C = np.zeros_like(A)

    det_A = det(A)
    for i in range(rows):
        for j in range(cols):
            C[i, j] = cofactor(A, i, j)
    C_t = np.transpose(C)

    inv = C_t / det_A
    return inv


def cofactor(A, i, j):
    m1 = np.concatenate((A[:i, :j], A[:i, j+1:]), axis=1)
    m2 = np.concatenate((A[i+1:, :j], A[i+1:, j+1:]), axis=1)
    minor = np.concatenate((m1, m2), axis=0)

    cofactor = (-1)**(i+j+2)*det(minor)
    return cofactor


if __name__ == '__main__':
    main()

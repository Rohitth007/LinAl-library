# Convert a matrix A to an orthogonal matrix Q
# and return its QR decomposition
# Not for non-full rank matrices but
# May give slightly weird answers when A is not full rank.
import numpy as np
from get_matrix import get_matrix


def main():
    A = get_matrix('Input Full Column Rank Matrix')

    Q, R = QR_decomp(A)

    print('Orthogonal Matrix, Q:\n', Q)
    print('Upper Triangular Matrix, R:\n', R)


def QR_decomp(A):
    '''a,b,c are columns of A.
    Then a' = a and q1 = a'/|a'|
        b' = b - q1t*b*q1 and q2 = b'/|b'|
        c' = c - q1t*c*q1 - q2t*c*q2 and q3 = c'/|c'|
    q1t*b*q1 is same as (at*b/at*a)*a same as x*a which equals p
    Rearranging a = q1t*a*q1
                b = q1t*b*q1 + q2t*b*q2
                c = q1t*c*q1 + q2t*c*q2 + q3t*c*q3
    This in Matrix from is A = QR'''
    rows, cols = A.shape
    Q = np.empty_like(A)
    R = np.zeros_like(A)

    for j in range(cols):
        col = A[:, j]
        for c in range(j):
            q = Q[:, c]
            R[c, j] = scale_factor = np.dot(q, col)
            col -= scale_factor*q  # projection of col on q (qt*col*q)
        try:
            R[j, j] = mag = np.sqrt(np.dot(col, col))
        except:
            pass  # When rank is not full
        Q[:, j] = col/mag

    return Q, R


if __name__ == '__main__':
    main()

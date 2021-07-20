# Performs Gauss Jordan Elimination on any matrix
import numpy as np
from get_matrix import get_matrix
from rref import rref


def main():
    '''Finds inverse using gauss jordan'''
    # Get Matrix from user
    A = get_matrix()
    rows, cols = A.shape
    if rows != cols:
        print('Matrix is not square!')
        exit(1)

    inv = gj(A)

    print('\nInverse of the given Matrix is:\n', inv)


def gj(A, verbose=True, returnPivot=False):
    rows, cols = A.shape
    # Identity Matrix of same size
    I = np.identity(rows)
    E = np.zeros([rows, cols])

    # Augmented matrix
    aug = np.concatenate((A, I), axis=1)
    R = rref(aug, verbose=verbose, returnPivot=returnPivot)
    if returnPivot:
        R, pivots = R
    if verbose:
        print('\n', R)

    # Extracts Elimination Matrix
    for r in range(rows):
        for c in range(cols, rows + cols):
            E[r, c-cols] = R[r, c]

    if returnPivot:
        return E, pivots
    return E


if __name__ == '__main__':
    main()

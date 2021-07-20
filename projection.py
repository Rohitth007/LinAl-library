# Projects b onto the columnspace of A
# Does not handle the case of A not having full rank
import numpy as np
from get_matrix import get_matrix
from inverse import inv


def main():
    A = get_matrix('Input Full Column Rank Matrix(A) to project onto')
    ortho = True if input('Is Matrix Orthogonal, y or n?') == 'y' else Flase
    b = get_matrix('Input vector to be projected as matrix')

    p, P = project(b, A, ortho=ortho)

    print('\nProjection of b onto the columnspace of A:\n', p)
    print('\nProjection Matrix P transforming b to p:\n', P)


def project(b, A, ortho=False):
    rows, cols = A.shape
    At = np.transpose(A)

    if ortho:
        if rows == cols:
            P = np.identity(rows)
            p = b
            return p, P
        else:
            P = A@At
    else:
        P = A@inv(At@A)@At

    p = P@b
    return p, P


if __name__ == '__main__':
    main()

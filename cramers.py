import numpy as np
from get_matrix import get_matrix
from determinant import det


def main():
    A = get_matrix('Input Invertible matrix to perform Crammers Rule')
    b = get_matrix('Input vector b as matrix')
    _, cols = A.shape

    try:
        x = crammer(A, b)
    except:
        print("\nError: Can't perform Crammer's Rule on this matrix try Ax=b.py")
        exit(1)

    print('Solution to Ax=b using Crammer\'s Rule:\n', x)


def crammer(A, b):
    rows, cols = A.shape
    x = np.empty((cols, 1))

    for j in range(cols):
        # Replace the jth row with b -> Bj
        B = A.copy()
        for i in range(rows):
            B[i, j] = b[i]

        x[j] = det(B) / det(A)

    return x


if __name__ == '__main__':
    main()

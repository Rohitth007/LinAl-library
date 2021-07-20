# Finds Eigenvalues and Eigenvectors when eigenvalues are not irrational
#
# Improvement:
# Irrational sympy calculations(approximations) give wrong answers
from determinant import det
from get_matrix import get_matrix
from funda_subspaces import subspaces
import numpy as np
import sympy as sym


def main():
    rows = 1
    cols = 0
    while rows != cols:
        A = get_matrix('Enter Square Matrix to find Eigenvalue of')
        rows, cols = A.shape
        if rows == cols:
            break
        print('Matrix is not square!')

    try:
        L, S = eig(A)
    except:
        print('Irrational Approximations leads to wrong results in elimination.')
        exit(1)

    print('\nEigenvalue Matrix/Eigenbasis Matrix L:\n', L)
    print('Eigenvector Matrix S:\n', S)
    print('\nA can be diagonalsied as:\n A = S*L*S\u207B\n\tor\n S\u207B*A*S = L')


def eig(A, verbose=True):
    rows, cols = A.shape
    I = np.identity(rows)

    # Find eigenvalues
    lambdaa = sym.symbols('lambdaa')
    char_eqn = sym.simplify(det(A - lambdaa*I))
    eigenvalues = sym.solve(char_eqn)

    # Repeated eigenvalues case
    if len(eigenvalues) != rows:
        print('Eigen vectors may not be independent.')

    # To decide which data type numpy matrix should use
    if not all([type(e) == sym.core.numbers.Float for e in eigenvalues]):
        dt = np.complex64
        eigenvalues = [complex(e) for e in eigenvalues]
    else:
        dt = np.float64

    if verbose:
        print('\nCharacteristic Equation:\n', char_eqn)
        print('\nEigenvalues of given matrix is:\n', eigenvalues)

    S = np.empty((cols, 0))         # Eigenvector Matrix
    L = np.zeros_like(A, dtype=dt)  # Eigenvalue Matrix

    for i, l in enumerate(eigenvalues):
        if verbose:
            if dt == np.float64:
                print(f'\nEigenvalue = {l:.2f}')
            else:
                print(f'\nEigenvalue = {l}')
        L[i, i] = l

        # Sympy-Numpy convertion
        # https://stackoverflow.com/questions/10129213/combining-numpy-with-sympy
        M = sym.Matrix(A-lambdaa*I).subs(lambdaa, l)
        M = np.array(M)

        # Nullspace of M is eigenvector
        _, _, _, _, eigvec, _, _ = subspaces(M, verbose=False, dtype=dt)
        print(f'Eigenvector =\n', eigvec) if verbose else print('', end='')
        S = np.append(S, eigvec, axis=1)

    return L, S


if __name__ == '__main__':
    main()

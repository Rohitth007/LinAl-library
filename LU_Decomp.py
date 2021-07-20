# Finds The LU Decomposition of a matrix A
# PA = LU
# Improvements: Doesn't work with sympy symbols. Line #69
import numpy as np
from get_matrix import get_matrix


def main():
    # Get Matrix from user
    A, rows, cols = get_matrix()

    P, L, U = lu_decomp(A)

    # Print Result
    print("\nP:\n", P)
    print("\nA:\n", A)
    print("\nL:\n", L)
    print("\nU:\n", U)
    print("\nL*U:\n", np.matmul(L, U))


def lu_decomp(A, verbose=True):
    rows, cols = A.shape
    U = np.copy(A)
    # P & L are always square matrices even if A is not.
    L = np.zeros((rows, rows))
    P = np.zeros((rows, rows))
    changed = {i: i for i in range(rows)}

    for i in range(rows):
        j = i
        # Find Non-zero Pivot
        not_found = True
        while not_found and j < cols:
            if U[i, j]:
                not_found = False
            else:
                for ex in range(i+1, rows):
                    # Exchange Rows
                    if U[ex, j]:
                        not_found = False
                        U[i], U[ex] = U[ex], np.copy(U[i])
                        if verbose:
                            print("\nExchange:\n", U)

                        # Take note of exchanged rows
                        changed[i] = ex
                        changed[ex] = i
                        break
                else:
                    j += 1

        # i,j is your pivot

        # Permutation Matrix
        P[i, changed[i]] = 1

        # Lower Triangular Matrix
        L[i, changed[i]] = 1

        # If no pivots found in a row, not even row exchanges
        if j == cols:
            continue

        # Elimination
        for r in range(i+1, rows):
            p = U[r, j]/U[i, j]
            try:  # doesn't work with sympy symbols
                L[r, changed[i]] = p
            except:
                pass

            for c in range(j, cols):
                U[r, c] = U[r, c] - p*U[i, c]

        # Result of each elimination
        if verbose:
            print(f"\nStep {i+1}:\n", U)

    return P, L, U


if __name__ == '__main__':
    main()

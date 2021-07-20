# Converts any matrix to its Reduced Row Echelon Form (RREF)
# Irrational approximation leads to wrong answers. Eg: Fibonacci Matix
import numpy as np
from get_matrix import get_matrix


def main():
    # Get Matrix from user
    A = get_matrix()
    print(A)
    R = rref(A)

    # Print Result
    print("\nReduced Row Echelon Form:\n", R)


def rref(A, returnPivot=False, verbose=True):
    rows, cols = A.shape
    R = np.copy(A)
    Pivot = []

    for i in range(rows):
        if i < cols:
            j = i
            # Find Non-zero Pivot
            not_found = True
            while not_found and j < cols:
                if R[i, j]:
                    not_found = False
                else:
                    for ex in range(i+1, rows):
                        # Exchange Rows
                        if R[ex, j]:
                            not_found = False
                            R[i], R[ex] = R[ex], np.copy(R[i])
                            if verbose:
                                print("\nExchange:\n", R)
                            break
                    else:
                        j += 1

            # i,j is your pivot
            # If no pivots found in a row, not even row exchanges
            if j == cols:
                continue

            Pivot.append(j)

            # Elimination
            for r in range(rows):
                k, div = 0, 1
                if r == i:
                    div = R[r, j]
                else:
                    k = R[r, j]/R[i, j]

                for c in range(j, cols):
                    R[r, c] = R[r, c]/div - k*R[i, c]

            # Result of each elimination
            if verbose:
                print(f"\nStep {i+1}:\n", R)

    if returnPivot:
        return R, Pivot
    return R


if __name__ == '__main__':
    main()

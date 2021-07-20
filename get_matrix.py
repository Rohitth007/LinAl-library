import numpy as np

# Get Matrix from user


def get_matrix(disp=None):
    if disp:
        print("\n", disp)
    rows = int(input("Enter no. of rows: "))
    cols = int(input("Enter no. of columns: "))

    A = np.empty([rows, cols])

    print("Input your Matrix row-wise as csv:")
    for r in range(rows):
        x = input(f"Row{r+1}>> ")
        x = list(map(float, x.split(",")))
        A[r] = x

    # Example:
    # A = np.array([[ 1, -1, -1, 1],
    #               [ 2,  0,  2, 0],
    #               [ 0, -1, -2, 0],
    #               [ 3, -3, -2, 4]])
    # rows = cols = 4

    print("\nGiven Matrix:\n", A)
    print("\n   rows =", rows)
    print("   columns =", cols)

    return A

# Polynomial curve fitting using Least Squares method
# Does not deal with inputs like: curve passes through origin
# Maybe use sympy?
import numpy as np
import matplotlib.pyplot as plt
from Ax_b import solution


def main():
    deg = int(input('Input degree of curve to fit: '))
    # Get data points from user
    points = []
    while len(points) < deg:
        not_done = True
        print('\nEnter points:')
        print('  No.of points should be more than degree')
        print("  Press 'q' when done.")
        while not_done:
            print('Input data points as (x,y):')
            point = input('> ')
            if point == 'q':
                not_done = False
            else:
                point = list(map(int, point.split(',')))
                points.append(point)

    # Find coefficients using least squares
    coeff = curve_fit(points, deg)
    print('\nCoefficients:')
    for i, c in enumerate(coeff[:, 0]):
        print(f'  a{i} = {c}')

    # ### PLOTTING ###
    # Find x max & min for x_lim of plot
    x_max = x_min = points[0][0]
    for x, _ in points:
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
    min_lim = 1.3*x_min if x_min < 0 else 0.7*x_min
    max_lim = 0.7*x_max if x_max < 0 else 1.3*x_max
    # Form points on nth deg polynomial
    y_val = []
    x_val = np.linspace(max_lim, min_lim, 100)
    for x in x_val:
        y = 0
        for i in range(deg+1):
            y += coeff[i, 0]*x**i
        y_val.append(y)
    # Plot points
    plt.plot(x_val, y_val, linewidth=3.5)
    for point in points:
        plt.scatter(point[0], point[1], color='black')
    plt.show()


def curve_fit(points, deg):
    if len(points) < deg:
        print('Error: No.of points should be more than degree')
        exit(1)

    A = np.empty((len(points), deg+1))
    b = np.empty((len(points), 1))
    # Form matrix A and b
    for i, point in enumerate(points):
        x, y = point
        b[i] = y
        for j in range(deg+1):
            A[i, j] = x**j

    At = np.transpose(A)
    aug = np.concatenate((A, b), axis=1)
    aug = At@aug
    coeff, _ = solution(aug, verbose=False)

    # ALITER: More computation
    # coeff = inv(At@A)@At@b
    # print(coeff)
    return coeff


if __name__ == '__main__':
    main()

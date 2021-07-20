# Finds the kth iteration of the first order difference equation in matri form.
#
# Improvement:
# Does not consider Repeated eigenvales
import numpy as np
from get_matrix import get_matrix
from eigen import eig
from inverse import inv


def main():
    A = get_matrix(
        'Convert your difference equation to a\nfirst order equation in matrix form and input matrix')
    u = get_matrix('Input Initial Condition as a column matrix')
    k = int(input('Which iteration do you want to find? Value of k:  '))

    u_k = diff_eqn(A, u, k)
    print('u after k iterations, u_k:\n', u_k)


def diff_eqn(A, u, k):
    try:
        L, S = eig(A, verbose=False)
    except:
        print('Irrational Approximations leads to wrong results in elimination.')
        exit(1)

    stable = []
    steady = []
    unstable = []
    for i in range(L.shape[0]):
        L_k[i, i] = L[i, i]**k

        if L[i, i] > 1:
            unstable.append(i)
        elif L[i, i] == 1:
            steady.append(i)
        else:
            stable.append(i)

    if len(stable) = L.shape[0]:
        print('System is stable as all eigenvalues are < 1')
    elif len(unstable) != 0:
        print(f'System is unstable for eigenvalues:\n',
              [L[i, i] for i in unstable])
    else:
        print('System reaches steady state')

        # Finds steady state
        c = inv(S)@u
        steady_state = np.empty((rows, 0))
        for i in steady:
            steady_state += c[i]*S[:, i]
        print('Steady State:\n', steady_state)

        u_k = S@L_k@c
        return u_k, steady_state

    u_k = S@L_k@inv(S)@u
    return u_k


if __name__ == '__main__':
    main()

import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import scipy.io


def approximation_poly(n, x, N):
    mu = np.array([np.mean(n), np.std(n)])
    n2 = (n - mu[0]) / (mu[1])
    poly_factor = np.polyfit(n2, x, N)
    x_approx = np.polyval(poly_factor, n2)
    return poly_factor, x_approx


def approximation_trig(N, n, x):
    n = n / np.max(n)
    S = np.zeros((N, N))
    for k in range(1, N+1):
        for l in range(1, N+1):
            for i in range(len(x)):
                S[k-1, l-1] += math.cos(l * n[i]) * math.cos(k * n[i])

    t = np.zeros((N, 1))
    for k in range(1, N+1):
        for i in range(len(x)):
            t[k-1, 0] += x[i] * math.cos(k * n[i])

    c = np.linalg.solve(S, t)
    c1 = np.zeros((N, len(n)))
    for i in range(1, N+1):
        c1[i-1, :] = np.cos(i*n)

    x_approx = np.dot(c1.T, c).flatten()
    return x_approx


def get_diff(approx, exact, N):
    err = 0
    for i in range(N):
        err += pow((approx[i] - exact[i]), 2)
    return err


def main_drone():
    N = 50

    TRAJ1 = scipy.io.loadmat('traj1.mat')
    n_traj1 = TRAJ1['n']
    x_traj1 = TRAJ1['x']
    y_traj1 = TRAJ1['y']
    z_traj1 = TRAJ1['z']
    n_traj1 = np.reshape(n_traj1, (np.prod(n_traj1.shape),))
    x_traj1 = np.reshape(x_traj1, (np.prod(x_traj1.shape),))
    y_traj1 = np.reshape(y_traj1, (np.prod(y_traj1.shape),))
    z_traj1 = np.reshape(z_traj1, (np.prod(z_traj1.shape),))

    poly_factor_x, xa = approximation_poly(n_traj1, x_traj1, N)
    poly_factor_y, ya = approximation_poly(n_traj1, y_traj1, N)
    poly_factor_z, za = approximation_poly(n_traj1, z_traj1, N)
    figure1 = plt.figure()
    ax = figure1.add_subplot(111, projection='3d')
    ax.scatter(x_traj1, y_traj1, z_traj1, c='b', marker='.')
    ax.scatter(xa, ya, za, c='r', marker='o', linestyle='-', linewidth=0.01)
    ax.set_xlabel("X")
    ax.set_xlabel("Y")
    ax.set_xlabel("Z")
    plt.title('polynomial approximation for traj 1\nred - approximation')
    plt.show()

    TRAJ2 = scipy.io.loadmat('traj2.mat')
    n_traj2 = TRAJ2['n']
    x_traj2 = TRAJ2['x']
    y_traj2 = TRAJ2['y']
    z_traj2 = TRAJ2['z']

    n_traj2 = np.reshape(n_traj2, (np.prod(n_traj2.shape),))
    x_traj2 = np.reshape(x_traj2, (np.prod(x_traj2.shape),))
    y_traj2 = np.reshape(y_traj2, (np.prod(y_traj2.shape),))
    z_traj2 = np.reshape(z_traj2, (np.prod(z_traj2.shape),))

    poly_factor_x, xa = approximation_poly(n_traj2, x_traj2, N)
    poly_factor_y, ya = approximation_poly(n_traj2, y_traj2, N)
    poly_factor_z, za = approximation_poly(n_traj2, z_traj2, N)
    figure2 = plt.figure()
    ax = figure2.add_subplot(111, projection='3d')
    ax.scatter(x_traj2, y_traj2, z_traj2, c='b', marker='.')
    ax.scatter(xa, ya, za, c='r', marker='o', linestyle='-', linewidth=0.01)
    ax.set_xlabel("X")
    ax.set_xlabel("Y")
    ax.set_xlabel("Z")
    plt.title('polynomial approximation for traj 2\nred - approximation')
    plt.show()

    err = []
    M = len(n_traj2)
    minv = 1 / M
    # I calculated the inverse of M before the loop to make it quicker,
    # since division has much higher time complexity than multiplication
    for K in range(70):
        poly_factor_x, xa = approximation_poly(n_traj2, x_traj2, K)
        poly_factor_y, ya = approximation_poly(n_traj2, y_traj2, K)
        poly_factor_z, za = approximation_poly(n_traj2, z_traj2, K)
        errx = (math.sqrt(get_diff(xa, x_traj2, K))) * minv
        erry = (math.sqrt(get_diff(ya, y_traj2, K))) * minv
        errz = (math.sqrt(get_diff(za, z_traj2, K))) * minv
        err.append(errx + erry + errz)
    plt.semilogy(err)
    plt.title('graph of error for polynomial approx for traj2')
    plt.xlabel('N')
    plt.ylabel('err')
    plt.show()
    # the error is caused by the Runge's effect

    xa = approximation_trig(N, n_traj1, x_traj1)
    ya = approximation_trig(N, n_traj1, y_traj1)
    za = approximation_trig(N, n_traj1, z_traj1)
    figure3 = plt.figure()
    ax = figure3.add_subplot(111, projection='3d')
    ax.scatter(x_traj1, y_traj1, z_traj1, c='b', marker='.')
    ax.scatter(xa, ya, za, c='r', marker='o', linestyle='-', linewidth=0.01)
    ax.set_xlabel("X")
    ax.set_xlabel("Y")
    ax.set_xlabel("Z")
    plt.title('trigonometric approximation for traj 1\nred - approximation')
    plt.show()

    xa = approximation_trig(N, n_traj2, x_traj2)
    ya = approximation_trig(N, n_traj2, y_traj2)
    za = approximation_trig(N, n_traj2, z_traj2)
    figure4 = plt.figure()
    ax = figure4.add_subplot(111, projection='3d')
    ax.scatter(x_traj2, y_traj2, z_traj2, c='b', marker='.')
    ax.scatter(xa, ya, za, c='r', marker='o', linestyle='-', linewidth=0.01)
    ax.set_xlabel("X")
    ax.set_xlabel("Y")
    ax.set_xlabel("Z")
    plt.title('trigonometric approximation for traj 2\nred - approximation')
    plt.show()

    err = []
    M = len(n_traj2)
    minv = 1 / M
    for K in range(70):
        xa = approximation_trig(K, n_traj2, x_traj2)
        ya = approximation_trig(K, n_traj2, y_traj2)
        za = approximation_trig(K, n_traj2, z_traj2)
        errx = (math.sqrt(get_diff(xa, x_traj2, K))) * minv
        erry = (math.sqrt(get_diff(ya, y_traj2, K))) * minv
        errz = (math.sqrt(get_diff(za, z_traj2, K))) * minv
        err.append(errx + erry + errz)
    plt.semilogy(err)
    plt.title('graph of error for trigonometric approx for traj2')
    plt.xlabel('N')
    plt.ylabel('err')
    plt.show()

    err = []
    M = len(n_traj1)
    minv = 1 / M
    for K in range(70):
        xa = approximation_trig(K, n_traj1, x_traj1)
        ya = approximation_trig(K, n_traj1, y_traj1)
        za = approximation_trig(K, n_traj1, z_traj1)
        errx = (math.sqrt(get_diff(xa, x_traj1, K))) * minv
        erry = (math.sqrt(get_diff(ya, y_traj1, K))) * minv
        errz = (math.sqrt(get_diff(za, z_traj1, K))) * minv
        err.append(errx + erry + errz)
    plt.semilogy(err)
    plt.title('graph of error for trigonometric approx for traj1')
    plt.xlabel('N')
    plt.ylabel('err')
    plt.show()

    N = np.argmin(err)
    M = np.min(err)
    print(f"Best N: {N}")
    print(f"Minimum error: {M}")


main_drone()

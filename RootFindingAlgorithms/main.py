# task 2 comment
# the convergence rate for bisect is almost linear but takes significantly more iterations
# secant is much better choice because it takes fewer iterations and gives more accurate results
# however secant is slightly more complicated, despite that I believe it is better to utilize it over bisection
import math
import matplotlib.pyplot as plt
import scipy.optimize


def apply_function(func, x):
    return func(x)


def compute_impedance(omega):
    if omega == 0:
        return -75
    z = 1 / (math.sqrt(1 / pow(725, 2) + pow((omega * 0.00008 - 1 / (omega * 2)), 2)))
    value = z - 75
    return value


def compute_velocity(t):
    value = 2000 * math.log(150000 / (150000 - 2700 * t)) - 9.81 * t - 750
    return value


def bisect(a, b, eps, func):
    x_vect = []
    x_diff = []
    for i in range(1, 1000):
        x = (a + b) / 2
        if (abs(apply_function(func, x)) <= eps) & (abs(b - a) <= eps):
            break
        elif (apply_function(func, a) * apply_function(func, x)) < 0:
            b = x
        else:
            a = x
        x_vect.append(x)
        if i > 1:
            x_diff.append(abs(x_vect.__getitem__(i - 2) - x))
        else:
            x_diff.append(x)
        fx = apply_function(func, x)

    nr_of_iter = len(x_vect)
    return x_vect, x_diff, fx, nr_of_iter


def secant(a, b, eps, func):
    x_vect = []
    x_diff = []
    f1 = apply_function(func, a)
    f2 = apply_function(func, b)
    for i in range(1, 1000):
        if (abs(f1 - f2) <= eps) & (abs(a - b) <= eps):
            break
        x0 = a - f1 * ((a - b) / (f1 - f2))
        f0 = apply_function(func, x0)
        b = a
        f2 = f1
        a = x0
        f1 = f0
        x_vect.append(x0)
        if i > 1:
            x_diff.append(abs(x_vect.__getitem__(i - 2) - x0))
        else:
            x_diff.append(x0)

    fx = apply_function(func, x0)
    nr_of_iter = len(x_vect)
    return x_vect, x_diff, fx, nr_of_iter


def main():
    a = 0
    b = 50

    # figure 1
    x_vect, x_diff, fx, nr_of_iter = bisect(a, b, 1e-12, compute_impedance)
    plt.plot(x_vect, 'C1')
    x_vect, x_diff, fx, nr_of_iter = secant(a, b, 1e-12, compute_impedance)
    plt.plot(x_vect, 'C2')
    plt.xlabel('iterations')
    plt.ylabel('values of x')
    plt.title('iterations vs values of x for bisect and secant for impedance')
    plt.show()

    # figure 2
    x_vect, x_diff, fx, nr_of_iter = bisect(a, b, 1e-12, compute_impedance)
    plt.semilogy(x_diff, 'C1')
    x_vect, x_diff, fx, nr_of_iter = secant(a, b, 1e-12, compute_impedance)
    plt.semilogy(x_diff, 'C2')
    plt.ylabel('differences in x')
    plt.xlabel('iterations')
    plt.title('iterations vs differences in x for bisect and secant for impedance\nusing semilogy')
    plt.show()

    # figure 3
    x_vect, x_diff, fx, nr_of_iter = bisect(a, b, 1e-12, compute_velocity)
    plt.plot(x_vect, 'C1')
    x_vect, x_diff, fx, nr_of_iter = secant(a, b, 1e-12, compute_velocity)
    plt.plot(x_vect, 'C2')
    plt.xlabel('iterations')
    plt.ylabel('values of x')
    plt.title('iterations vs values of x for bisect and secant for velocity')
    plt.show()

    # figure 4
    x_vect, x_diff, fx, nr_of_iter = bisect(a, b, 1e-12, compute_velocity)
    plt.semilogy(x_diff, 'C1')
    x_vect, x_diff, fx, nr_of_iter = secant(a, b, 1e-12, compute_velocity)
    plt.semilogy(x_diff, 'C2')
    plt.ylabel('differences in x')
    plt.xlabel('iterations')
    plt.title('iterations vs differences in x for bisect and secant for velocity\nusing semilogy')
    plt.show()

    root1, info_dict1, ier1, msg1 = scipy.optimize.fsolve(math.tan, 6, full_output=True)
    iterations1 = info_dict1['nfev']
    root2, info_dict2, ier2, msg2 = scipy.optimize.fsolve(math.tan, 4.5, full_output=True)
    iterations2 = info_dict2['nfev']


main()

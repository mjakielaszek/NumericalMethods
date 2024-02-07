import numpy as np
import matplotlib.pyplot as plt
import scipy.io


def emacalc(n, day, prices):
    alpha = 2 / (n+1)
    numerator = 0
    denominator = 0
    base = 1 - alpha
    for i in range(n + 1):
        value_deno = pow(base, i)
        denominator += value_deno

    currday = day - 1
    for i in range(n + 1, 0, -1):
        buff = pow(base, (n + 1) - i)
        value = prices.__getitem__(currday)[0]
        value = value * buff
        numerator += value
        currday -= 1

    ema = numerator / denominator
    return ema

def emacalc1(n, day, prices):
    alpha = 2 / (n+1)
    numerator = 0
    denominator = 0
    base = 1 - alpha
    for i in range(n+1):
        denominator += pow(base, i)

    currday = day - 1
    for i in range(n + 1, 0, -1):
        buff = pow(base, (n + 1) - i)
        value = prices.__getitem__(currday)
        value = value * buff
        numerator += value
        currday -= 1

    ema = numerator / denominator
    return ema

def main():
    mat = scipy.io.loadmat('wig20.mat')
    wig20 = mat['WIG20']
    macd = [0] * wig20.size
    signal = [0] * wig20.size
    for i in range(27, wig20.size):
        macd[i] = emacalc(12, i, wig20) - emacalc(26, i, wig20)
        signal[i] = emacalc1(9, i, macd)

    plt.ylim(-150, 150)
    plt.xlim(0, 1000)
    plt.plot(macd)
    plt.plot(signal)
    plt.legend(['macd', 'signal'])
    plt.show()


main()
# the indicator can be useful in most cases however it seems to react to changes on the market too late to make reasonable decisions based on it
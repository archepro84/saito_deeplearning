from typing import *
import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    y = x > 0
    # return y
    return y.astype(np.int)


def step_function2(x):
    return np.array(x > 0, dtype=np.int)


def note_3_2_3():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function2(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def note_3_2_4():
    x = np.array([-1.0, 1.0, 2.0])
    sigmoid()


if __name__ == '__main__':
    # note_3_2_3()



    pass

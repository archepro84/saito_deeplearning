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
    result = sigmoid(x)
    print(result)

    t = np.array([1.0, 2.0, 3.0])
    print(1.0 + t)
    print(1.0 / t)

    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


def relu(x):
    return np.maximum(0, x)


def note_3_2_6():
    x = np.array([-1, -0.1, 1, 2])
    result = relu(x)
    print(result)


def note_3_3_1():
    A = np.array([1, 2, 3, 4])
    print(A)
    print(np.ndim(A))
    print(A.shape)
    print(A.shape[0])
    print()

    B = np.array([[1, 2], [3, 4], [5, 6]])
    print(B)
    print(np.ndim(B))
    print(B.shape)


def note_3_3_2():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print(A.shape)
    print(B.shape)
    result = np.dot(A, B)
    print(result)

    C = np.array([[1, 2, 3], [4, 5, 6]])
    D = np.array([[1, 2], [3, 4], [5, 6]])
    print(C.shape)
    print(D.shape)
    result = np.dot(C, D)
    print(result)

    E = np.array([[1, 2], [3, 4]])
    print(C.shape)
    print(E.shape)
    try:
        print(np.dot(C, E))
    except ValueError:
        print(ValueError)
    print()

    A = np.array([[1, 2], [3, 4], [5, 6]])
    B = np.array([7, 8])
    print(A.shape)
    print(B.shape)
    result = np.dot(A, B)
    print(result)

    pass


def note_3_3_3():
    X = np.array([1, 2])
    W = np.array([[1, 3, 5], [2, 4, 6]])
    print(X.shape)
    print(W.shape)
    Y = np.dot(X, W)
    print(Y)
    pass


# **
def note_3_4_2():
    X = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])

    print(W1.shape)
    print(X.shape)
    print(B1.shape)
    print(np.dot(X, W1))

    A1 = np.dot(X, W1) + B1
    print(A1)

    Z1 = sigmoid(A1)
    print(Z1)


def note_3_4_2_2():
    Z1 = np.array([0.57444252, 0.66818777, 0.75026011])
    W2 = np.array([[0.1, 0.4], [0.4, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])
    print(Z1.shape)
    print(W2.shape)
    print(B2.shape)

    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)
    print(A2)
    print(Z2)


def identity_function(x):
    return x


def note_3_4_2_3():
    Z2 = np.array([0.6569648, 0.7710107])
    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])

    A3 = np.dot(Z2, W3) + B3
    Y = identity_function(A3)  # 항동함수 / 은닉층 > 출력층의 활성화 함수의 흐름을 통일하기위해 작성
    print(A3)
    print(Y)


def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    Y = identity_function(a3)
    return Y


def note_3_4_3():
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)


def note_3_5_1():
    a = np.array([0.3, 2.9, 4.0])
    exp_a = np.exp(a)
    print(exp_a)

    sum_exp_a = np.sum(exp_a)
    print(sum_exp_a)
    y = exp_a / sum_exp_a
    print(y)

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

if __name__ == '__main__':
    # note_3_2_3()
    # note_3_2_4()
    # note_3_3_1()
    # note_3_3_2()
    # note_3_3_3()
    # note_3_4_2()
    # note_3_4_2_2()
    # note_3_4_2_3()
    # note_3_4_3()
    note_3_5_1()

    pass

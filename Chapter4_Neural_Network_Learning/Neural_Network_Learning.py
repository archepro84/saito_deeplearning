import sys, os

sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist


# y : 추정 레이블, t : 정답 레이블
def sum_squared_error(y, t):
    return (1 / 2) * np.sum((y - t) ** 2)


def note4_2_1():
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    print(sum_squared_error(np.array(y), np.array(t)))

    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    print(sum_squared_error(np.array(y), np.array(t)))


# np.log(0)을 입력하면 -Infinity가 발생하므로 0이 되지않도록 delta 사용
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def note4_2_2():
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    print(cross_entropy_error(np.array(y), np.array(t)))

    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    print(cross_entropy_error(np.array(y), np.array(t)))


def note4_2_3():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)

    print(x_train.shape)
    print(t_train.shape)

    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    print(batch_mask)
    print(x_batch)
    print(t_batch)


def cross_entropy_error_batch(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    # one-hot encoding일 경우 ex)[0,0,1,0,0,0,0,0,0,0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size  # 1e-7 : np.log(0) Error 방지

    #
    # one-hot encoding이 아닐경우 (숫자 레이블로 주어졌을 경우) ex) 1,2,7,5
    return -np.sum(np.log(y[np.arrange(batch_size), t] + 1e-7)) / batch_size


def note4_2_4():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - (f(x - h))) / (2 * h)


if __name__ == '__main__':
    # note4_2_1()
    # note4_2_2()
    # note4_2_3()
    note4_2_4()
    pass

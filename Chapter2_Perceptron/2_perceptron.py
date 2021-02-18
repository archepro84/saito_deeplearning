import numpy as np

"""
퍼셉트론을 정의한다.
b = 편향 (bias)
w = 가중치(weight)
"""


def AND(x1, x2, debuglog=True):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7

    if np.sum(x * w) + b > 0:
        result = True
    else:
        result = False
    if debuglog:
        print(f"AND({x1},{x2}) : {result}")
    return result


def OR(x1, x2, debuglog=True):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.3

    if np.sum(x * w) + b > 0:
        result = True
    else:
        result = False
    if debuglog:
        print(f"OR({x1},{x2}) : {result}")
    return result


def NAND(x1, x2, debuglog=True):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7

    if np.sum(x * w) + b > 0:
        result = True
    else:
        result = False
    if debuglog:
        print(f"NAND({x1},{x2}) : {result}")
    return result


def XOR(x1, x2, debuglog=True):
    xd1 = NAND(x1, x2, False)
    xd2 = OR(x1, x2, False)
    result = AND(xd1, xd2, False)

    if debuglog:
        print(f"XOR({x1},{x2}) : {result}")


if __name__ == "__main__":
    AND(0, 0)
    AND(0, 1)
    AND(1, 0)
    AND(1, 1)
    print()
    NAND(0, 0)
    NAND(0, 1)
    NAND(1, 0)
    NAND(1, 1)
    print()
    OR(0, 0)
    OR(0, 1)
    OR(1, 0)
    OR(1, 1)
    print()
    XOR(0, 0)
    XOR(0, 1)
    XOR(1, 0)
    XOR(1, 1)

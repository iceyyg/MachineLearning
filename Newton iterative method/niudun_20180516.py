import numpy as np
import matplotlib.pyplot as plt
'''x = [i for i in range(100)]

x_ = np.array(x)

y_ = x_**2

print(x_, y_)

plt.plot(x_, y_)
plt.plot(x_, x_)
plt.show()'''


def newton(x0, yibusilong):
    x1 = 1 / 2 * (x0 + 2 / x0)
    count = 1
    x_ = [x1, ]
    while abs(x1 - x0) >= yibusilong:
        x0 = x1
        x1 = 1 / 2 * (x0 + 2 / x0)
        count += 1
        x_.append(x1)
    return x1, count, x_


def erfenfa(a, b, yibusilong):
    mid = (a+b)/2
    count = 1
    x_ = [mid, ]
    while b-a >= yibusilong:
        if (a**2-2)*(mid**2-2) < 0:
            b = mid
        else:
            a = mid
        mid = (a+b)/2
        x_.append(mid)
        count += 1
    return mid, count, x_


a, b, y_new = newton(1, 0.000001)
c, d, y_erfen = erfenfa(1, 2, 0.000001)
print(a, b, y_new)
print(c, d, y_erfen)
print(len(y_new))
print(len(y_erfen))
y_ = [1.414, ]*max(len(y_new), len(y_erfen))
x1 = [i for i in range(b)]
x2 = [i for i in range(d)]
plt.plot(x1, y_new)
plt.plot(x2, y_erfen)
# plt.plot(x2, y_)
plt.show()









import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1, 100) 
s_values = np.linspace(-1/3, 2, 25) 

def normal(x,s):
    return 3/2*(1-s)*x**2-3*(1-s)*x+1
def overshot(x,s):
    x_zero = 3 * s
    y = np.zeros_like(x)
    idx_x = x <= x_zero
    x_valid = x[idx_x]
    y[idx_x] = 1 - (2 * x_valid) / (3 * s) + (x_valid ** 2) / (9 * s ** 2)
    return y

plt.figure(figsize=(10, 6))

for s in s_values:
    if s <= 1/3 and s > 0:
        y = overshot(x, s)
        plt.plot(x, y, color='blue', alpha=0.7)
    else:
        y = normal(x, s)
        plt.plot(x, y, color='red', alpha=0.7)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of f(x) for different s values')
plt.legend()
plt.show()





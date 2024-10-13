import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumtrapz
from scipy.misc import derivative

# Parameters
x = np.linspace(0, 1, 100)
s_values = np.linspace(-1/3, 2, 25)
v_0 = 1  # Example value for v_0
t_0 = 1  # Example value for t_0

# Function definitions
def normal(x, s):
    return 3/2 * (1 - s) * x**2 - 3 * (1 - s) * x + 1

def overshot(x, s):
    x_zero = 3 * s
    y = np.zeros_like(x)
    idx_x = x <= x_zero
    x_valid = x[idx_x]
    y[idx_x] = 1 - (2 * x_valid) / (3 * s) + (x_valid ** 2) / (9 * s ** 2)
    return y

# Plot setup for the functions
plt.figure(figsize=(12, 8))
cmap = plt.get_cmap('viridis')

for i, s in enumerate(s_values):
    color = cmap(i / len(s_values))
    if s <= 1/3 and s > 0:
        y = overshot(x, s)
        label = f'Overshot (s={s:.2f})'
    else:
        y = normal(x, s)
        label = f'Normal (s={s:.2f})'
        
    plt.plot(x, y, color=color, alpha=0.7, label=label if i % 5 == 0 else "")

# Calculate and plot cumulative integrals for each function
plt.figure(figsize=(14, 10))

for i, s in enumerate(s_values):
    color = cmap(i / len(s_values))
    if s <= 1/3 and s > 0:
        y = overshot(x, s)
    else:
        y = normal(x, s)

    # Cumulative integral of f(x) * s
    integral_cumulative = cumtrapz(y * s, x, initial=0)
    plt.plot(x, integral_cumulative, linestyle='-', color=color, alpha=0.7, label=f's = {s:.2f}' if i % 5 == 0 else "")

plt.xlabel('x')
plt.ylabel('Cumulative Integral of f(x) * s')
plt.title('Cumulative Integral of f(x) * s for Different s Values')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize='small')
plt.grid(True)

# Calculate and plot the cumulative integral of the squared derivative
plt.figure(figsize=(14, 10))

for i, s in enumerate(s_values):
    color = cmap(i / len(s_values))
    if s <= 1/3 and s > 0:
        y = overshot(x, s)
    else:
        y = normal(x, s)

    # Compute the derivative of the function
    dy_dx = [derivative(lambda x_val: normal(x_val, s) if s > 1/3 else overshot(x_val, s), xi, dx=1e-6) for xi in x]
    derivative_squared = [(v_0 / t_0) * d**2 for d in dy_dx]

    # Cumulative integral of the squared derivative
    integral_squared_cumulative = cumtrapz(derivative_squared, x, initial=0)
    plt.plot(x, integral_squared_cumulative, linestyle='--', color=color, alpha=0.7, label=f's = {s:.2f}' if i % 5 == 0 else "")

plt.xlabel('x')
plt.ylabel('Cumulative Integral of Squared Derivative scaled by (v_0/t_0)')
plt.title('Cumulative Integral of Squared Derivative for Different s Values')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize='small')
plt.grid(True)

# Show all plots
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import time
import seaborn as sns
sns.set(style="whitegrid")
sns.color_palette("pastel")
from scipy.optimize import minimize, dual_annealing
import time

# Extended known solutions for the Thomson problem for N = 2 to N = 30
known_solutions = {
    2: 0.5,
    3: 1.7320508075688772,
    4: 3.6742346141747673,
    5: 6.474691494020355,
    6: 9.98528137423857,
    7: 14.452977414042193,
    8: 19.675287861047955,
    9: 25.759986531797333,
    10: 32.71694904598099,
    11: 40.59645049750075,
    12: 49.16525306768319,
    13: 58.85386006512183,
    14: 69.2465834491761,
    15: 80.55953282866065,
    16: 92.91165544950907,
    17: 106.046338878962,
    18: 120.08446759937802,
    19: 134.80988650091557,
    20: 150.50219314441252,
    21: 166.80361584614275,
    22: 183.90789804565918,
    23: 201.93879659327523,
    24: 220.74416582557067,
    25: 240.478732019879,
    26: 260.85541685375794,
    27: 281.9403627227507,
    28: 303.94258247334897,
    29: 326.9624638622561,
    30: 350.6336772748756
}

# Define the function to compute the potential energy
def E(phi, N, charge_products):
    fixed_position = np.array([[0, 0, 1]])
    phi_angles = phi[::2]
    theta_angles = phi[1::2]
    positions = np.stack((
        np.sin(theta_angles) * np.cos(phi_angles),
        np.sin(theta_angles) * np.sin(phi_angles),
        np.cos(theta_angles)
    ), axis=1)
    positions = np.vstack([fixed_position, positions])
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    distances += np.eye(N) * 1e-12
    i_upper = np.triu_indices_from(distances, k=1)
    return np.sum(charge_products[i_upper] / distances[i_upper])

# Using Simulated Annealing for global optimization
def E_simulated_annealing(phi, N, charge_products):
    return E(phi, N, charge_products)

# Define optimization methods
methods = ['CG', 'BFGS', 'Nelder-Mead', 'Powell', 'L-BFGS-B', 'Simulated Annealing']

# Increasing the range of N values
N_values_extended = list(range(2, 31))

# Dictionaries to hold relative errors and times for each method
relative_errors_per_method = {method: [] for method in methods}
times_per_method = {method: [] for method in methods}

# Perform optimization for each N and method
for N in N_values_extended:
    e = [1] * N
    charge_products = np.outer(e, e)
    x_0 = np.random.rand(2 * (N - 1)) * 2 * np.pi

    for method in methods[:-1]:  # all except simulated annealing
        start_time = time.time()
        result = minimize(E, x_0, args=(N, charge_products), method=method)
        end_time = time.time()

        relative_error = np.abs(result.fun - known_solutions.get(N, result.fun)) / known_solutions.get(N, result.fun)
        relative_errors_per_method[method].append(relative_error)
        times_per_method[method].append(end_time - start_time)

    # Simulated Annealing (global optimizer followed by a local optimizer)
    bounds = [(0, 2 * np.pi)] * (2 * (N - 1))
    start_time = time.time()
    result_sa = dual_annealing(E_simulated_annealing, bounds=bounds, args=(N, charge_products))
    result_local = minimize(E, result_sa.x, args=(N, charge_products), method='L-BFGS-B')
    end_time = time.time()

    relative_error_sa = np.abs(result_local.fun - known_solutions.get(N, result_local.fun)) / known_solutions.get(N, result_local.fun)
    relative_errors_per_method['Simulated Annealing'].append(relative_error_sa)
    times_per_method['Simulated Annealing'].append(end_time - start_time)

# Plotting the results with the extended N values
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Plot relative errors for each method
for method in methods:
    ax1.plot(N_values_extended, relative_errors_per_method[method], label=method)
ax1.set_title('Relative Error to Known Solutions ')
ax1.set_xlabel('N (Number of Electrons)')
ax1.set_ylabel('Relative Error')
ax1.legend()

# Right: Plot optimization times for each method
for method in methods:
    ax2.plot(N_values_extended, times_per_method[method], label=method)
ax2.set_title('Optimization Time for Different Methods')
ax2.set_xlabel('N (Number of Electrons)')
ax2.set_ylabel('Time (seconds)')
ax2.legend()

plt.tight_layout()
plt.show()

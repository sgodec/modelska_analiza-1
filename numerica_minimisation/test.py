import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import time

# Known energies for the Thomson problem up to N=50
known_energies = {
    2: 0.500000, 3: 1.732051, 4: 6.474691, 5: 9.985281, 6: 14.452977,
    7: 19.675287, 8: 25.759986, 9: 32.716949, 10: 40.596450,
    11: 49.165253, 12: 58.853287, 13: 69.324718, 14: 80.670244,
    15: 92.911655, 16: 106.050404, 17: 120.084467, 18: 135.044371,
    19: 150.919853, 20: 167.729137, 25: 275.134103, 30: 401.290899,
    35: 545.441280, 40: 707.331733, 45: 886.961098, 50: 1084.423636
}

# Energy function for given positions
def E(angles, N):
    # Fixed positions at the north and south poles
    fixed_positions = np.array([[0, 0, 1], [0, 0, -1]])
    num_variable = N - 2

    # Extract angles and scale to their appropriate ranges
    theta = angles[:num_variable] * np.pi        # Polar angle [0, pi]
    phi_angles = angles[num_variable:] * 2 * np.pi  # Azimuthal angle [0, 2*pi]

    # Positions using spherical coordinates
    positions = np.stack([
        np.sin(theta) * np.cos(phi_angles),
        np.sin(theta) * np.sin(phi_angles),
        np.cos(theta)
    ], axis=1)

    # Combine all positions
    positions = np.vstack([fixed_positions, positions])

    # Compute pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)

    # Avoid division by zero
    np.fill_diagonal(distances, np.inf)

    # Compute the energy
    energy = np.sum(1.0 / distances[np.triu_indices(N, k=1)])
    return energy

# Methods to compare (including a global optimization method)
methods = ['Nelder-Mead', 'BFGS', 'CG', 'Powell', 'TNC', 'Differential Evolution']

# Logarithmically spaced N values from 2 to 50
N_values = np.unique(np.round(np.logspace(np.log10(2), np.log10(50), num=15)).astype(int))

# Store results
results = {method: [] for method in methods}
times = {method: [] for method in methods}
errors = {method: [] for method in methods}

# Loop over different values of N
for N in N_values:
    if N < 2:
        continue

    num_variable = N - 2
    x_0 = np.random.rand(2 * num_variable)  # Initial angles between 0 and 1

    for method in methods:
        start_time = time.time()

        if method == 'Differential Evolution':
            bounds = [(0, 1)] * (2 * num_variable)
            result = differential_evolution(E, bounds, args=(N,), tol=1e-6)
        else:
            result = minimize(E, x_0, args=(N,), method=method)

        elapsed_time = time.time() - start_time

        # Store final energy and time
        results[method].append(result.fun)
        times[method].append(elapsed_time)

        # Calculate relative error
        if N in known_energies:
            error = np.abs((result.fun - known_energies[N]) / known_energies[N]) * 100
            errors[method].append((N, error))
            print(f"N = {N:<2}, Method: {method:<20}, Energy: {result.fun:>12.6f}, Time: {elapsed_time:>6.2f} sec, Error: {error:>7.4f}%")
        else:
            print(f"N = {N:<2}, Method: {method:<20}, Energy: {result.fun:>12.6f}, Time: {elapsed_time:>6.2f} sec")

# Enhanced plotting of relative error vs N
plt.figure(figsize=(14, 7))
for method in methods:
    if errors[method]:
        Ns, errs = zip(*errors[method])
        plt.plot(Ns, errs, label=method, marker='o', linewidth=2)
plt.title('Relative Error vs N for Different Methods', fontsize=16)
plt.xlabel('N (Number of Electrons)', fontsize=14)
plt.ylabel('Relative Error (%)', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.xticks(N_values, N_values)
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Enhanced plotting of execution time vs N
plt.figure(figsize=(14, 7))
for method in methods:
    plt.plot(N_values, times[method], label=method, marker='o', linewidth=2)
plt.title('Execution Time vs N for Different Methods', fontsize=16)
plt.xlabel('N (Number of Electrons)', fontsize=14)
plt.ylabel('Time (Seconds)', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.xticks(N_values, N_values)
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


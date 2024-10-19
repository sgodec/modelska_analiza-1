import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing
from scipy.integrate import quad

# Set the number of electrons for the first six Thomson problems
Ns = range(2, 14)

# List of superellipse parameters n to try
n_values = [2, 4, 6, 12]  # You can adjust this list for different superellipse shapes

# List of parameters a and b for the superellipse
a_values = [1, 2, 3]  # Different horizontal scaling factors
b_values = [1, 0.5, 0.33]  # Different vertical scaling factors (to maintain area)

# Dictionary to store energy for each n, a, b, and N
energy_data = {(n, a, b): [] for n in n_values for a, b in zip(a_values, b_values)}

# Function to compute the area of the superellipse for a given n, a, and b
def superellipse_area(n, a, b):
    def integrand(x):
        return b * (1 - (x / a) ** n) ** (1 / n)
    # Integrate from 0 to a, and multiply by 4 (for symmetry)
    return 4 * quad(integrand, 0, a)[0]

# Target area for scaling (Area of the circle, n=2, a=b=1)
A_target = np.pi

# Loop over different superellipse parameters n, a, and b
for n in n_values:
    for a, b in zip(a_values, b_values):
        A_n = superellipse_area(n, a, b)
        s_n = np.sqrt(A_target / A_n)  # Scaling factor to ensure constant area
        
        # Loop over different values of N
        for N in Ns:
            e = [1] * N  # Charges of the electrons
            
            # Initial guess for the optimization (angles in radians)
            x_0 = np.random.rand(N) * 2 * np.pi
            
            # Charge products for potential energy calculation
            charge_products = np.outer(e, e)
            
            # Function to compute the potential energy for given angular positions
            def E(phi):
                # Positions based on superellipse parameter n, a, and b
                positions = np.stack((
                    s_n * a * np.sign(np.cos(phi)) * np.abs(np.cos(phi))**(2/n),
                    s_n * b * np.sign(np.sin(phi)) * np.abs(np.sin(phi))**(2/n)
                ), axis=1)
                
                # Calculate pairwise distances between electrons
                diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
                distances = np.linalg.norm(diff, axis=-1)
                
                i_upper = np.triu_indices_from(distances, k=1)
                return np.sum(charge_products[i_upper] / distances[i_upper])
            
            # Perform the optimization to minimize the potential energy
            result = dual_annealing(E, bounds=[(0, 2 * np.pi)] * N)
            
            # Store the optimized energy for this combination of N, n, a, and b
            energy_data[(n, a, b)].append(result.fun)

# Now we can plot the energy for different N, n, a, and b
plt.figure(figsize=(10, 6))
for (n, a, b), energies in energy_data.items():
    plt.plot(Ns, energies, label=f'n = {n}, a = {a}, b = {b}', marker='o')

plt.xlabel('Number of electrons (N)')
plt.ylabel('Minimum potential energy (E)')
plt.yscale('log')
plt.title('Energy vs Number of Electrons for Different Superellipse Parameters (a, b)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('energy_vs_N_for_different_n_a_b.png')
plt.show()


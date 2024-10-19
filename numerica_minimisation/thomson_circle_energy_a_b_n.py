import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing
from scipy.integrate import quad
from matplotlib.cm import get_cmap

Ns = range(2, 20)

# Fixed n = 2 (ellipse case)
n = 2

# Wide variety of aspect ratios
aspect_ratios = np.linspace(0.1, 10, 15)  # Aspect ratio a/b, ranging from thin to wide ellipses

# Fixed b = 1, since we are focusing on the ratio a/b
b = 1

# Dictionary to store energy for each aspect ratio and N
energy_data = {ratio: [] for ratio in aspect_ratios}

# Function to compute the area of the ellipse for given a and b (n=2 case)
def ellipse_area(a, b):
    return np.pi * a * b

# Target area for scaling (Area of the circle, n=2, a=b=1)
A_target = np.pi

# Loop over different aspect ratios
for ratio in aspect_ratios:
    a = ratio * b  # Set 'a' based on the desired aspect ratio
    
    A_ab = ellipse_area(a, b)
    s_ab = np.sqrt(A_target / A_ab)  # Scaling factor to ensure constant area
    
    # Loop over different values of N
    for N in Ns:
        e = [1] * N  # Charges of the electrons
        
        # Initial guess for the optimization (angles in radians)
        x_0 = np.random.rand(N) * 2 * np.pi
        
        # Charge products for potential energy calculation
        charge_products = np.outer(e, e)
        
        # Function to compute the potential energy for given angular positions
        def E(phi):
            # Positions based on ellipse parameters a and b
            positions = np.stack((
                s_ab * a * np.cos(phi),
                s_ab * b * np.sin(phi)
            ), axis=1)
            
            # Calculate pairwise distances between electrons
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            distances = np.linalg.norm(diff, axis=-1)
            
            i_upper = np.triu_indices_from(distances, k=1)
            return np.sum(charge_products[i_upper] / distances[i_upper])
        
        # Perform the optimization to minimize the potential energy
        result = dual_annealing(E, bounds=[(0, 2 * np.pi)] * N)
        
        # Store the optimized energy for this aspect ratio and N
        energy_data[ratio].append(result.fun)

# Now we can plot the energy for different N and aspect ratios
plt.figure(figsize=(10, 6))
cmap = get_cmap('viridis')  # You can change 'viridis' to other colormaps like 'plasma', 'inferno', etc.
num_colors = len(energy_data)
colors = cmap(np.linspace(0, 1, num_colors))

# Plot each aspect ratio with its assigned color from the colormap
for (i, (ratio, energies)) in enumerate(energy_data.items()):
    plt.plot(Ns, energies, label=f'Aspect ratio a/b = {ratio:.2f}', marker='o', color=colors[i])

plt.xlabel('Number of electrons (N)')
plt.ylabel('Minimum potential energy (E)')
plt.yscale('log')
plt.title('Energy vs Number of Electrons for Different Aspect Ratios (a/b)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('energy_vs_N_for_different_aspect_ratios_final.png')
plt.show()

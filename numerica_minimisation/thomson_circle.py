import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.patches import Circle
from scipy.integrate import quad
from scipy.optimize import dual_annealing

# Set the number of electrons for the first six Thomson problems
Ns = range(2, 14)

n = 12  # Superellipse parameter
# Create a figure with subplots
fig = plt.figure(figsize=(18, 12))

def superellipse_area(n):
    def integrand(x):
        return (1 - x**n)**(1/n)
    # Integrate from 0 to 1, and multiply by 4 (for symmetry)
    return 4 * quad(integrand, 0, 1)[0]

# Target area for scaling (Area of the circle, n=2)
A_target = np.pi

# Loop over the different values of N
for idx, N in enumerate(Ns, start=1):
    e = [1] * N  # Charges of the electrons
    
    # Initial guess for the optimization (angles in radians)
    x_0 = np.random.rand(N) * 2 * np.pi
    
    # Charge products for potential energy calculation
    charge_products = np.outer(e, e)
    
     # Compute the area of the superellipse for the current n
    A_n = superellipse_area(n)
    
    # Scaling factor to ensure constant area
    s_n = np.sqrt(A_target / A_n)

    # Initialize the subplot
    def E(phi):
        # Fixed position for the first electron at the north pole
        positions = np.stack((
           s_n* np.sign(np.cos(phi)) * np.abs(np.cos(phi))**(2/n),
            s_n * np.sign(np.sin(phi)) * np.abs(np.sin(phi))**(2/n)
        ), axis=1)
    
    
        # Calculate pairwise distances between electrons
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=-1)
        
        i_upper = np.triu_indices_from(distances, k=1)
        return np.sum(charge_products[i_upper] / distances[i_upper])
    
    # Perform the optimization to minimize the potential energy
    result = dual_annealing(E,bounds=[(0, 2 * np.pi)] * (N))

    
    phi_opt = result.x
    
    positions = np.stack((
       s_n* np.sign(np.cos(phi_opt)) * np.abs(np.cos(phi_opt))**(2/n),
        s_n* np.sign(np.sin(phi_opt)) * np.abs(np.sin(phi_opt))**(2/n)
    ), axis=1)
    
    # Initialize the subplot
    ax = fig.add_subplot(3, 4, idx)
    
    theta = np.linspace(0, 2 * np.pi, 1000)
    superellipse_x = s_n*np.sign(np.cos(theta)) * np.abs(np.cos(theta))**(2/n)
    superellipse_y = s_n*np.sign(np.sin(theta)) * np.abs(np.sin(theta))**(2/n)

    ax.fill(superellipse_x, superellipse_y, color='hotpink', alpha=0.5)

    electron_colors = plt.cm.plasma(np.linspace(0, 1, N))
    ax.scatter(
        positions[:, 0], positions[:, 1],
        color=electron_colors, s=150, edgecolors='k', linewidth=0.5
    )
    
    # Set plot limits and labels
    ax.set_title(f"N = {N}, E = {result.fun:.2f}", fontsize=10)

    ax.set_xlabel('')
    ax.set_ylabel('')

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove grid lines
    ax.grid(False)
    ax.set_aspect('equal')
    ax.set_axis_off()

# Adjust layout and display the plot
plt.savefig(f'energy_circle_{n}')
plt.show()

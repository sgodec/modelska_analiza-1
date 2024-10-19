import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Set the number of electrons for the first six Thomson problems
Ns = 8

# Create a figure with subplots
fig, ax = plt.subplots(figsize=(16,9))
# Loop over the different values of N
for N in np.logspace(np.log10(2),np.log10(20ax.legend(handles=[line1, line2])0),num=20,dtype=int):
    e = [1] * N  # Charges of the electrons
    print(N) 
    # Initial guess for the optimization (angles in radians)
    x_0 = np.random.rand(2 * (N - 1)) * 2 * np.pi
    
    # Charge products for potential energy calculation
    charge_products = np.outer(e, e)
    
    def E(phi):
        # Fixed position for the first electron at the north pole
        fixed_position = np.array([[0, 0, 1]])
    
        # Extract phi (azimuthal) and theta (polar) angles for other electrons
        phi_angles = phi[::2]
        theta_angles = phi[1::2]
    
        # Convert spherical coordinates to Cartesian coordinates
        positions = np.stack((
            np.sin(theta_angles) * np.cos(phi_angles),
            np.sin(theta_angles) * np.sin(phi_angles),
            np.cos(theta_angles)
        ), axis=1)
    
        # Include the fixed electron's position
        positions = np.vstack([fixed_position, positions])
    
        # Calculate pairwise distances between electrons
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=-1)
        
        # Avoid division by zero by adding a small epsilon where distances are zero
        distances += np.eye(N) * 1e-12
    
        # Sum over unique pairs to calculate total potential energy
        i_upper = np.triu_indices_from(distances, k=1)
        return np.sum(charge_products[i_upper] / distances[i_upper])
    
    result = minimize(E, x_0, method='CG')
    phi_opt = result.fun
    ax.scatter(N,phi_opt/(N*(N-1)/2), marker='o', linestyle='-')
plt.title('Minimized Energy vs. Number of Electrons (N)')
plt.xlabel('Number of Electrons (N)')
plt.ylabel('Minimized Energy')
plt.grid(True)
    
# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('thomson_solution_energy_N.png', dpi=500)
plt.show()

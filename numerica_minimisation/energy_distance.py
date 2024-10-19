import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# Set the number of electrons for the first six Thomson problems
phi_opt_ar = []
distances_min_ar = []
distances_max_ar = []
distances_sigma_ar = []
distances_average_ar = []
# Create a figure with subplots
fig, ax = plt.subplots(1,2,figsize=(16,9))

N_new = np.logspace(np.log10(2),np.log10(150),num=20,dtype=int)
for N in N_new:
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
    phi = result.x
    phi_opt = result.fun
    fixed_position = np.array([[0, 0, 1]])

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
    i_upper = np.triu_indices_from(distances, k=1)
    distances_average = np.average(distances[i_upper].flatten())
    distances_max = np.max(distances[i_upper].flatten())
    distances_min = np.min(distances[i_upper].flatten())
    distances_sigma = np.std(distances[i_upper].flatten())
    phi_opt_ar.append(phi_opt/(N*(N-1)/2))
    distances_min_ar.append(distances_min)
    distances_max_ar.append(distances_max)
    distances_sigma_ar.append(distances_sigma)
    distances_average_ar.append(distances_average)

ax[0].scatter(N_new,phi_opt_ar, marker='o', linestyle='-',s = 60,c='darkviolet',label='$E(N)$')
ax[0].plot(N_new,N_new*0+1, linestyle='--',c='darkgreen',label='$\\lim_{N \\to \\infty} \\frac{2 E(N)}{N (N-1) }$')
ax[1].scatter(N_new,distances_average_ar, marker='v',s=60, linestyle='-',c='hotpink',label = '$\\bar{d}$')
ax[1].scatter(N_new,distances_min_ar, marker='P', s=60,linestyle='-',c='royalblue',label='min$(d)$')
ax[1].scatter(N_new,distances_max_ar, marker='*',s =60, linestyle='-',c='darkseagreen',label='max$(d)$')
ax[1].scatter(N_new,distances_sigma_ar, marker='o', s=60,linestyle='-',c='firebrick',label='$\\sigma_d$')


ax[1].set_title('Distances (d) between electrons')
ax[1].set_xlabel('Number of Electrons (N)')
ax[1].set_ylabel('Distances')
ax[0].set_title('Minimised energy (E) vs number of elctrons (N)')
ax[0].set_xlabel('Number of Electrons (N)')
ax[0].set_ylabel('Minimized Energy')
ax[0].legend()
ax[1].legend()
ax[0].grid(True)
ax[1].grid(True)
    
# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('thomson_solution_distance_N.png', dpi=500)
plt.show()

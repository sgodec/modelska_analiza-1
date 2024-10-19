import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
sns.set(style="whitegrid", palette='pastel')

# Set the number of electrons for the first six Thomson problems
Ns = range(2, 40)
dipol_ar = []
quad_norm_ar = []

# Create a figure and axis
fig, ax1 = plt.subplots(figsize=(12, 8))
N_new = []

# Loop over the different values of N
for idx, N in enumerate(Ns, start=1):
    e = [1] * N  # Charges of the electrons
    
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
    
    # Perform the optimization to minimize the potential energy
    result = minimize(E, x_0, method='CG')
    
    phi_opt = result.x
    fixed_position = np.array([[0, 0, 1]])
    phi_angles = phi_opt[::2]
    theta_angles = phi_opt[1::2]
    
    positions = np.stack((
        np.sin(theta_angles) * np.cos(phi_angles),
        np.sin(theta_angles) * np.sin(phi_angles),
        np.cos(theta_angles)
    ), axis=1)
    positions = np.vstack([fixed_position, positions])

    Q = np.zeros((3, 3))
    for pos in positions:
        for i in range(3):
            for j in range(3):
                Q[i, j] += 3 * pos[i] * pos[j] - (i == j) 
    
    # Normalize the quadrupole tensor by the number of electrons
    
    # Calculate the quadrupole moment norm (trace of the tensor for simplicity)
    quad_norm = np.linalg.norm(Q)
    dipol = np.linalg.norm(np.sum(positions,axis=0)) 
    quad_norm_ar.append(quad_norm)
    dipol_ar.append(dipol)

    N_new.append(N)

# Plot the dipole moment on the primary y-axis
ax1.scatter(N_new, dipol_ar, marker='v', c='hotpink', s=60, label='$V_p(N)$')
ax1.set_xlabel('Number of electrons N')
ax1.set_ylabel('$e_i r_i$', color='hotpink')
ax1.set_yscale('symlog')
ax1.tick_params(axis='y', labelcolor='hotpink')

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()
ax2.scatter(N_new, quad_norm_ar, marker='o', c='royalblue', s=60, label='$V_q(N)$')
ax2.set_ylabel('$q_{i,j} r_i r_j$', color='royalblue')
ax2.set_yscale('symlog')
ax2.tick_params(axis='y', labelcolor='royalblue')

# Add titles and legends
ax1.set_title('Dipole and Quadrupole Moment as a Function of N')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Tight layout and save the figure
plt.tight_layout()
plt.savefig('dipol_quadropole_single_plot.png')

plt.show()

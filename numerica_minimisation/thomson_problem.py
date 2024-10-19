import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Set the number of electrons for the first six Thomson problems
Ns = range(2,14)

# Create a figure with subplots
fig = plt.figure(figsize=(18, 12))

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
    
    # Compute the convex hull of the positions
    if N >= 4:
        hull = ConvexHull(positions)
    else:
        # For N < 4, ConvexHull may not work properly, so we manually define edges
        hull = None
    
    # Plot the electrons on the sphere and connect them
    # Create a sphere mesh
    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
    x_sphere = np.sin(v) * np.cos(u)
    y_sphere = np.sin(v) * np.sin(u)
    z_sphere = np.cos(v)
    
    # Initialize the subplot
    ax = fig.add_subplot(3, 4, idx, projection='3d')
    
    # Plot the sphere surface
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='hotpink', alpha=0.3, linewidth=0)
    

    electron_colors = cm.plasma(np.linspace(0, 1, N))
    ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        color=electron_colors, s=150, edgecolors='k', linewidth=0.5
    )
    
    # Connect the electrons
    if hull is not None:
        # For N >= 4, use the convex hull to find the edges
        for simplex in hull.simplices:
            simplex = np.append(simplex, simplex[0])  # Close the loop
            ax.plot(positions[simplex, 0], positions[simplex, 1], positions[simplex, 2],color='black', linewidth=1)
    else:
        # For N < 4, manually plot the edges
        for i in range(N):
            for j in range(i + 1, N):
                ax.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    [positions[i, 2], positions[j, 2]],
                    color='black', linewidth=1
                )
    
        # Set plot limits and labels
    ax.set_title(f"N = {N}, E ={result.fun:.2f}",fontsize = 14)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Remove grid lines
    ax.grid(False)
    ax.margins(0, 0, 0)

    # Remove panes (background walls)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Remove pane edge lines
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1,1,1])  # Equal aspect ratio
    
    # Adjust viewing angle for better visualization
    ax.view_init(elev=20, azim=30)

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('thomson_solution.png', dpi=500)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator

N = 2
e = [1] * (N+2)
x_0 = np.random.rand(2 * (N - 1)) 
print(x_0)
charge_products = np.outer(e, e)

def E(phi,n=2):
    #position of first electron
    fixed_position = np.array([[0, 0, 1]])
    fixed_position_1 = np.array([[0, 0, -1]])
    fixed_position_2 = np.array([[1, 1, 0]])

    #position of other electrons
    positions = np.stack(((np.cos(phi[::2])*np.sin(phi[1::2]))**(2//n),(np.sin(phi[::2])*np.sin(phi[1::2]))**(2//n),np.cos(phi[1::2])**(2//n)),axis=1)
    positions = np.vstack([fixed_position,positions])
    positions = np.vstack([fixed_position_1,positions])

    #differece of diff[i,j] = position[i]-position[j]
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    #norm
    distances = np.linalg.norm(diff, axis=-1)

    i_upper = np.triu_indices_from(distances, k=1) 

    return np.sum(charge_products[i_upper] / distances[i_upper])


# Generate a grid of phi_1 (longitude) and theta_1 (latitude)
phi_1 = np.linspace(0, 2 * np.pi, 100)  
theta_1 = np.linspace(0.01, np.pi, 50)


theta_1, phi_1 = np.meshgrid(theta_1,phi_1)
phi = np.vstack([phi_1.ravel(),theta_1.ravel()]).T

x = np.cos(phi_1)*np.sin(theta_1)
y = np.sin(phi_1)*np.sin(theta_1)
z = np.cos(theta_1)

energy = np.apply_along_axis(E, 1, phi).reshape(phi_1.shape)
print(energy)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    x, y, z, facecolors=plt.cm.coolwarm(np.log(energy)),shade = False)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_zlabel('')

# Remove axis ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Remove grid lines
ax.grid(False)

# Remove panes (background walls)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Remove pane edge lines
ax.xaxis.pane.set_edgecolor('none')
ax.yaxis.pane.set_edgecolor('none')
ax.zaxis.pane.set_edgecolor('none')
ax.set_axis_off()


# Create the color bar
mappable = plt.cm.ScalarMappable(cmap='coolwarm')
mappable.set_array(np.log(energy))
fig.colorbar(
    mappable, ax=ax, shrink=0.4, aspect=5,
    label='Potencial $\phi$ N=3 '
)
#plt.savefig('potencial_plot.pdf', dpi=300, bbox_inches='tight')
plt.show()

#result = scp.optimize.minimize(E,x_0,method='Nelder-Mead')

#print("Optimized angles:", np.mod(result.x, 2 * np.pi))
#print("Minimum energy:", result.fun)
#print("Convergence message:", result.message)

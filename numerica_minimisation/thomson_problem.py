import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

N = 2
e = [1] * (N)
x_0 = np.random.rand(2 * (N - 1)) 
print(x_0)
charge_products = np.outer(e, e)

def E(phi,n=2):
    #position of first electron
    fixed_position = np.array([[0, 0, 1]])

    #position of other electrons
    positions = np.stack(((np.cos(phi[::2])*np.sin(phi[1::2]))**4,(np.sin(phi[::2])*np.sin(phi[1::2]))**4,np.cos(phi[1::2])**4),axis=1)
    positions = np.vstack([fixed_position,positions])

    #differece of diff[i,j] = position[i]-position[j]
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    #norm
    distances = np.linalg.norm(diff, axis=-1)

    i_upper = np.triu_indices_from(distances, k=1) 

    return np.sum(charge_products[i_upper] / distances[i_upper])


result = scp.optimize.minimize(E,x_0,method='Nelder-Mead')

print("Optimized angles:", np.mod(result.x, 2 * np.pi))
print("Minimum energy:", result.fun)
print("Convergence message:", result.message)

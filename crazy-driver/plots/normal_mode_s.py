import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

#constants
l = 200
t_0 = 10

def speed_normal(t,v_0):
    return 1/2 * (v_0-l/t_0) * t**3/t_0**2 - 3/2*(v_0-l/t_0)*t**2/t_0 + v_0*t 

def speed_under(t,v_0):
    y = np.zeros_like(t) + l
    y[np.where(t <= 3*l/v_0)] = 1/2 * (2/3*v_0) * t[np.where(t <= 3*l/v_0)]**3 /(3*l/v_0)**2 - 3/2*(2/3 *v_0)*t[np.where(t <= 3*l/v_0)]**2 /(3*l/v_0) + v_0* t[np.where(t <= 3*l/v_0)]
    return y


t = np.linspace(0, 10, 400)
v_0 = np.linspace(-50, 150, 28)
v_0 = np.append(v_0,3*l/t_0)
print(v_0)
plt.figure(figsize=(14, 8), dpi=100)

colors = cm.viridis(np.linspace(0, 1, len(v_0)))

for idx, v_0 in enumerate(v_0):
    if v_0 < 3 *l/t_0 or v_0 < 0:
        y_values = speed_normal(t, v_0)
        plt.plot(t, y_values, label=f'v_0 = {v_0:.2f} m/s', linestyle='--',lw=2, color=colors[idx])
    elif v_0 == 3*l/t_0:  
        y_values = speed_normal(t, v_0)
        plt.plot(t, y_values, label=f'v_0 = {v_0:.2f} m/s', color='r')
        
    else:
        y_values = speed_under(t, v_0)
        plt.plot(t, y_values, label=f'v_0 = {v_0:.2f} m/s', linestyle='-', color=colors[idx])

plt.title('Graf $x(t)^{*}$ v odvisnosti od zacetne hitrosti', fontsize=20)
plt.xlabel('$t[s]$', fontsize=16)
plt.ylabel('$x(t)^*[m]$', fontsize=16)
plt.legend(title='$v_0$', fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
regime_change_y = 3 * l / t_0
plt.tight_layout()

plt.show()

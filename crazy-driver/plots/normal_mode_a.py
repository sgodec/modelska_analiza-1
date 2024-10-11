import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

#constants
l = 200
t_0 = 10

def speed_normal(t,v_0):
    return (3 * (v_0-l/t_0) * 1/t_0 *(t/t_0-1))**2 

def speed_under(t,v_0):
    y = np.zeros_like(t)
    y[np.where(t <= 3*l/v_0)] =( 3 * (2/3*v_0) * (t[np.where(t <= 3*l/v_0)]/(3*l/v_0) - 1)* 1/(3*l/v_0))**2
    return y


t = np.linspace(0, 10, 400)
v_0 = np.linspace(-50, 150, 27)
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
        plt.plot(t, y_values, label=f'v_0 = $3 L/t_0$ m/s', color='r')
        
    else:
        y_values = speed_under(t, v_0)
        plt.plot(t, y_values, label=f'v_0 = {v_0:.2f} m/s', linestyle='-', color=colors[idx])
ymin, ymax = plt.ylim()
regime_change_y = 3 * l / t_0
plt.axhline(y=0, color='Orange',lw=2, linestyle='--', label=r'v_0 = $L/t_0$ m/s')
plt.title('Graf $(a(t)^{*})^2$ v odvisnosti od zacetne hitrosti', fontsize=20)
plt.xlabel('$t[s]$', fontsize=16)
plt.ylabel('$(a(t)^*)^2 ([m/s^2])^2 $', fontsize=16)
plt.grid(True)
plt.legend(title='$v_0$', fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlim([0,t_0])
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

#constants
l = 200
t_0 = 10

def speed_normal(v_0):
    return (3 * (v_0-l/t_0)**2 * 1/t_0) 

def speed_under(v_0):
    return 3 * (2/3*v_0)**2* 1/(3*l/v_0)


v_0 = np.linspace(-50, 150, 100)
v_0 = np.append(v_0,3*l/t_0)
plt.figure(figsize=(14, 8), dpi=100)

colors = cm.viridis(np.linspace(0, 1, len(v_0)))

for idx, v_0 in enumerate(v_0):
    if v_0 < 3 *l/t_0 or v_0 < 0:
        y_values = speed_normal(v_0)
        if idx % 5 == 0:
            plt.scatter(v_0, y_values, label=f'v_0 = {v_0:.2f} m/s',lw=2, color=colors[idx])
        else:
            plt.scatter(v_0, y_values,lw=2, color=colors[idx])
    elif v_0 == 3*l/t_0:  
        y_values = speed_normal(v_0)
        plt.scatter(v_0, y_values, label=f'v_0 = $3 L/t_0$ m/s', color='r')
        
    else:
        y_values = speed_under(v_0)
        
        if idx % 5 == 0:
            plt.scatter(v_0, y_values, label=f'v_0 = {v_0:.2f} m/s',lw=2, color=colors[idx])
        else:
            plt.scatter(v_0, y_values,lw=2, color=colors[idx])
ymin, ymax = plt.ylim()
regime_change_y = 3 * l / t_0
plt.title('Graf $\int_0^{t_0} (a(t)^{*})^2 dt$ v odvisnosti od zacetne hitrosti', fontsize=20)
plt.xlabel('$v_0[m/s]$', fontsize=16)
plt.ylabel('$\int_0^{t_0} (a(t)^{*})^2 dt $', fontsize=16)
plt.axvline(l/t_0, color='Orange',lw=2, linestyle='--', label=r'v_0 = $L/t_0$ m/s')
plt.axvline(3*l/t_0, color='red',lw=2, linestyle='--', label=r'v_0 = $3L/t_0$ m/s')
plt.text(l/t_0 * 2, 1000, r'$v_0 > L/t_0 $', horizontalalignment='center', fontsize=14, color='orange')
plt.text(l/t_0 * 6, 1000, r'$v_0 > 3 L/t_0 $', horizontalalignment='center', fontsize=14, color='Red')
plt.grid(True)
plt.legend(title='$v_0$', fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


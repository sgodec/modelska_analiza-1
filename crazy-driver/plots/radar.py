import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

#constants
l = 200
t_0 = 10

v_radar = 40
def speed_normal(t,v_0):
    y = np.zeros_like(t)+v_radar
    t_1 = 3*(v_radar *t_0-l)/(v_radar-v_0)
    print(t_1)
    y[np.where(t <= t_1)] = 3/2 * (v_radar*(t_0/t_1-1)+(v_0-l/t_1)) * t[np.where(t <= t_1)]**2/t_1**2 - 3* (v_radar*(t_0/t_1-1)+(v_0-l/t_1))*t[np.where(t <= t_1)]/t_1 + v_0 
    return y

t = np.linspace(0, 10, 400)
v_0 = np.linspace(0, 25, 10)
print(v_0)
plt.figure(figsize=(14, 8), dpi=100)

colors = cm.viridis(np.linspace(0, 1, len(v_0)))

for idx, v_0 in enumerate(v_0):
        y_values = speed_normal(t, v_0)
        plt.plot(t, y_values, label=f'v_0 = {v_0:.2f} m/s', linestyle='--',lw=2, color=colors[idx])
ymin, ymax = plt.ylim()
regime_change_y = 3 * l / t_0
plt.axhline(y=l/t_0, color='Orange',lw=2, linestyle='--', label=r'v_0 = $L/t_0$ m/s')
plt.title('Graf $v(t)^{*}$ v odvisnosti od zacetne hitrosti', fontsize=20)
plt.xlabel('$t[s]$', fontsize=16)
plt.ylabel('$v(t)^*[m/s]$', fontsize=16)
plt.grid(True)
plt.legend(title='$v_0$', fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.text(t.mean(), 3*l/t_0 * 1.5, r'$v_0 > 3 L/t_0 $', horizontalalignment='center', fontsize=12, color='red')
plt.axhspan(regime_change_y, ymax, facecolor='gray', alpha=0.3)
plt.xlim([0,t_0])
plt.tight_layout()
plt.show()


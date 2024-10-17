import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

food_dict = pd.read_csv('zivila_2.csv')

sns.set(style="whitegrid")
set2_palette = sns.color_palette("Set2")[2:]
sns.set_palette(set2_palette)

food_dict['Teza_g'] = [100.0] * len(food_dict)

masa = 70 

minimum = 'energija_kcal'

# Initial Constraints (for Diet 1)
min_energija = [None, 'energija_kcal']
max_energija = [None, 'energija_kcal']
min_mascobe = [70, 'mascobe_g']
max_mascobe = [None, 'mascobe_g']
min_oghidrati = [310.0, 'ogljikovi_hidrati_g']
max_oghidrati = [None, 'ogljikovi_hidrati_g']
min_proteini = [50.0, 'proteini_g']
max_proteini = [None, 'proteini_g']
min_ca = [1000.0, 'Ca_mg']
max_ca = [None, 'Ca_mg']
min_fe = [18.0, 'Fe_mg']
max_fe = [None, 'Fe_mg']
min_c = [None, 'Vitamin_C_mg']
max_c = [None, 'Vitamin_C_mg']
min_k = [None, 'Kalij_mg']
max_k = [None, 'Kalij_mg']
min_na = [None, 'Natrij_mg']
max_na = [None, 'Natrij_mg']
min_cena = [None, 'Cena_EUR']
max_cena = [None, 'Cena_EUR']
min_Teza_g = [None, 'Teza_g']
max_Teza_g = [2000.0, 'Teza_g']
min_sladkor = [None, 'sladkor_g']
max_sladkor = [None, 'sladkor_g']
min_vlaknine = [None, 'vlaknine_g']
max_vlaknine = [None, 'vlaknine_g']
min_satfat = [None, 'nasicene_mascobe_g']
max_satfat = [None, 'nasicene_mascobe_g']
min_Histidine_g = [None, 'Histidine_g']
max_Histidine_g = [None, 'Histidine_g']
min_Isoleucine_g = [None, 'Isoleucine_g']
max_Isoleucine_g = [None, 'Isoleucine_g']
min_Leucine_g = [None, 'Leucine_g']
max_Leucine_g = [None, 'Leucine_g']
min_Lysine_g = [None, 'Lysine_g']
max_Lysine_g = [None, 'Lysine_g']
min_Methionine_g = [None, 'Methionine_g']
max_Methionine_g = [None, 'Methionine_g']
min_Phenylalanine_g = [None, 'Phenylalanine_g']
max_Phenylalanine_g = [None, 'Phenylalanine_g']
min_Threonine_g = [None, 'Threonine_g']
max_Threonine_g = [None, 'Threonine_g']
min_Tryptophan_g = [None, 'Tryptophan_g']
max_Tryptophan_g = [None, 'Tryptophan_g']

arr_diet1 = [
    min_energija, max_energija, min_mascobe, max_mascobe,
    min_oghidrati, max_oghidrati, min_proteini, max_proteini,
    min_ca, max_ca, min_fe, max_fe,min_cena,max_cena,min_Teza_g,max_Teza_g
]

min_c[0] = 60.0
min_k[0] = 3500.0
min_na[0] = 500.0
max_na[0] = 2400.0
arr_diet2 = arr_diet1.copy()
arr_diet2.extend([
    min_c, max_c, min_k, max_k, min_na, max_na
])

min_sladkor = [None, 'sladkor_g']
max_sladkor = [25.0, 'sladkor_g']
min_vlaknine = [30.0, 'vlaknine_g']
max_vlaknine = [None, 'vlaknine_g']
max_satfat = [15.0, 'nasicene_mascobe_g']
min_Histidine_g = [masa * 0.01, 'Histidine_g']
max_Histidine_g = [None, 'Histidine_g']
min_Isoleucine_g = [masa * 0.02, 'Isoleucine_g']
max_Isoleucine_g = [None, 'Isoleucine_g']
min_Leucine_g = [0.039 * masa, 'Leucine_g']
max_Leucine_g = [None, 'Leucine_g']
min_Lysine_g = [0.03 * masa, 'Lysine_g']
max_Lysine_g = [None, 'Lysine_g']
min_Methionine_g = [0.015 * masa, 'Methionine_g']
max_Methionine_g = [None, 'Methionine_g']
min_Phenylalanine_g = [0.025 * masa, 'Phenylalanine_g']
max_Phenylalanine_g = [None, 'Phenylalanine_g']
min_Tryptophan_g = [0.004 * masa, 'Tryptophan_g']
max_Tryptophan_g = [None, 'Tryptophan_g']
arr_diet3 = arr_diet2.copy()
arr_diet3.extend([
    min_sladkor, max_sladkor, min_vlaknine, max_vlaknine,
    min_satfat, max_satfat, min_Histidine_g, max_Histidine_g,
    min_Isoleucine_g, max_Isoleucine_g, min_Leucine_g, max_Leucine_g,
    min_Lysine_g, max_Lysine_g, min_Methionine_g, max_Methionine_g,
    min_Phenylalanine_g, max_Phenylalanine_g, min_Tryptophan_g, max_Tryptophan_g
])

def linear_program(arr, minimum, bounds):
    def constraint(arr):
        b = np.array([(-1) ** (i + 1) * arr[i][0] for i in range(len(arr)) if arr[i][0] is not None])
        A = np.array([(-1) ** (i + 1) * np.array(food_dict[arr[i][1]].tolist()) for i in range(len(arr)) if arr[i][0] is not None])
        return A, b

    A, b = constraint(arr)
    c = np.array(food_dict[minimum].tolist())
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    if result.success:
        return result.x, result.fun, result.slack, A, b, result['ineqlin']['marginals']
    else:
        return None

x_values = np.linspace(20, 1000, 20)

num_foods_diet1 = []
num_foods_diet2 = []
num_foods_diet3 = []

price_diet1 = []
price_diet2 = []
price_diet3 = []

for x in x_values:
    bounds = [(0, x / 100.0)] * len(food_dict)  # Divide by 100 because quantities are in 100g units

    # Solve for Diet 1
    solution1 = linear_program(arr_diet1, minimum, bounds)
    if solution1 is not None:
        x1, fun1, slack1, A1, b1, dual1 = solution1
        num_included1 = np.sum(x1 > 0)
        price_diet1.append(fun1)
    else:
        num_included1 = 0
        price_diet1.append(np.nan)  # Use NaN to indicate infeasibility
    num_foods_diet1.append(num_included1)

    solution2 = linear_program(arr_diet2, minimum, bounds)
    if solution2 is not None:
        x2, fun2, slack2, A2, b2, dual2 = solution2
        num_included2 = np.sum(x2 > 0)
        price_diet2.append(fun2)
    else:
        num_included2 = 0
        price_diet2.append(np.nan)
    num_foods_diet2.append(num_included2)

    solution3 = linear_program(arr_diet3, minimum, bounds)
    if solution3 is not None:
        x3, fun3, slack3, A3, b3, dual3 = solution3
        num_included3 = np.sum(x3 > 0)
        price_diet3.append(fun3)
    else:
        num_included3 = 0
        price_diet3.append(np.nan)
    num_foods_diet3.append(num_included3)

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

axs[0].scatter(x_values, num_foods_diet1, marker='o', s=50, label='Diet 1')
axs[0].scatter(x_values, num_foods_diet2, marker='s', s=50, label='Diet 2')
axs[0].scatter(x_values, num_foods_diet3, marker='^', s=50, label='Diet 3')
axs[0].set_title('Stevilo razlicnih zivil v odvisnosti od kolicine zivila')
axs[0].set_xlabel('Omejena kolicina zivila [g]')
axs[0].set_ylabel('Stevilo zivil')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(x_values, np.array(price_diet1), marker='o', label='Diet 1')
axs[1].plot(x_values, np.array(price_diet2), marker='s', label='Diet 2')
axs[1].plot(x_values, np.array(price_diet3), marker='^', label='Diet 3')
axs[1].set_title('Minimizirana energijska vrednost v odvisnosti od kolicine zivila')
axs[1].set_xlabel('Omejena kolicina zivila [g]')
axs[1].set_ylabel('Minimizirana energijska vrednost [EUR]')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig('stevilo_min_kcal.pdf', format='pdf', dpi=300)
plt.show()

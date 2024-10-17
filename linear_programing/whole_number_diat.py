import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import milp
from scipy.optimize import Bounds, LinearConstraint, milp
import seaborn as sns
sns.set(style="whitegrid")
set2_palette = sns.color_palette("Set2")[2:]
sns.set_palette(set2_palette)

food_dict = pd.read_csv('grocery_data.csv')

print(food_dict)
masa = 70

#Optimiziramo
minimum = 'proteini_g'

#Omejitve
min_energija = [2000,'energija_kcal']
max_energija = [None,'energija_kcal']
min_mascobe = [70,'mascobe_g']
max_mascobe = [None,'mascobe_g']
min_oghidrati = [310.,'ogljikovi_hidrati_g']
max_oghidrati = [None,'ogljikovi_hidrati_g']
min_proteini = [None,'proteini_g']
max_proteini = [None,'proteini_g']
min_ca = [1000.,'Ca_mg']
max_ca = [None,'Ca_mg']
min_fe = [18.,'Fe_mg']
max_fe = [None,'Fe_mg']
min_c = [None,'Vitamin_C_mg']
max_c = [None,'Vitamin_C_mg']
min_k = [None,'Kalij_mg']
max_k = [None,'Kalij_mg']
min_na = [None,'Natrij_mg']
max_na = [None,'Natrij_mg']
min_cena = [None,'Cena_EUR']
max_cena = [25,'Cena_EUR']
min_Teza_g = [None,'quantity_g']
max_Teza_g = [2000,'quantity_g']
min_sladkor = [None,'sladkor_g']
max_sladkor = [None,'sladkor_g']
min_vlaknine = [None,'vlaknine_g']
max_vlaknine = [None,'vlaknine_g']
min_satfat = [None,'nasicene_mascobe_g']
max_satfat = [None,'nasicene_mascobe_g']
min_Histidine_g = [None,'Histidine_g']
max_Histidine_g = [None,'Histidine_g']
min_Isoleucine_g = [None,'Isoleucine_g']
max_Isoleucine_g = [None,'Isoleucine_g']
min_Leucine_g = [None,'Leucine_g']
max_Leucine_g = [None,'Leucine_g']
min_Lysine_g = [None,'Lysine_g']
max_Lysine_g = [None,'Lysine_g']
min_Methionine_g = [None,'Methionine_g']
max_Methionine_g = [None,'Methionine_g']
min_Phenylalanine_g = [None,'Phenylalanine_g']
max_Phenylalanine_g = [None,'Phenylalanine_g']
min_Threonine_g = [None,'Threonine_g']
max_Threonine_g = [None,'Threonine_g']
min_Tryptophan_g = [None,'Tryptophan_g']
max_Tryptophan_g = [None,'Tryptophan_g']


#matrix of restrictions
arr = [min_energija,max_energija,min_mascobe,max_mascobe,min_oghidrati,max_oghidrati,min_proteini,max_proteini,min_ca,max_ca,min_fe,max_fe,min_c,max_c,min_k,max_k,min_na,max_na,min_cena,max_cena,min_Teza_g,max_Teza_g,min_sladkor,max_sladkor,min_vlaknine,max_vlaknine,min_satfat,max_satfat,min_Histidine_g,max_Histidine_g, min_Isoleucine_g,max_Isoleucine_g,min_Leucine_g,max_Leucine_g,min_Lysine_g,max_Lysine_g,min_Methionine_g,max_Methionine_g,min_Phenylalanine_g,max_Phenylalanine_g,min_Threonine_g,max_Threonine_g ,min_Tryptophan_g,max_Tryptophan_g]

#bounds for each food
from scipy.optimize import Bounds

# Define bounds for each decision variable
bounds = Bounds([0]*len(food_dict), [np.inf]*len(food_dict))

integrality = np.ones(len(food_dict), dtype=int)


def linear_program(arr,minimum,bounds):
    def constraint(arr):
        A = []
        lb = []
        ub = []
        for i in range(len(arr)):
            if arr[i][0] is not None:
                nutrient_values = np.array(food_dict[arr[i][1]].tolist())
                if i % 2 == 0:  # Min constraint
                    A.append(nutrient_values)
                    lb.append(arr[i][0])
                    ub.append(np.inf)
                else:  # Max constraint
                    A.append(nutrient_values)
                    lb.append(-np.inf)
                    ub.append(arr[i][0])
        A = np.array(A)
        lb = np.array(lb)
        ub = np.array(ub)
        return A, lb, ub

    A, lb, ub = constraint(arr)
    c =-np.array( food_dict[minimum].tolist())
    result = milp(c=c, integrality=integrality, constraints= LinearConstraint(A,lb,ub), bounds=bounds)

    if result.success:
        return result.x, result.fun

problem_1, final_1 = linear_program(arr,minimum,bounds)

min_c[0] = 60.
min_k[0] = 3500.
min_na[0] = 500.
max_na[0] = 2400.

arr_2 = [min_energija,max_energija,min_mascobe,max_mascobe,min_oghidrati,max_oghidrati,min_proteini,max_proteini,min_ca,max_ca,min_fe,max_fe,min_c,max_c,min_k,max_k,min_na,max_na,min_cena,max_cena,min_Teza_g,max_Teza_g,min_sladkor,max_sladkor,min_vlaknine,max_vlaknine,min_satfat,max_satfat,min_Histidine_g,max_Histidine_g, min_Isoleucine_g,max_Isoleucine_g,min_Leucine_g,max_Leucine_g,min_Lysine_g,max_Lysine_g,min_Methionine_g,max_Methionine_g,min_Phenylalanine_g,max_Phenylalanine_g,min_Threonine_g,max_Threonine_g ,min_Tryptophan_g,max_Tryptophan_g]

problem_2, final_2 = linear_program(arr_2,minimum,bounds)

min_sladkor = [None,'sladkor_g']
max_sladkor = [30,'sladkor_g']
min_vlaknine = [30,'vlaknine_g']
max_vlaknine = [None,'vlaknine_g']
min_satfat = [None,'nasicene_mascobe_g']
max_satfat = [100,'nasicene_mascobe_g']
min_Histidine_g = [masa*0.01,'Histidine_g']
max_Histidine_g = [None,'Histidine_g']
min_Isoleucine_g = [masa*0.02,'Isoleucine_g']
max_Isoleucine_g = [None,'Isoleucine_g']
min_Leucine_g = [0.039*masa,'Leucine_g']
max_Leucine_g = [None,'Leucine_g']
min_Lysine_g = [0.03*masa,'Lysine_g']
max_Lysine_g = [None,'Lysine_g']
min_Methionine_g = [0.015*masa,'Methionine_g']
max_Methionine_g = [None,'Methionine_g']
min_Phenylalanine_g = [0.025*masa,'Phenylalanine_g']
max_Phenylalanine_g = [None,'Phenylalanine_g']
min_Threonine_g = [None,'Threonine_g']
max_Threonine_g = [None,'Threonine_g']
min_Tryptophan_g = [0.004*masa,'Tryptophan_g']
max_Tryptophan_g = [None,'Tryptophan_g']

arr_3 = [min_energija,max_energija,min_mascobe,max_mascobe,min_oghidrati,max_oghidrati,min_proteini,max_proteini,min_ca,max_ca,min_fe,max_fe,min_c,max_c,min_k,max_k,min_na,max_na,min_cena,max_cena,min_Teza_g,max_Teza_g,min_sladkor,max_sladkor,min_vlaknine,max_vlaknine,min_satfat,max_satfat,min_Histidine_g,max_Histidine_g, min_Isoleucine_g,max_Isoleucine_g,min_Leucine_g,max_Leucine_g,min_Lysine_g,max_Lysine_g,min_Methionine_g,max_Methionine_g,min_Phenylalanine_g,max_Phenylalanine_g,min_Threonine_g,max_Threonine_g ,min_Tryptophan_g,max_Tryptophan_g]


problem_3, final_3 = linear_program(arr_3,minimum,bounds)

index = np.logical_or(problem_1 > 0.0, problem_2 > 0.0)
index = np.logical_or(index, problem_3 > 0.0)

num_entries = np.sum(index)
indices = np.arange(num_entries)

bar_width = 0.25  

fig, ax = plt.subplots(figsize=(16,9))

bar1 = ax.barh(indices - bar_width, food_dict['quantity_g'].iloc[index].values * problem_1[index], bar_width,
              label=f'Pogoj 1: {minimum} ({final_1:.1f})',
              edgecolor='grey', linewidth=0.7)
ax.bar_label(bar1, padding=5, fmt='%.0f', fontsize=8)

bar2 = ax.barh(indices, food_dict['quantity_g'].iloc[index].values * problem_2[index], bar_width,
              label=f'Pogoj 2: {minimum} ({final_2:.1f})',
              edgecolor='grey', linewidth=0.7)
ax.bar_label(bar2, padding=5, fmt='%.0f', fontsize=8)

bar3 = ax.barh(indices + bar_width, food_dict['quantity_g'].iloc[index].values * problem_3[index], bar_width,
              label=f'Pogoj 3: {minimum} ({final_3:.1f})',
              edgecolor='grey', linewidth=0.7)
ax.bar_label(bar3, padding=5, fmt='%.0f', fontsize=8)

ax.set_title('Maksimizacija proteinov ob danih pogojih', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Količina hrane [g]', fontsize=14, labelpad=10)
ax.set_ylabel('Živilo', fontsize=14, labelpad=10)

ax.set_yticks(indices)
ax.set_yticklabels(food_dict['zivilo'][index], rotation=0, ha='right', fontsize=12)

ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

ax.legend(title='Pogoji', fontsize=12, title_fontsize=13, loc='center right')

plt.tight_layout()
plt.savefig('maksimizacija_protein_whole.pdf', format='pdf', dpi=300)

plt.show()

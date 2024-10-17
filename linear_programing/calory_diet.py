import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


#data 
food_dict = pd.read_csv('zivila_2.csv')
#Definicija slovarja    
#Ime zivila:	energija[kcal]	mascobe[g]	ogljikovi hidrati[g]	proteini[g]	Ca[mg]	Fe[mg]	Vitamin C[mg]	Kalij[mg] Natrij[mg]	Cena[EUR]
#vrednosti za 100g vsakega zivila

#Linearni problem
# max c^t x
# Ax < b
#x >= 0

sns.set(style="whitegrid")
set2_palette = sns.color_palette("Set2")[2:]
sns.set_palette(set2_palette)





food_dict['Teza_g'] = [100.]*len(food_dict)
#1)
'''Minimiziraj kolicino kalorij, ce je priporocen minimalni dnevni vnos 70 g mascob, 310 g ogljikovih
hidratov, 50 g proteinov, 1000 mg kalcija ter 18 mg zeleza. Dnevni obroki naj kolicinsko ne
presezejo dveh kilogramov hrane. Upostevate lahko se minimalne vnose za vitamin C (60 mg),
kalij (3500 mg) in sprejemljiv interval za natrij (500 mg – 2400 mg), ki so tudi na voljo v tabeli.'''
#masa
masa = 70

#Optimiziramo
minimum = 'Cena_EUR'

#Omejitve
min_energija = [2000,'energija_kcal']
max_energija = [None,'energija_kcal']
min_mascobe = [70,'mascobe_g']
max_mascobe = [None,'mascobe_g']
min_oghidrati = [310.,'ogljikovi_hidrati_g']
max_oghidrati = [None,'ogljikovi_hidrati_g']
min_proteini = [50.,'proteini_g']
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
max_cena = [None,'Cena_EUR']
min_Teza_g = [None,'Teza_g']
max_Teza_g = [2000.,'Teza_g']
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
arr = [min_energija,max_energija,min_mascobe,max_mascobe,min_oghidrati,max_oghidrati,min_proteini,max_proteini,min_ca,max_ca,min_fe,max_fe,min_cena,max_cena,min_Teza_g,max_Teza_g]

#bounds for each food
bounds = [(0,None)]*len(food_dict)
#linear program
def linear_program(arr,minimum,bounds):
    def constraint(arr):
        b = np.array([(-1)**(i+1) * arr[i][0] for i in range(len(arr)) if arr[i][0] != None])
        A = np.array([(-1)**(i+1) * np.array(food_dict[arr[i][1]].tolist()) for i in range(len(arr)) if arr[i][0] != None])
        return A,b
    
    A, b = constraint(arr)
    c = np.array( food_dict[minimum].tolist())
    result = linprog(c, A_ub= A, b_ub= b, bounds=bounds, method='highs')

    if result.success:
        return result.x, result.fun, result.slack, A, b, result['ineqlin']['marginals']

problem_1, final_1, slack_1, A_1, b_1, dual_1 = linear_program(arr,minimum,bounds)

slack = A_1 @ problem_1
num_entries = len(slack)
indices = np.arange(num_entries)

bar_width = 0.35  

fig1, ax1 = plt.subplots(figsize=(16,9))

bar1 = ax1.barh(indices - bar_width/2, b_1, bar_width,
              label=f'Omejitve',
              edgecolor='grey', linewidth=0.7)
ax1.bar_label(bar1, padding=5, fmt='%.0f', fontsize=10)
bar2 = ax1.barh(indices + bar_width/2, slack, bar_width,
              label=f'Izbrane kolicine',
              edgecolor='grey', linewidth=0.7)
ax1.bar_label(bar2, padding=1, fmt='%.0f', fontsize=10)

ax1.set_title('Omejitve pri dieti 1', fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel('Kolicina', fontsize=14, labelpad=10)
ax1.set_yticks(indices)
ax1.set_yticklabels([i[1] for i in arr if i[0]!= None], rotation=0, ha='right', fontsize=10)
ax1.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
ax1.legend(title='Dosezene vrednosti', fontsize=12, title_fontsize=13, loc='upper left') 
plt.tight_layout()
#plt.savefig('minimizacija_mascob_pogoj1.pdf', format='pdf', dpi=300)


min_c[0] = 60.
min_k[0] = 3500.
min_na[0] = 500.
max_na[0] = 2400.

arr_2 = [min_energija,max_energija,min_mascobe,max_mascobe,min_oghidrati,max_oghidrati,min_proteini,max_proteini,min_ca,max_ca,min_fe,max_fe,min_c,max_c,min_k,max_k,min_na,max_na,min_cena,max_cena,min_Teza_g,max_Teza_g]

problem_2, final_2, slack_2,A_2,b_2,dual_2 = linear_program(arr_2,minimum,bounds)

min_sladkor = [None,'sladkor_g']
max_sladkor = [25.,'sladkor_g']
min_vlaknine = [30.,'vlaknine_g']
max_vlaknine = [80.,'vlaknine_g']
min_satfat = [None,'nasicene_mascobe_g']
max_satfat = [15.,'nasicene_mascobe_g']
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

problem_3, final_3, slack_3, A_3, b_3,dual_3 = linear_program(arr_3,minimum,bounds)


index = np.logical_or(problem_1 > 0.0, problem_2 > 0.0)
index = np.logical_or(index, problem_3 > 0.0)

num_entries = np.sum(index)
indices = np.arange(num_entries)

bar_width = 0.25  

fig, ax = plt.subplots(figsize=(16,9))

bar1 = ax.barh(indices - bar_width, 100 * problem_1[index], bar_width,
              label=f'Pogoj 1: {minimum} ({final_1:.1f})',
              edgecolor='grey', linewidth=0.7)
ax.bar_label(bar1, padding=5, fmt='%.0f', fontsize=8)

bar2 = ax.barh(indices, 100 * problem_2[index], bar_width,
              label=f'Pogoj 2: {minimum} ({final_2:.1f})',
              edgecolor='grey', linewidth=0.7)
ax.bar_label(bar2, padding=5, fmt='%.0f', fontsize=8)

bar3 = ax.barh(indices + bar_width, 100 * problem_3[index], bar_width,
              label=f'Pogoj 3: {minimum} ({final_3:.1f})',
              edgecolor='grey', linewidth=0.7)
ax.bar_label(bar3, padding=5, fmt='%.0f', fontsize=8)

ax.set_title('Minimizacija mascob ob danih pogojih', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Količina hrane [g]', fontsize=14, labelpad=10)
ax.set_ylabel('Živilo', fontsize=14, labelpad=10)

ax.set_yticks(indices)
ax.set_yticklabels(food_dict['zivilo'][index], rotation=0, ha='right', fontsize=12)

ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

ax.legend(title='Pogoji', fontsize=12, title_fontsize=13, loc='upper right')

plt.tight_layout()
#plt.savefig('minimizacija_mascob.pdf', format='pdf', dpi=300)
slack = A_2 @ problem_2
num_entries = len(slack)
indices = np.arange(num_entries)

bar_width = 0.35  

fig1, ax1 = plt.subplots(figsize=(16,9))

bar1 = ax1.barh(indices - bar_width/2, b_2, bar_width,
              label=f'Omejitve',
              edgecolor='grey', linewidth=0.7)
ax1.bar_label(bar1, padding=5, fmt='%.0f', fontsize=10)
bar2 = ax1.barh(indices + bar_width/2, slack, bar_width,
              label=f'Izbrane kolicine',
              edgecolor='grey', linewidth=0.7)

ax1.set_title('Omejitve pri dieti 2', fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel('Količina', fontsize=14, labelpad=10)
ax1.set_ylabel('vsebina', fontsize=14, labelpad=10)
ax1.set_yticks(indices)
ax1.set_yticklabels([i[1] for i in arr_2 if i[0]!= None], rotation=0, ha='right', fontsize=10)
ax1.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
ax1.legend(title='Pogoji', fontsize=12, title_fontsize=13, loc='upper left') 
plt.tight_layout()
#plt.savefig('minimizacija_mascob_pogoj2.pdf', format='pdf', dpi=300)

slack = A_3 @ problem_3
num_entries = len(slack)
indices = np.arange(num_entries)

bar_width = 0.35  

fig1, ax1 = plt.subplots(figsize=(16,9))

bar1 = ax1.barh(indices - bar_width/2, b_3, bar_width,
              label=f'Omejitve',
              edgecolor='grey', linewidth=0.7)
ax1.bar_label(bar1, padding=10, fmt='%.0f', fontsize=7.5)
bar2 = ax1.barh(indices + bar_width/2, slack, bar_width,
              label=f'Izbrane kolicine',
              edgecolor='grey', linewidth=0.7)

ax1.bar_label(bar2, padding=1, fmt='%.0f', fontsize=7.5)
ax1.set_title('Omejitve pri dieti 3', fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel('Količina', fontsize=14, labelpad=10)
ax1.set_ylabel('vsebina', fontsize=14, labelpad=10)
ax1.set_yticks(indices)
ax1.set_yticklabels([i[1] for i in arr_3 if i[0]!= None], rotation=0, ha='right', fontsize=12)
ax1.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
ax1.legend(title='Pogoji', fontsize=12, title_fontsize=13, loc='upper left') 
plt.tight_layout()
#plt.savefig('minimizacija_mascob_pogoj3.pdf', format='pdf', dpi=300)
# Function to generate LaTeX table for constraints and dual variables
all_latex_tables = ""

# Function to generate LaTeX table for constraints and dual variables
def generate_latex_table(arr, dual_vars, problem_number):
    constraint_list = []
    constraint_index = 0

    for i in range(len(arr)):
        if arr[i][0] != None:
            RHS_value = arr[i][0]
            variable_name = arr[i][1]

            # Determine constraint type
            if i % 2 == 0:
                constraint_type = '$\\geq$'
            else:
                constraint_type = '$\\leq$'

            # Get the dual variable value
            dual_value = dual_vars[constraint_index]

            # Store the constraint
            constraint_list.append([variable_name, constraint_type, RHS_value, dual_value*RHS_value*0.1])

            # Increment constraint_index
            constraint_index += 1

    # Generate LaTeX table
    latex_table = f"\\begin{{table}}[h!]\n\\centering\n\\begin{{tabular}}{{|l|c|c|c|}}\n\\hline\n"
    latex_table += "Constraint & Type & RHS Value & Dual Variable \\\\\n\\hline\n"

    for constraint in constraint_list:
        variable_name = constraint[0]
        constraint_type = constraint[1]
        RHS_value = constraint[2]
        dual_value = constraint[3]
        latex_table += f"{variable_name} & {constraint_type} & {RHS_value} & {dual_value:.4f} \\\\\n"

    latex_table += "\\hline\n\\end{tabular}\n"
    latex_table += f"\\caption{{Constraints and Dual Variables for Problem {problem_number}}}\n"
    latex_table += f"\\label{{tab:problem{problem_number}_constraints}}\n\\end{{table}}\n\n"

    return latex_table

# Generate LaTeX tables for each problem and append to all_latex_tables
latex_table_1 = generate_latex_table(arr, dual_1, 1)
latex_table_2 = generate_latex_table(arr_2, dual_2, 2)
latex_table_3 = generate_latex_table(arr_3, dual_3, 3)

all_latex_tables = latex_table_1 + latex_table_2 + latex_table_3

# Save all LaTeX tables to a single text file
filename = "latex_tables_all_problems.txt"
with open(filename, 'w') as file:
    file.write(all_latex_tables)

print(f"All LaTeX tables saved to {filename}")

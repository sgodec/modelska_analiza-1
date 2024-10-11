import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

#data 
with open('zivila.dat', 'r') as file:
    lines = [line.replace('\t\t','\t').strip().split('\t') for line in file if not line.startswith('#')]
    food_dict = {item[0]: list(map(float,item[1:])) for item in lines}
    file.close()
#Definicija slovarja    
#Ime zivila:	energija[kcal]	mascobe[g]	ogljikovi hidrati[g]	proteini[g]	Ca[mg]	Fe[mg]	Vitamin C[mg]	Kalij[mg] Natrij[mg]	Cena[EUR]
#vrednosti za 100g vsakega zivila

#Linearni problem
# max c^t x
# Ax < b
#x >= 0

print(food_dict)
#1)
'''Minimiziraj kolicino kalorij, ce je priporocen minimalni dnevni vnos 70 g mascob, 310 g ogljikovih
hidratov, 50 g proteinov, 1000 mg kalcija ter 18 mg zeleza. Dnevni obroki naj kolicinsko ne
presezejo dveh kilogramov hrane. Upostevate lahko se minimalne vnose za vitamin C (60 mg),
kalij (3500 mg) in sprejemljiv interval za natrij (500 mg – 2400 mg), ki so tudi na voljo v tabeli.'''

c = [food_dict[key][0] for key in food_dict.keys()]
b = np.array([0,70,310,50,1000,18,-20])
A = [[0] + food_dict[key][1:6] + [-1] for key in food_dict.keys()]
A = np.array(A).T
bounds = [(0,None) for i in c]
result = linprog(c, A_ub= -A, b_ub= -b, bounds=bounds, method='highs')

if result.success:
    print('Optimal solution:', result.x)
    print('Optimal value:', result.fun)
else:
    print('No solution found.')

non_zero_items = [(food, quantity) for food, quantity in zip(food_dict.keys(), result.x) if quantity > 0]

foods, quantities = zip(*non_zero_items)

plt.figure(figsize=(10, 6))
plt.barh(foods, quantities)
plt.xlabel('Quantity')
plt.ylabel('Food Item')
plt.title('Optimal Diet with Lowest Calories')
plt.tight_layout()
plt.show()

c = [food_dict[key][0] for key in food_dict.keys()]
b = np.array([0,70,310,50,1000,18,60,3500,500,-2400,-20])
A = [[0] + food_dict[key][1:9] + [-food_dict[key][8]] + [-1] for key in food_dict.keys()]
A = np.array(A).T
bounds = [(0,None) for i in c]
result = linprog(c, A_ub= -A, b_ub= -b, bounds=bounds, method='highs')

if result.success:
    print('Optimal solution:', result.x)
    print('Optimal value:', result.fun)
else:
    print('No solution found.')

non_zero_items = [(food, quantity) for food, quantity in zip(food_dict.keys(), result.x) if quantity > 0]

foods, quantities = zip(*non_zero_items)

plt.figure(figsize=(10, 6))
plt.barh(foods, quantities)
plt.xlabel('Quantity')
plt.ylabel('Food Item')
plt.title('Optimal Diet with Lowest Calories')
plt.tight_layout()
plt.show()

'''Kako se rezultat razlikuje, ce zahtevamo minimalno 2000 kcal in namesto energije minimiziramo vnos mascob?'''

c = [food_dict[key][1] for key in food_dict.keys()]
b = np.array([2000,0,310,50,1000,18,-20])
A = [[food_dict[key][0]] + [0] + food_dict[key][2:6] + [-1] for key in food_dict.keys()]
A = np.array(A).T
bounds = [(0,None) for i in c]
result = linprog(c, A_ub= -A, b_ub= -b, bounds=bounds, method='highs')

if result.success:
    print('Optimal solution:', result.x)
    print('Optimal value:', result.fun)
else:
    print('No solution found.')

non_zero_items = [(food, quantity) for food, quantity in zip(food_dict.keys(), result.x) if quantity > 0]

foods, quantities = zip(*non_zero_items)

plt.figure(figsize=(10, 6))
plt.barh(foods, quantities)
plt.xlabel('Quantity')
plt.ylabel('Food Item')
plt.title('Optimal Diet with Lowest Calories')
plt.tight_layout()
plt.show()

c = [food_dict[key][1] for key in food_dict.keys()]
b = np.array([2000,0,310,50,1000,18,60,3500,500,-2400,-20])
A = [[food_dict[key][0]] + [0] + food_dict[key][2:9] + [-food_dict[key][8]] + [-1] for key in food_dict.keys()]
A = np.array(A).T
bounds = [(0,None) for i in c]
result = linprog(c, A_ub= -A, b_ub= -b, bounds=bounds, method='highs')

if result.success:
    print('Optimal solution:', result.x)
    print('Optimal value:', result.fun)
else:
    print('No solution found.')

non_zero_items = [(food, quantity) for food, quantity in zip(food_dict.keys(), result.x) if quantity > 0]

foods, quantities = zip(*non_zero_items)

plt.figure(figsize=(10, 6))
plt.barh(foods, quantities)
plt.xlabel('Quantity')
plt.ylabel('Food Item')
plt.title('Optimal Diet with Lowest Calories')
plt.tight_layout()
plt.show()

#Namesto kalorij minimiziraj se ceno. Kako se varcevanje odraza na zdravi prehrani?

c = [food_dict[key][-1] for key in food_dict.keys()]
b = np.array([2000,60,310,50,1000,18,60,3500,500,-2400,-20,0])
A = [food_dict[key][:9] + [-food_dict[key][8]] + [-1] + [0] for key in food_dict.keys()]
A = np.array(A).T
bounds = [(0,None) for i in c]
result = linprog(c, A_ub= -A, b_ub= -b, bounds=bounds, method='highs',  options={'dual_feasibility_tolerance': 1e-9})

if result.success:
    print('Optimal solution:', result.x)
    print('Optimal value:', result.fun)
else:
    print('No solution found.')

non_zero_items = [(food, quantity) for food, quantity in zip(food_dict.keys(), result.x) if quantity > 0]

foods, quantities = zip(*non_zero_items)

plt.figure(figsize=(10, 6))
plt.barh(foods, quantities)
plt.xlabel('Quantity')
plt.ylabel('Food Item')
plt.title('Optimal Diet with Lowest Calories')
plt.tight_layout()
plt.show()

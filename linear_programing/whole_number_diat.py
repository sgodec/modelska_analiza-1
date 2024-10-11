import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import milp
from scipy.optimize import Bounds, LinearConstraint, milp


food_dict = pd.read_csv('grocery_data.csv')
columns_to_include = ['energija[kcal]','energija[kcal]','mascobe[g]', 'ogljikovi_hidrati[g]','proteini[g]','Ca[mg]','Fe[mg]','Vitamin_C[mg]','Kalij[mg]','Natrij[mg]','Natrij[mg]','quantity[g]','Cena[EUR]' ]
rules = [1,-1,1,1,1,1,1,1,1,1,-1,-1,0]

c = food_dict['Cena[EUR]'].tolist()
b = np.array([2000,-4000,60,310,50,1000,18,60,3500,500,-2400,-2000,0])
A = np.array([rules[index]*np.array(food_dict[col].tolist()) for index,col in enumerate(columns_to_include)])
# Define bounds (all variables must be non-negative)
bounds = Bounds(0, np.inf)

# Define integer constraints (all variables must be integer)
integrality = np.ones(len(c), dtype=bool)  # All variables should be integers

# Define the linear constraint for the MILP
linear_constraint = LinearConstraint(-A, -np.inf, -b)

# Solve the MILP problem
result = milp(c=c, integrality=integrality, constraints=[linear_constraint], bounds=bounds)

# Check the result and display
if result.success:
    print('Optimal solution:', result.x)
    print('Optimal value:', result.fun)
else:
    print('No solution found.')

print(food_dict[np.bool(result.x)])

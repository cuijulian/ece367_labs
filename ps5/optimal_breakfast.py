import numpy as np
from scipy.optimize import linprog

# Inequality equations, LHS
A_ineq = np.array([[-70, 70, -107, 107, 45], [-121, 121, -500, 500, 40], [-65, 65, 0, 0, 60]]).transpose()

# Inequality equations, RHS
B_ineq = np.array([[-2000, 2250, -5000, 10000, 1000]])

# Cost function
c = np.array([0.15, 0.25, 0.05])

# Limit of 10 servings per ingredient
corn_b = [0, 10]
milk_b = [0, 10]
bread_b = [0, 10]

# Solve the LP
sol = linprog(c, A_ub=A_ineq, b_ub=B_ineq, bounds=(corn_b, milk_b, bread_b), method='interior-point')
print("The optimal variable x* is: " + str(sol["x"]))
print("The optimum value p* is: " + str(sol["fun"]))

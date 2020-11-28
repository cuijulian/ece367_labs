import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# Create system of equations
A = np.array([[1, 1], [0, 1]])
b = np.array([[0.5], [1]])
y = np.array([[1], [0]])

a1 = np.linalg.matrix_power(A, 9).dot(b)
a2 = np.linalg.matrix_power(A, 8).dot(b)
a3 = np.linalg.matrix_power(A, 7).dot(b)
a4 = np.linalg.matrix_power(A, 6).dot(b)
a5 = np.linalg.matrix_power(A, 5).dot(b)
a6 = np.linalg.matrix_power(A, 4).dot(b)
a7 = np.linalg.matrix_power(A, 3).dot(b)
a8 = np.linalg.matrix_power(A, 2).dot(b)
a9 = A.dot(b)
a10 = b

A_hat = np.hstack((a1, a2, a3, a4, a5, a6, a7, a8, a9, a10))

# Solve for l1 norm
numRows = len(A_hat)
numCols = len(A_hat[0])

Aeq = np.hstack((A_hat, np.zeros((numRows, numCols))))
Beq = y

c = np.vstack((np.zeros((numCols, 1)), np.ones((numCols, 1))))
A_ub = np.vstack((np.hstack((np.identity(numCols), -1 * np.identity(numCols))), np.hstack((-1 * np.identity(numCols), -1 * np.identity(numCols)))))
b_ub = np.zeros((2 * numCols, 1))

# T must be greater than or equal to zero
X = [-1 * np.inf, np.inf]
T = [0, np.inf]

# Solve the equivalent LP
sol = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=Aeq, b_eq=Beq, bounds=(X, X, X, X, X, X, X, X, X, X, T, T, T, T, T, T, T, T, T, T))
print(sol)
p_optimal = sol['x']
p_optimal = p_optimal[:numCols]
p_optimal = np.insert(p_optimal, 0, 0) # Insert initial force

# Find the positions and velocities at each time period
x = np.zeros((11, 1))
x_dot = np.zeros((11, 1))

for n in range(1, 11):
    x_dot[n] = x_dot[n - 1] + p_optimal[n]
    x[n] = x[n - 1] + x_dot[n - 1] + 0.5 * p_optimal[n]

# Plot optimal f, x, and x_dot over time
fig, (f1, f2, f3) = plt.subplots(3, 1)
t = np.arange(11)

f1.step(t, p_optimal)
f1.set_title('Optimal Force Over Time')
f1.set_xlabel('t')
f1.set_ylabel('f(t)')
f1.set_xticks(np.arange(min(t), max(t)+1, 1))
f1.grid()

f2.plot(t, x)
f2.set_title('Position Over Time')
f2.set_xlabel('t')
f2.set_ylabel('x(t)')
f2.set_xticks(np.arange(min(t), max(t)+1, 1))
f2.grid()

f3.plot(t, x_dot)
f3.set_title('Velocity Over Time')
f3.set_xlabel('t')
f3.set_ylabel('x_dot(t)')
f3.set_xticks(np.arange(min(t), max(t)+1, 1))
f3.grid()

plt.tight_layout()

# Solve for l inf norm
Aeq = np.hstack((A_hat, np.zeros((numRows, 1))))
Beq = y

c = np.vstack((np.zeros((numCols, 1)), np.ones((1, 1))))
A_ub = np.vstack((np.hstack((np.identity(numCols), -1 * np.ones((numCols, 1)))), np.hstack((-1 * np.identity(numCols), -1 * np.ones((numCols, 1))))))
b_ub = np.zeros((2 * numCols, 1))

# Solve the equivalent LP
sol = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=Aeq, b_eq=Beq, bounds=(X, X, X, X, X, X, X, X, X, X, T))
print(sol)
p_optimal = sol['x']
p_optimal = p_optimal[:numCols]
p_optimal = np.insert(p_optimal, 0, 0) # Insert initial force

# Find the positions and velocities at each time period
x = np.zeros((11, 1))
x_dot = np.zeros((11, 1))

for n in range(1, 11):
    x_dot[n] = x_dot[n - 1] + p_optimal[n]
    x[n] = x[n - 1] + x_dot[n - 1] + 0.5 * p_optimal[n]

# Plot optimal f, x, and x_dot over time
fig, (f1, f2, f3) = plt.subplots(3, 1)
t = np.arange(11)

f1.step(t, p_optimal)
f1.set_title('Optimal Force Over Time')
f1.set_xlabel('t')
f1.set_ylabel('f(t)')
f1.set_xticks(np.arange(min(t), max(t)+1, 1))
f1.grid()

f2.plot(t, x)
f2.set_title('Position Over Time')
f2.set_xlabel('t')
f2.set_ylabel('x(t)')
f2.set_xticks(np.arange(min(t), max(t)+1, 1))
f2.grid()

f3.plot(t, x_dot)
f3.set_title('Velocity Over Time')
f3.set_xlabel('t')
f3.set_ylabel('x_dot(t)')
f3.set_xticks(np.arange(min(t), max(t)+1, 1))
f3.grid()

plt.tight_layout()
plt.show()

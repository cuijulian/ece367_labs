import numpy as np
from cvxopt import solvers, matrix
import matplotlib.pyplot as plt

# Given data
P = 2*matrix([[0.2, -0.2, -0.12, 0.02], [-0.2, 1.4, 0.02, 0.0], [-0.12, 0.02, 1.0, -0.4], [0.02, 0.0, -0.4, 0.2]])
q = matrix([0.0, 0.0, 0.0, 0.0])
G = matrix([[-1.1, -1.0, 0.0, 0.0, 0.0], [-1.35, 0.0, -1.0, 0.0, 0.0], [-1.25, 0.0, 0.0, -1.0, 0.0], [-1.05, 0.0, 0.0, 0.0, -1.0]])
A = matrix([1.0, 1.0, 1.0, 1.0], (1, 4))
b = matrix(1.0)

# Vary the minimum expected return between 1.05 and 1.35
r_min_range = np.arange(1.05, 1.35, 0.01)

# For each minimum expected return find the QP solution and variance
variance_range = []
IBM_allocation = []
google_allocation = []
apple_allocation = []
intel_allocation = []

for r_min in r_min_range:
    h = matrix([-1 * r_min, 0.0, 0.0, 0.0, 0.0])
    sol = solvers.qp(P, q, G, h, A, b)
    IBM_allocation.append(sol['x'][0])
    google_allocation.append(sol['x'][1])
    apple_allocation.append(sol['x'][2])
    intel_allocation.append(sol['x'][3])
    variance_range.append((sol['x'].T * matrix([[0.2, -0.2, -0.12, 0.02], [-0.2, 1.4, 0.02, 0.0], [-0.12, 0.02, 1.0, -0.4], [0.02, 0.0, -0.4, 0.2]]) * sol['x'])[0])

# Plot variance for each minimum expected return
fig1 = plt.figure()
plt.plot(r_min_range, variance_range)
fig1.suptitle('Expected Return to Variance')
plt.xlabel('Expected Return (r_min)')
plt.ylabel('Variance')
plt.grid()

# Plot portfolio composition
fig2 = plt.figure()
plt.plot(r_min_range, IBM_allocation, label="IBM")
plt.plot(r_min_range, google_allocation, label="Google")
plt.plot(r_min_range, apple_allocation, label="Apple")
plt.plot(r_min_range, intel_allocation, label="Intel")
fig2.suptitle('Portfolio Composition over Varying Risk/Returns')
plt.xlabel('Expected Return (r_min)')
plt.ylabel('Allocation (out of total of 1)')
plt.legend()
plt.grid()

plt.show()

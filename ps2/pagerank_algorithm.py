import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Import pagerank_adj.mat matrix (2571 x 2571)
mat = scipy.io.loadmat('pagerank_adj.mat')["J"]

x = np.ones((2571, 1))

# Sum of all columns in mat
column_sums = np.zeros(2571)
for col in range(2571):
    for row in range(2571):
        column_sums[col] += mat[row][col]

# Create matrix A
A = np.zeros((2571, 2571))
for row in range(2571):
    for col in range(2571):
        A[row][col] = mat[row][col] / column_sums[col]

# Verify columns of A sum to 1 using matrix multiplication
result = A.transpose().dot(x)  # Results in vector of ones

# Normalize x
x_norm = x / np.linalg.norm(x)

# Vector of log of error terms
log_error_PI = np.zeros(10)
log_error_SIPI = np.zeros(10)
log_error_rayleigh = np.zeros(10)

# Power iteration algorithm
for k in range(10):
    y = A.dot(x_norm)
    x_norm = y / np.linalg.norm(y)
    evalue = x_norm.transpose().dot(A.dot(x_norm))
    log_error_PI[k] = np.log10(np.linalg.norm(A.dot(x_norm) - x_norm))

# Shift-invert power iteration algorithm
x_norm = x / np.linalg.norm(x)
sigma = 0.99
for k in range(10):
    y = np.linalg.inv(A - sigma * np.identity(2571)).dot(x_norm)
    x_norm = y / np.linalg.norm(y)
    evalue = x_norm.transpose().dot(A.dot(x_norm))
    log_error_SIPI[k] = np.log10(np.linalg.norm(A.dot(x_norm) - x_norm))

# Rayleigh quotient iteration algorithm
x_norm = x / np.linalg.norm(x)
for k in range(10):
    # Index starts at 0
    if k > 1:
        sigma = (x_norm.transpose().dot(A).dot(x_norm)) / (x_norm.transpose().dot(x_norm))
    y = np.linalg.inv(A - sigma * np.identity(2571)).dot(x_norm)
    x_norm = y / np.linalg.norm(y)
    log_error_rayleigh[k] = np.log10(np.linalg.norm(A.dot(x_norm) - x_norm))

# Plot log of the error
plt.plot(range(10), log_error_PI, label="Power iteration error")
plt.plot(range(10), log_error_SIPI, label="Shift-invert power iteration error")
plt.plot(range(10), log_error_rayleigh, label="Rayleigh quotient iteration error")
plt.xlabel('k value')
plt.ylabel('log(e(k+1))')
plt.title('Log error value plot')
plt.legend()

# PageRank score is the approximated eigenvector x(k+1)
top_five_indices = sorted(range(len(x_norm)), key=lambda i: x_norm[i])[-5:]
bottom_five_indices = sorted(range(len(x_norm)), key=lambda i: x_norm[i])[:5]

for p in top_five_indices: print(p)
for p in bottom_five_indices: print(p)

plt.show()

import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Load yalefaces.mat data
M = scipy.io.loadmat('yalefaces.mat')["M"]

# Flatten each 2x2 matrix to a col vector and find the mean
M_flattened = np.zeros((1024, 2414))
sum = np.zeros((1024, 1))
for i in range(2414):
    M_flattened[:, i] = np.reshape(M[:, :, i], (1, 1024))
    # Sum x vectors to find mean
    for j in range(1024):
        sum[j] += M_flattened[j, i]

x_mean = sum / 2414

# Create X with all 2414 centred image vectors
X = np.zeros((1024, 2414))
for i in range(1024):
    for j in range(2414):
        X[i][j] = M_flattened[i][j] - x_mean[i]

C = X.dot(np.transpose(X))

# Compute eigenvector/eigenvalue pairs for C
eigenvalues, eigenvectors = np.linalg.eig(C)

# Sort the eigenvalues
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

# Plot log(eigenvalues) over j
j = range(1, 1025)
log_eigenvalues = np.log10(eigenvalues)
plt.plot(j, log_eigenvalues)
plt.title('log(eigenvalues)')
plt.xlabel('j')
plt.ylabel('log(eigenvalues)')

# Reshape eigenvectors to get eigenfaces
eigenfaces = np.zeros((32, 32, 1024))
for i in range(1024):
    eigenfaces[:, :, i] = np.reshape(eigenvectors[:, i], (32, 32))

# Plot eigenfaces of largest 10 eigenvalues
fig_highest_eigenvalues, (f1, f2, f3, f4, f5, f6, f7, f8, f9, f10) = plt.subplots(1, 10)
f1.imshow(eigenfaces[:,:,0])
f2.imshow(eigenfaces[:,:,1])
f3.imshow(eigenfaces[:,:,2])
f4.imshow(eigenfaces[:,:,3])
f5.imshow(eigenfaces[:,:,4])
f6.imshow(eigenfaces[:,:,5])
f7.imshow(eigenfaces[:,:,6])
f8.imshow(eigenfaces[:,:,7])
f9.imshow(eigenfaces[:,:,8])
f10.imshow(eigenfaces[:,:,9])

# Plot eigenfaces of smallest 10 eigenvalues
fig_lowest_eigenvalues, (f11, f12, f13, f14, f15, f16, f17, f18, f19, f20) = plt.subplots(1, 10)
f11.imshow(eigenfaces[:,:,1014])
f12.imshow(eigenfaces[:,:,1015])
f13.imshow(eigenfaces[:,:,1016])
f14.imshow(eigenfaces[:,:,1017])
f15.imshow(eigenfaces[:,:,1018])
f16.imshow(eigenfaces[:,:,1019])
f17.imshow(eigenfaces[:,:,1020])
f18.imshow(eigenfaces[:,:,1021])
f19.imshow(eigenfaces[:,:,1022])
f20.imshow(eigenfaces[:,:,1023])


# Function that returns projection of vector onto subspace of B(j)
def projection_onto_B(x_mean, j):
    alpha = np.zeros((j, 1))
    proj = np.zeros((1024, 1))
    for i in range(j):
        # Add projection of vector onto jth vector in subspace
        alpha[i, 0] = (np.inner(x_mean, eigenvectors[:, i])/np.inner(eigenvectors[:, i], eigenvectors[:, i]))
        eigenvector = eigenvectors[:, i]
        # Add projection component
        for k in range(1024):
            proj[k] = proj[k] + (alpha[i, 0] * eigenvector)[k]
    return alpha, proj


j_values = [2, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10]

# Project image J(1) onto B
y = np.zeros((1024, 10))
fig_y = plt.figure(4)
# Loop through j values
for i in range(10):
    # Project X[:, 0] onto subspace v(1) to v(j) to get y_mean(i, j)
    projection = projection_onto_B(X[:, 0], j_values[i])[1]
    for j in range(1024):
        y[j, i] = projection[j]
    # Add x_mean vector to y_mean(i,j) to get y(i,j)
    for j in range(1024):
        y[j, i] += x_mean[j]

    # Reshape y and plot
    y_plot = np.reshape(y[:, i], (32, 32))
    plot = fig_y.add_subplot(3, 10, i + 1)
    plot.imshow(y_plot)

# Project image J(1076) onto B
for i in range(10):
    # Project X[:, 1075] onto subspace v(1) to v(j) to get y_mean(i, j)
    projection = projection_onto_B(X[:, 1075], j_values[i])[1]
    for j in range(1024):
        y[j, i] = projection[j]
    # Add x_mean vector to y_mean(i,j) to get y(i,j)
    for j in range(1024):
        y[j, i] += x_mean[j]

    # Reshape y and plot
    y_plot = np.reshape(y[:, i], (32, 32))
    plot = fig_y.add_subplot(3, 10, i + 11)
    plot.imshow(y_plot)

# Project image J(2043) onto B
for i in range(10):
    # Project X[:, 2042] onto subspace v(1) to v(j) to get y_mean(i, j)
    projection = projection_onto_B(X[:, 2042], j_values[i])[1]
    for j in range(1024):
        y[j, i] = projection[j]
    # Add x_mean vector to y_mean(i,j) to get y(i,j)
    for j in range(1024):
        y[j, i] += x_mean[j]

    # Reshape y and plot
    y_plot = np.reshape(y[:, i], (32, 32))
    plot = fig_y.add_subplot(3, 10, i + 21)
    plot.imshow(y_plot)


# Project images in set I onto B(25)
image_set = [0, 1, 6, 2042, 2043, 2044]
coeffs = np.zeros((25, 6))
for i in range(6):
    alphas = projection_onto_B(X[:, image_set[i]], 25)[0]
    # Store alpha values into coeffs
    for j in range(25):
        coeffs[j, i] = alphas[j, 0]

# Tabulate Euclidean distances between the pairwise c(i) vectors
distances = np.zeros((6, 6))
for i in range(6):
    for j in range(i + 1, 6):
        x = coeffs[:, i]
        y = coeffs[:, j]
        distances[i, j] = np.linalg.norm(x - y)

plt.show()
print("Done!")

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

M = sio.loadmat('sparseCoding.mat')["M"]
H = sio.loadmat('sparseCoding.mat')["H"]
M_tilda = H.dot(M).dot(np.transpose(H))

bins = np.arange(0, 255, 25)
bins = bins[1:]
bins_transform = np.arange(0, 255, 0.2)
bins_transform = bins_transform[1:]

# Histogram of M
fig1 = plt.figure()
plt.hist(np.hstack(M), bins, ec='black')
fig1.suptitle('Histogram of Image M')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Histogram of M_tilda
fig2 = plt.figure()
plt.hist(np.hstack(M_tilda), bins_transform, ec='black')
fig2.suptitle('Histogram of Wavelet Transform M_tilda')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Count number of non-zero coefficients in M and M_tilda
non_zero_M = 0
non_zero_M_tilda = 0

for i in range(0, 256):
    for j in range(0, 256):
        if M[i][j] > 0:
            non_zero_M += 1
        if M_tilda[i][j] > 0:
            non_zero_M_tilda += 1

print("The number of non-zero coefficients in M is " + str(non_zero_M))
print("The number of non-zero coefficients in M_tilda is " + str(non_zero_M_tilda))

# Find optimal X*
lam = 30
X = np.zeros((256, 256))
for i in range(0, 256):
    for j in range(0, 256):
        m_tilda_value = M_tilda[i][j]
        if abs(m_tilda_value) <= lam:
            X[i][j] = 0
        else:
            X[i][j] = m_tilda_value - lam * np.sign(m_tilda_value)

# Find compression factor and MSE of X
non_zero_X = 0
for i in range(0, 256):
    for j in range(0, 256):
        if X[i][j] > 0:
            non_zero_X += 1
compression_factor = float(non_zero_X) / (256.0 ** 2)

M_approx = np.transpose(H).dot(X).dot(H)
error_squared_sum = 0
for i in range(0, 256):
    for j in range(0, 256):
        error_squared_sum += np.power(M[i][j] - M_approx[i][j], 2)
MSE = error_squared_sum / (256.0 ** 2)

# Display image M and its approximation M_approx
fig, (f1, f2) = plt.subplots(1, 2)
f1.set_title('Original Image')
f1.imshow(M[:, :])
f2.set_title('Image Approximation')
f2.imshow(M_approx[:, :])

# Histogram of X*
fig3 = plt.figure()
plt.hist(np.hstack(X), bins_transform, ec='black')
fig3.suptitle('Histogram of Image Approximation X*')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.show()

import math
import numpy as np
import scipy.io

# Import wordVecV.mat matrix
V = scipy.io.loadmat('wordVecV.mat')["V"]

# Get the “raw” term-by-document matrix
M = np.zeros((1651, 10))
for i in range(1651):
    for j in range(10):
        if V[i][j] > 0:
            M[i][j] = 1

# Get normalized M matrix
M_tilda = np.zeros((1651, 10))
norms_of_columns = np.zeros(10)

for j in range(10):
    squared_column_sum = 0
    for i in range(1651):
        squared_column_sum += pow(M[i][j], 2)
    norms_of_columns[j] = np.sqrt(squared_column_sum)

for i in range(1651):
    for j in range(10):
        M_tilda[i][j] = M[i][j]/norms_of_columns[j]

# Singular value decomposition of M_tilda
U, S, V_transpose = np.linalg.svd(M_tilda)


# Function that returns projection of vector onto subspace of u(k)
def projection_onto_u(vector, k):
    projection = np.zeros((1651, 1))
    for i in range(k):
        # Add projection of vector onto kth vector in subspace
        alpha = (np.inner(vector, U[:, i])/np.inner(U[:, i], U[:, i]))
        u_vector = U[:, i]
        # Add projection component
        for j in range(1651):
            projection[j] = projection[j] + (alpha * u_vector)[j]
    return projection

# Function that returns cosine similarity between two vectors
def cosine_similarity(d, q):
    # Term A is numerator, term B is denominator part, term C is other denominator part
    A = B = C = 0
    for i in range(1651):
        A += d[i] * q[i]
        B += d[i] * d[i]
        C += q[i] * q[i]
    similarity = A / (np.sqrt(B) * np.sqrt(C))
    return float(similarity)

# For every pair of d's(matrix columns of M_tilda) find the cosine similarity. Repeat for decreasing k
cosine_similarities = np.zeros((45, 9))

for k in range(9,0,-1):
    row = 0
    for i in range(10):
        for j in range(i + 1, 10):
            # Project document vectors onto subspace spanned by u(k)
            d_projection = projection_onto_u(M_tilda[:, i], k)
            q_projection = projection_onto_u(M_tilda[:, j], k)

            cosine_similarities[row][9 - k] = (cosine_similarity(d_projection, q_projection))
            row += 1

print("The most similar documents are the last two, Barack Obama and George W. Bush")
print("The lowest k can be without changing the answer is k = 3")

import math
import numpy as np
import scipy.io

# Import wordVecV.mat matrix
mat = scipy.io.loadmat('wordVecV.mat')["V"]

# Create w array
warray = np.empty((1651, 10))

for i in range(1651):
    # Calculate number of articles this word appears in
    document_frequency = 0
    for d in range(10):
        if mat[i, d] > 0:
            document_frequency += 1
    for j in range(10):
        # Calculate how many words are in this article
        num_words_in_article = sum(mat[:, j])
        # warray[i][j] = equation value
        warray[i][j] = (mat[i, j] / num_words_in_article) * math.sqrt(math.log(10.0 / document_frequency, 10))


distances = []
distances_normalized = []
distances_TFIDF = []
angles = []
angles_normalized = []

# For every pair of v's(i.e. matrix columns) find the Euclidean distance and angle
for i in range(10):
    for j in range(i + 1, 10):
        vector_1 = mat[:, i]
        vector_2 = mat[:, j]
        distances.append(np.linalg.norm(vector_1 - vector_2))
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angles.append(np.arccos(dot_product))

        # TF-IDF vectors
        vector_1 = warray[:, i]
        vector_2 = warray[:, j]
        distances_TFIDF.append(np.linalg.norm(vector_1 - vector_2))

        # Normalized vectors
        vector_1_normalized = mat[:, i] / sum(mat[:, i])
        vector_2_normalized = mat[:, j] / sum(mat[:, j])
        distances_normalized.append(np.linalg.norm(vector_1_normalized - vector_2_normalized))
        unit_vector_1 = vector_1_normalized / np.linalg.norm(vector_1_normalized)
        unit_vector_2 = vector_2_normalized / np.linalg.norm(vector_2_normalized)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angles_normalized.append(np.arccos(dot_product))

minimum_distance_index = distances.index(min(distances))
minimum_angle_index = angles.index(min(angles))
minimum_normalized_distance_index = distances_normalized.index(min(distances_normalized))
minimum_normalized_angle_index = angles_normalized.index(min(angles_normalized))
minimum_distance_TFIDF_index = distances_TFIDF.index(min(distances_TFIDF))

print(minimum_distance_index) # Value of 39 -> corresponds to v7 and v8
print(minimum_angle_index) # Value of 44 -> corresponds to v9 and v10
print(minimum_normalized_distance_index) # Value of 44 -> corresponds to v9 and v10
print(minimum_normalized_angle_index) # Value of 44 -> corresponds to v9 and v10
print(minimum_distance_TFIDF_index) # Value of 44 -> corresponds to v9 and v10

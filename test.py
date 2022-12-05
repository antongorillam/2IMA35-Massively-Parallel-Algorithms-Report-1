from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import normalize

import numpy as np
import matplotlib.pyplot as plt
import math
IMG_DIM = 28
N_SAMPLES = 100


(train_X, train_y), _ = mnist.load_data()
train_X, train_y = train_X[:N_SAMPLES], train_y[:N_SAMPLES]

# Visualisation
# plt.imshow(train_X[2].reshape(IMG_DIM, IMG_DIM), cmap=plt.get_cmap('gray'))
# plt.show()
# DIV = train_X.max()
# print(f"DIV: {DIV}")

# Distance matrix
distances = np.zeros((N_SAMPLES,N_SAMPLES))
for i, img_i in enumerate(train_X):
    for j, img_j in enumerate(train_X):
        distances[i, j] = np.sqrt((img_i - img_j)**2).sum() 

# normalized_distances = normalize(distances, axis=0, norm="l2")


distances = distances / distances.max() 
print(f"distances: \n{distances}")
print(f"distances: {distances[4,1]}")
print(f"distances: {distances[1,4]}")
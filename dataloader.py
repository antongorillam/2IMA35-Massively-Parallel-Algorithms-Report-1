from email.mime import image
from tkinter.messagebox import RETRY
from tensorflow.keras.datasets import mnist, fashion_mnist
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
import numpy as np
import matplotlib.pyplot as plt

from plotter import Plotter
import affinityclustering as ac

IMG_DIM = 28
N_SAMPLES = 2500
NUMPY_ARRAY_FOLDER = "numpy_arrays/" + str(N_SAMPLES) + "_samples" 
IMAGE_FOLDER = "images/" + str(N_SAMPLES) + "_samples"

class Dataloader():

    def computeDistanceMatrix(self, show=False): 
        (train_X, train_y), _ = fashion_mnist.load_data()
        train_X, train_y = train_X[:N_SAMPLES], train_y[:N_SAMPLES]

        if show:
            # Visualisation
            image_idx = np.random.randint(len(train_X))
            plt.imshow(train_X[image_idx].reshape(IMG_DIM, IMG_DIM), cmap=plt.get_cmap('gray'))
            plt.show()

        distances = np.zeros((N_SAMPLES,N_SAMPLES))
        for i, img_i in enumerate(train_X):
            for j, img_j in enumerate(train_X):
                distances[i, j] = np.sqrt((img_i - img_j)**2).sum() 
        
        distances = distances
        np.save(f"{NUMPY_ARRAY_FOLDER}/distances.npy", distances)
        np.save(f"{NUMPY_ARRAY_FOLDER}/labels.npy", train_y)

    def computeCoordiatesAndNewDistances(self, reduction_model="TSNE"):
        distances = np.load(f"{NUMPY_ARRAY_FOLDER}/distances.npy")

        if reduction_model == "MDS":
            model = MDS(n_components=2, metric=False)
        elif reduction_model == "TSNE":
            model = TSNE(n_components=2, perplexity=15, random_state=42)
        else:
            model = PCA(n_components=2)

        coordinates = model.fit_transform(distances)
        new_distances = np.zeros((N_SAMPLES,N_SAMPLES))
        for i, first in enumerate(coordinates):
            for j, second in enumerate(coordinates):
                new_distances[i, j] = np.sqrt((first[0] - second[0])**2 + (first[1] - second[1])**2)

        np.save(f"{NUMPY_ARRAY_FOLDER}/coordinates.npy", coordinates)
        np.save(f"{NUMPY_ARRAY_FOLDER}/new_distances.npy", new_distances)

    # Construct Graph
    def constructGraph(self, distances):
        vertex_dict = {}
        for i in range(len(distances)):
            neighbor_dict = {}
            for j in range(len(distances)):
                if i != j:
                    neighbor_dict[j] = distances[i,j]
            vertex_dict[i] = neighbor_dict
        
        return vertex_dict

    def loadData(self):
        return {
            "distances" : np.load(f"{NUMPY_ARRAY_FOLDER}/distances.npy"),
            "coordinates" : np.load(f"{NUMPY_ARRAY_FOLDER}/coordinates.npy"),
            "labels" : np.load(f"{NUMPY_ARRAY_FOLDER}/labels.npy"),
            "new_distances" : np.load(f"{NUMPY_ARRAY_FOLDER}/new_distances.npy"),        
        }
        
def generateData():
    """
    Run file to generate data
    """
    dl = Dataloader()
    dl.computeDistanceMatrix()
    dl.computeCoordiatesAndNewDistances()
    data = dl.loadData()
    new_distances, distances, coordinates, labels = data["new_distances"], data["distances"], data["coordinates"], data["labels"]
    plotter = Plotter(vertex_coordinates=coordinates, name_dataset="MNIST", file_loc=IMAGE_FOLDER)
    plotter.plot_vertex_coordinates(coordinates, labels, "ground_truth")

if __name__ == '__main__':
    # N_SAMPLES_LIST = [100, 250, 500, 1000, 2500]
    # for n_samples in N_SAMPLES_LIST:
    #     N_SAMPLES = n_samples
    #     NUMPY_ARRAY_FOLDER = "numpy_arrays/" + str(N_SAMPLES) + "_samples" 
    #     IMAGE_FOLDER = "images/" + str(N_SAMPLES) + "_samples"
    #     generateData()
    generateData()

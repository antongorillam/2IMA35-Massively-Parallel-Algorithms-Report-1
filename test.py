from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS

from plotter import Plotter
import affinityclustering as ac
import numpy as np
import matplotlib.pyplot as plt
import math

IMG_DIM = 28
N_SAMPLES = 200

class Dataloader():

    def computeDistanceMatrix(self): 
        (train_X, train_y), _ = mnist.load_data()
        train_X, train_y = train_X[:N_SAMPLES], train_y[:N_SAMPLES]

        # Visualisation
        # plt.imshow(train_X[2].reshape(IMG_DIM, IMG_DIM), cmap=plt.get_cmap('gray'))
        # plt.show()

        distances = np.zeros((N_SAMPLES,N_SAMPLES))
        for i, img_i in enumerate(train_X):
            curr_dict = {}
            for j, img_j in enumerate(train_X):
                distances[i, j] = np.sqrt((img_i - img_j)**2).sum() 
        
        normalized_distances = distances
        # print(f"normalized_distances: \n{normalized_distances}")
        # print(f"normalized_distances: {normalized_distances[4,1]}")
        # print(f"normalized_distances: {normalized_distances[1,4]}")
        np.save(f"normalized_distances.npy", normalized_distances)
        np.save(f"labels.npy", train_y)

    def getDistanceMatrix(self, reduction_model="TSE"):
        distances = np.load("normalized_distances.npy")
        if reduction_model == "MDS":
            model = MDS(n_components=2, metric=False)
        elif reduction_model == "TSE":
            model = TSNE(n_components=2, perplexity=13)
        else:
            model = PCA(n_components=2)

        coordinates = model.fit_transform(distances)
        return distances, coordinates

    # Construct Graph
    def constructGraph(self):
        distances, coordinates = self.getDistanceMatrix()
        print(f"distances.shape: {distances.shape}")

        vertex_dict = {}
        for i in range(len(distances)):
            neighbor_dict = {}
            for j in range(len(distances)):
                if i != j:
                    neighbor_dict[j] = distances[i,j]
            vertex_dict[i] = neighbor_dict

        return vertex_dict, coordinates

def main():
    """
    coordinates ~ (SAMPLE_SIZE, 2)
    labels ~ (SAMPLE_SIZE,)
    """
    dl = Dataloader()
    dl.computeDistanceMatrix()
    vertex_dict, coordinates = dl.constructGraph()
    # adjencency_list = ac.get_nearest_neighbours(
    #     V=vertex_dict, 
    #     k=N_SAMPLES-1, 
    #     leaf_size=2, 
    #     buckets=True)

    labels = np.load("labels.npy")
    plotter = Plotter(vertex_coordinates=coordinates, name_dataset="MNIST", file_loc="images/")
    # runs, graph, yhats, contracted_leader, mst = \
    #     ac.affinity_clustering(adjencency_list, num_clusters=10)
    plotter.plot_vertex_coordinates(coordinates, labels)
    # plotter.plot_cluster(yhats[runs - 1], mst, coordinates, labels)
    # plotter.plot_mst_2d(mst, intermediate=True, plot_cluster=True, num_clusters=10)
    # print(f"runs: {runs}")
    # print(f"graph: {np.array(graph).shape}")
    # print(f"yhats: {np.array(yhats).shape}")
    # print(f"contracted_leader: {np.array(contracted_leader)}")
    # print('Graph size: ', len(graph), graph)

if __name__ == '__main__':
    main()  
    # print(f"vertex_dict[0]: {vertex_dict[0]}")
    # min_key = min(vertex_dict[0], key=vertex_dict[0].get)    

    # print(f"min_key: {min_key}")
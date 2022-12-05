from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import normalize

from plotter import Plotter
import affinityclustering as ac
import numpy as np
import matplotlib.pyplot as plt
import math
IMG_DIM = 28
N_SAMPLES = 20

class Dataloader():

    def getDistanceMatrix(self): 
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

        normalized_distances = distances / distances.max()
        print(f"normalized_distances: \n{normalized_distances}")
        print(f"normalized_distances: {normalized_distances[4,1]}")
        print(f"normalized_distances: {normalized_distances[1,4]}")
        np.save(f"normalized_distances.npy", normalized_distances)

    # Construct Graph
    def constructGraph(self):
        distances = np.load("normalized_distances.npy")
        print(f"distances.shape: {distances.shape}")

        vertex_dict = {}
        for i in range(len(distances)):
            neighbor_dict = {}
            for j in range(len(distances)):
                if i != j:
                    neighbor_dict[j] = distances[i,j]
            vertex_dict[i] = neighbor_dict

        return vertex_dict

def main():
    dl = Dataloader()
    # dl.getDistanceMatrix()
    vertex_dict = dl.constructGraph()
    adjencency_list = ac.get_nearest_neighbours(
        V=vertex_dict, 
        k=5, 
        leaf_size=2, 
        buckets=True)
    print(vertex_dict[0])    
    print(adjencency_list[0])

    cnt = 0
    diff = []
   
    # plotter.set_dataset(names_datasets[cnt])
    # plotter.update_string()
    # plotter.reset_round()
    runs_list, graph_list, yhats_list, contracted_leader_list, msts = [], [], [], [], []
    for i in range(10):
        runs, graph, yhats, contracted_leader, mst = ac.affinity_clustering(adjencency_list)
        runs_list.append(runs)
        graph_list.append(graph)
        yhats_list.append(yhats),
        contracted_leader_list.append(contracted_leader)
        msts.append(mst)
        print('Graph size: ', len(graph), graph)
        print('Runs: ', runs)
    diff.append(
        ac.find_differences(msts))
    cnt += 1


if __name__ == '__main__':
    main()
    # print(f"vertex_dict[0]: {vertex_dict[0]}")
    # min_key = min(vertex_dict[0], key=vertex_dict[0].get)    
    # print(f"min_key: {min_key}")
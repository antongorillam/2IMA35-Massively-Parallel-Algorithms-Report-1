from dataloader import Dataloader
from plotter import Plotter
import affinityclustering as ac
import numpy as np

IMG_DIM = 28
N_SAMPLES = 1000
NUMPY_ARRAY_FOLDER = "numpy_arrays/"

def main():
    """
    coordinates ~ (SAMPLE_SIZE, 2)
    labels ~ (SAMPLE_SIZE,)
    """
    dl = Dataloader()
    # dl.computeDistanceMatrix()
    # dl.getDistanceMatrix()
    data = dl.loadData()
    new_distances, distances, coordinates, labels = data["new_distances"], data["distances"], data["coordinates"], data["labels"]
    vertex_dict = dl.constructGraph(new_distances)
    print(vertex_dict) 
    # print(distances)

    adjencency_list = ac.get_nearest_neighbours(
        V=vertex_dict, 
        k=5, 
        leaf_size=2, 
        buckets=True)

    plotter = Plotter(vertex_coordinates=coordinates, name_dataset="MNIST", file_loc="images/")
    # runs, graph, yhats, contracted_leader, mst = \
    #     ac.affinity_clustering(adjencency_list, num_clusters=10)
    # plotter.plot_vertex_coordinates(coordinates, labels)
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
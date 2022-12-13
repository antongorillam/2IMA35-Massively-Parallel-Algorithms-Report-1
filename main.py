from dataloader import Dataloader
from plotter import Plotter
from pathlib import Path
import os
import affinityclustering as ac
import numpy as np

IMG_DIM = 28
N_SAMPLES = 250
NUMPY_ARRAY_FOLDER = "numpy_arrays/" + f"{N_SAMPLES}_samples"

def main():
    """
    coordinates ~ (SAMPLE_SIZE, 2)
    labels ~ (SAMPLE_SIZE,)
    """
    dl = Dataloader()
    data = dl.loadData(numpy_array_folder=NUMPY_ARRAY_FOLDER)
    new_distances, distances, coordinates, labels = data["new_distances"], data["distances"], data["coordinates"], data["labels"]
    vertex_dict = dl.constructGraph(distances)
    
    # adjencency_list = ac.get_nearest_neighbours(
    #     V=vertex_dict, 
    #     k=10,
    #     leaf_size=2, 
    #     buckets=True)
    
    # plotter = Plotter(vertex_coordinates=coordinates, name_dataset="MNIST", file_loc="images/")
    # plotter.plot_vertex_coordinates(coordinates, labels, "ground_truth")
    # runs, graph, yhats, contracted_leader, mst = \
    #     ac.affinity_clustering(adjencency_list, num_clusters=10)

    # print(f"runs: {runs}")
    # plotter.plot_vertex_coordinates(coordinates, np.array(contracted_leader), f"run_testlol_clustering")
    

if __name__ == '__main__':
    main()  
    # print(f"vertex_dict[0]: {vertex_dict[0]}")
    # min_key = min(vertex_dict[0], key=vertex_dict[0].get)    

    # print(f"min_key: {min_key}")
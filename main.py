from dataloader import Dataloader
from plotter import Plotter
from pathlib import Path
import os
import affinityclustering as ac
import numpy as np
import time

IMG_DIM = 28
N_SAMPLES = 250
dataset = "fashion_mnist"
NUMPY_ARRAY_FOLDER = "numpy_arrays/" + dataset + "/" + str(N_SAMPLES) + "_samples"
IMAGE_FOLDER = "images/" + dataset + "/" + str(N_SAMPLES) + "_samples"

def main():
    """
    coordinates ~ (SAMPLE_SIZE, 2)
    labels ~ (SAMPLE_SIZE,)
    """
    dl = Dataloader()
    data = dl.loadData(numpy_array_folder=NUMPY_ARRAY_FOLDER)
    new_distances, distances, coordinates, labels = data["new_distances"], data["distances"], data["coordinates"], data["labels"]
    #vertex_dict = dl.constructGraph(distances)
    time_start = time.perf_counter()
    vertex_dict = dl.constructGraph(new_distances)
    
    adjencency_list = ac.get_nearest_neighbours(
        V=vertex_dict,
        k=10,
        leaf_size=2,
        buckets=True)
    
    plotter = Plotter(vertex_coordinates=coordinates, name_dataset=dataset, file_loc=IMAGE_FOLDER)
    runs, graph, yhats, contracted_leader, mst = \
        ac.affinity_clustering(adjencency_list, num_clusters=10)

    time_finish = time.perf_counter()
    duration = time_finish - time_start
    print(f"Runs: {runs}, Time: {duration:.2f}")
    plotter.plot_vertex_coordinates(coordinates, np.array(contracted_leader), f"run_testlol_clustering", show_legend=False)

    

if __name__ == '__main__':
    main()  
    # print(f"vertex_dict[0]: {vertex_dict[0]}")
    # min_key = min(vertex_dict[0], key=vertex_dict[0].get)    

    # print(f"min_key: {min_key}")
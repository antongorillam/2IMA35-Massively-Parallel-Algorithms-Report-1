from dataloader import Dataloader
from plotter import Plotter
from pathlib import Path
import os
import affinityclustering as ac
import numpy as np
import time

IMG_DIM = 28
N_SAMPLES = 100
DATASET = "fashion_mnist"
NUMPY_ARRAY_FOLDER = "numpy_arrays/" + DATASET + "/" + str(N_SAMPLES) + "_samples"
IMAGE_FOLDER = "images/" + DATASET + "/" + str(N_SAMPLES) + "_samples"

def main(k_neighbours=10):
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
        k=k_neighbours,
        leaf_size=2,
        buckets=True)
    
    plotter = Plotter(vertex_coordinates=coordinates, name_dataset=DATASET, file_loc=IMAGE_FOLDER)
    plotter.plot_vertex_coordinates(coordinates, labels, "ground_truth")
    runs, graph, yhats, contracted_leader, mst = \
        ac.affinity_clustering(adjencency_list, num_clusters=10)

    time_finish = time.perf_counter()
    duration = time_finish - time_start
    print(f"Runs: {runs}, Time: {duration:.2f}s")
    plotter.plot_vertex_coordinates(coordinates, np.array(yhats[-1]), f"postTSNE_{N_SAMPLES}samples_k_{str(k_neighbours).zfill(2)}", show_legend=False)

if __name__ == '__main__':
    
    N_SAMPLES_LIST = [100, 250, 500, 1000, 2500]
    DATASET_LIST = ["fashion_mnist", "mnist"]

    for dataset in DATASET_LIST:
        for n_samples in N_SAMPLES_LIST:
            for k in range(1, 11):
                N_SAMPLES = n_samples 
                DATASET = dataset
                NUMPY_ARRAY_FOLDER = "numpy_arrays/" + DATASET + "/" + str(N_SAMPLES) + "_samples"
                IMAGE_FOLDER = "images/" + DATASET + "/" + str(N_SAMPLES) + "_samples"
                main(k_neighbours=k)
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from plotter import Plotter
import affinityclustering as ac
import numpy as np
import matplotlib.pyplot as plt
import math
from test import Dataloader

IMG_DIM = 28
N_SAMPLES = 200



def main():
    """
    coordinates ~ (SAMPLE_SIZE, 2)
    labels ~ (SAMPLE_SIZE,)
    """
    dl = Dataloader()
    vertex_dict, coordinates = dl.constructGraph()

    labels = np.load("labels.npy")
    plotter = Plotter(vertex_coordinates=coordinates, name_dataset="MNIST", file_loc="")
    plotter.plot_vertex_coordinates(coordinates, labels)


if __name__ == '__main__':
    main()  

"""

Plot graphs on original stitched image
"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from scipy.spatial import KDTree as KDTree

from features import calculate_KNN


def plot_graph(img_file_path:str,
               hd_name:str,
               num_to_plot:int):
    
    """
    Plot graph on raw image

    Parameters:
    img_file_path:str, folder name containing raw and labeled stitched images
    hd_name:str, name of hdf5 file containing graph data
    num_to_plot:int, number of raw images to plot on

    Returns: None
    
    """
     

    with h5py.File(hd_name, 'r') as g:
        img_names = list(set(g.attrs['roi_name_label']))[:num_to_plot]
        img_names_raw = list(set(g.attrs['roi_name_raw']))[:num_to_plot]

    for i in range(num_to_plot):
        stitched_img = np.load(os.path.join(img_file_path, img_names[i]))
        raw_img = np.load(os.path.join(img_file_path, img_names_raw[i]))

        props = regionprops(stitched_img)
        cent = np.zeros((len(props), 2))
        for i, p in enumerate(props):
            cent[i] = p.centroid[0]*4, p.centroid[1]*4

        indices = calculate_KNN(cent)

        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        ax[0].imshow(raw_img, cmap='gray')
        ax[1].imshow(raw_img, cmap='gray')
        for i in range(len(indices)):
            node = indices[i, 0]
            for g in range(5):
                node_2 = indices[i, g]
                if node_2 >= len(indices):
                    continue
                ax[1].plot([cent[node, 1], cent[node_2, 1]], [cent[node, 0], cent[node_2, 0]], 'r-')

        ax[1].plot(cent[:, 1], cent[:, 0], 'b.', markersize=10)
        ax[1].axis('off')
        ax[0].axis('off')
        ax[0].set_title('Original image')
        ax[1].set_title('Graph neural network')

        fig.tight_layout()
        plt.show()
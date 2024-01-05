"""

Functions for feature and k-Nearest Neighbor calculations 
"""

import numpy as np

#image packages
from skimage.transform import resize
from skimage.measure import regionprops
from skimage.feature import graycomatrix, graycoprops

#Graph packages
from scipy.spatial import KDTree as KDTree

def calc_features(label_img, nuc_img):

            """
            Calculate morphological features.

            Parameters:
            label_img: ndarray of labeled mask of nuclei
            nuc_img: ndarray of normalized intensity image

            Returns:
            feature_arr: 1 x 16 ndarray of features

            """

            feature_arr = np.zeros((1, 16))

            props = regionprops(label_img)

            img_temp = nuc_img.astype(np.uint8)
            dist = [1, 2]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            properties = ['ASM', 'contrast', 'dissimilarity', 'energy', 'homogeneity', 'correlation']


            gm = graycomatrix(img_temp, distances=dist, angles=angles, symmetric=True, normed=True)

            i = 0

            feature_arr[i, 0] = props[0].area
            feature_arr[i, 1] = props[0].area_convex
            feature_arr[i, 2] = props[0].axis_major_length
            feature_arr[i, 3] = props[0].axis_minor_length
            feature_arr[i, 4] = props[0].feret_diameter_max
            feature_arr[i, 5] = props[0].orientation
            feature_arr[i, 6] = props[0].eccentricity
            feature_arr[i, 7] = props[0].extent
            feature_arr[i, 8] = props[0].solidity
            feature_arr[i, 9] = props[0].perimeter

            for n in range(10, 16):
                feature_arr[i, n] = np.sum(graycoprops(gm, properties[n-13]).ravel())

            return feature_arr

def calculate_KNN(positions,
				  nearest_neighbors:int=5,
				  dist_threshold:int=300):

	"""
    Calculate KNN graph
    Parameters:
    positions: N x 2 ndarray of nuclei centroid positions
    nearest_neighbors: int, number of NN for the KNN graph
    dist_threshold: int, maximum distance allowed for an edge

    Return:
    indices: N x nearest_neighbors ndarray with nearest neighbors for each node
	"""

	tree = KDTree(positions)
	dist, indices = tree.query(positions, k=nearest_neighbors, distance_upper_bound = dist_threshold)

	return indices
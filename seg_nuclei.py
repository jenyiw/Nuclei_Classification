"""
Functions to segment nuclei and create a 512 x 512 image with the nuclei of interest in the center
"""
import os
import h5py
import numpy as np

from skimage.io import imsave, imread
from skimage.transform import resize
from skimage.measure import regionprops
from skimage.morphology import opening
from skimage.filters import gaussian, laplace

import seg_tools as sf
import features
from stardist.models import StarDist2D


def generate_nuclei_images(hd_name:str,
                    total_num:int,
                    list_patches:list,
                    save_path:str,
                    ):

    """
    Create a hdf5 file with one nuclei in every image. Background is 0. 

    Parameters:
    had_name: str, name of hdf5 file to save to
    total_num:int, total number of samples to create
    list_patches: List, list of patch names to process
    save_path: str, save path

    Return:
    None
    """

    #create h5py file
    with h5py.File(os.path.join(save_path, hd_name), 'a') as f:
        f.create_dataset('data', data=np.zeros((total_num, 512, 512)))
        f.attrs['num_samples'] = total_num

    patch_list = []
    raw_list = []
    #Create patches for individual nuclei
    #iterate over patch numbers
    for patch in list_patches:

        #check if required stitched image exist
        if not os.path.exists(os.path.join(save_path, f'{patch}_nuc_label.npy')):
            print(f'Stitched labeled image of {patch} does not exist. Skipping...')
            continue

        #load stitched image        
        canvas = np.load(os.path.join(save_path, f'{patch}_nuc_label.npy'))
        raw_canvas = imread(os.path.join(save_path, f'{patch}_stitched.jpg'))

        #get number of labels and area of each nuclei
        labels, count_labels = np.unique(canvas, return_counts=True)
        labels = labels[1:]
        count_labels = count_labels[1:]
        
        if total_num < len(labels):
            num_to_get = total_num
        else:
            num_to_get = len(labels)

        #create an image for each nuclei
        for i in range(num_to_get):

            l = labels[i]
            c = count_labels[i]

            #Reject if size is too small
            if c < 50:
                continue

            #get nuclei from raw image
            temp_arr = canvas.copy()
            temp_arr[temp_arr != l] = 0

            resized_label = resize(temp_arr, (dims[0], dims[1]))
            temp_nuc = np.zeros((512, 512))

            #get dimensions of nuclei
            coords = np.where(resized_label > 0)

            min_row = np.min(coords[0])
            max_row = np.max(coords[0])
            min_col = np.min(coords[1])
            max_col = np.max(coords[1])

            #  print(min_row, max_row, min_col, max_col)

            #get nuclei from raw image
            temp_nuc = raw_canvas[min_row:max_row, min_col:max_col]
            temp_nuc[resized_label != 1] = 0

            #get dimensions
            height_pad_top = int((512-temp_nuc.shape[0])//2)
            height_pad_bot = int(512-height_pad_top-temp_nuc.shape[0])
            width_pad_left = int((512-temp_nuc.shape[1])//2)
            width_pad_right = int(512-width_pad_left-temp_nuc.shape[1])

            #reject extremely large nuclei
            if (height_pad_top < 0) or (height_pad_bot < 0) or (width_pad_left < 0) or (width_pad_right < 0):
                print('skipping')
                continue

            #pad image to the right dimensions
            temp_nuc = np.pad(temp_nuc, ((height_pad_top, height_pad_bot),(width_pad_left, width_pad_right)))

            with h5py.File(hd_name, 'a') as g:
                g['data'][total_num-1, ...] = temp_nuc.astype(np.float32)
            patch_list.append(f'{patch}_nuc_label.npy')
            raw_list.append(f'{patch}_stitched.jpg')

            total_num -= 1
            if total_num%100 == 0:
                print(f'Number left to process: {total_num}')

    with h5py.File(os.path.join(save_path, hd_name), 'a') as f:
        f.attrs['roi_name_label'] = patch_list    
        f.attrs['roi_name_raw'] = raw_list                 

def generate_patches(hd_name:str,
                    total_num:int,
                    list_patches:list,
                    save_path:str,
                    ):

    """
    Create a hdf5 file with patches from raw stitched image.

    Parameters:
    had_name: str, name of hdf5 file to save to
    total_num:int, total number of samples to create
    list_patches: List, list of patch names to process
    save_path: str, save path

    Return:
    None
    """

    #create h5py file
    with h5py.File(os.path.join(save_path, hd_name), 'a') as f:
        f.create_dataset('data', data=np.zeros((total_num, 512, 512)))
        f.attrs['num_samples'] = total_num

    patch_list = []
    raw_list = []

    #Create patches for individual nuclei
    #iterate over patch numbers
    for patch in list_patches:

        #check if required stitched image exist
        if not os.path.exists(os.path.join(save_path, f'{patch}_nuc_label.npy')):
            print(f'Stitched labeled image of {patch} does not exist. Skipping...')
            continue

        #load stitched image        
        canvas = np.load(os.path.join(save_path, f'{patch}_nuc_label.npy'))
        raw_canvas = imread(os.path.join(save_path, f'{patch}_stitched.jpg'))

        #get number of labels and area of each nuclei
        props = regionprops(canvas)
        
        if total_num < len(props):
            num_to_get = total_num
        else:
            num_to_get = len(props)

        #create an image for each nuclei
        for i in range(num_to_get):

            #Reject if size is too small
            if props[i].area < 50:
                continue

            #get nuclei patch
            cent_x, cent_y = props[i].centroid
            new_cent_x = int(cent_x * 4)
            new_cent_y = int(cent_y * 4)

            bbox = raw_canvas[new_cent_x-256: new_cent_x+256, new_cent_y-256: new_cent_y+256]

            if (np.min(bbox.shape) < 512) or (np.sum(bbox) == 0):
                continue

            #save nuclei patch
            with h5py.File(hd_name, 'a') as g:
                g['data'][total_num-1, ...] = bbox.astype(np.float32)
            patch_list.append(f'{patch}_nuc_label.npy')
            raw_list.append(f'{patch}_stitched.jpg')

            total_num -= 1
            if total_num%100 == 0:
                print(f'Number left to process: {total_num}')

    with h5py.File(os.path.join(save_path, hd_name), 'a') as f:
        f.attrs['roi_name_label'] = patch_list    
        f.attrs['roi_name_raw'] = raw_list   

def generate_graph_data(hd_name:str,
                    total_num:int,
                    list_patches:list,
                    save_path:str,
                    ):

    """
    Create files for graph input for the GNN.

    Parameters:
    had_name: str, name of hdf5 file to save to
    total_num:int, total number of samples to create
    list_patches: List, list of patch names to process
    save_path: str, save path

    Return:
    None
    """

    #create hdf5 file
    with h5py.File(hd_name, 'w') as f:
        f.create_dataset('data', data=np.zeros((total_num, 6, 16)))
        f.create_dataset('edges', data=np.zeros((total_num, 5)))
        f.attrs['num_samples'] = total_num

    patch_list = []
    raw_list = []        

    #Create patches for individual nuclei
    #iterate over patch numbers
    for patch in list_patches:

        #check if required stitched image exist
        if not os.path.exists(os.path.join(save_path, f'{patch}_nuc_label.npy')):
            print(f'Stitched labeled image of {patch} does not exist. Skipping...')
            continue

        #load stitched image        
        canvas = np.load(os.path.join(save_path, f'{patch}_nuc_label.npy'))
        raw_canvas = imread(os.path.join(save_path, f'{patch}_stitched.jpg'))
        dims = raw_canvas.shape

        #get number of labels and area of each nuclei
        props = regionprops(canvas)

        #get edges with k nearest neighbors
        positions = np.zeros((len(props), 2))
        for num_regions, region in enumerate(props):
            label_value = canvas[int(region.centroid[0]), int(region.centroid[1])]
            positions[num_regions] = region.centroid

        edges = features.calculate_KNN(positions)        

        if total_num < len(props):
            num_to_get = total_num
        else:
            num_to_get = len(props)

        #create an image for each nuclei
        for i in range(num_to_get):

            #Reject if size is too small
            if props[i].area < 50:
                continue

            temp_arr = canvas.copy()
            temp_arr[temp_arr != props[i].label] = 0

            resized_label = resize(temp_arr, (dims[0], dims[1]))
            resized_label[resized_label > 0] = 1

            #get nuclei patch
            top, bottom, left, right = props[i].bbox

            temp_nuc = raw_canvas[top*4:bottom*4, left*4, right*4]
            temp_nuc[resized_label != 1] = 0
            temp_feature = features.calc_features(resized_label.astype(np.uint8), temp_nuc)        

            #save nuclei graph and other attributes
            with h5py.File(hd_name, 'a') as g:
                g['data'][total_num-1, ...] = temp_feature
                g['edges'][total_num-1, ...] = list(edges[i])                
            patch_list.append(f'{patch}_nuc_label.npy')
            raw_list.append(f'{patch}_stitched.jpg')


            total_num -= 1
            if total_num%100 == 0:
                print(f'Number left to process: {total_num}')

    with h5py.File(os.path.join(save_path, hd_name), 'a') as f:
        f.attrs['roi_name_label'] = patch_list    
        f.attrs['roi_name_raw'] = raw_list                      
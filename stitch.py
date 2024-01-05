"""
Functions for plotting low resolution images and visualizing graph

"""

import numpy as np
import os
import re
import glob
import matplotlib.pyplot as plt
from skimage.io import imsave, imread
from skimage import exposure
import shutil

def get_dims(filepath:str,
            filelist:list):
    
    """
    Get dimensions of images in folder. Assumes all images have the same dimensions.

    Parameters:
    filepath: str, folder where images are stored
    filelist: List, list of names of images in folder

    Returns:
    dims: dimensions of images
    """
      
    img = imread(os.path.join(filepath, filelist[0]))
    min_axis = np.argmin(img.shape)

    #images are processed as (channels, X, Y)
    img = np.moveaxis(img, min_axis, 0)
    dims = img.shape

    return dims


def stitch_images(filepath,
					 savepath,
					 run_name,
					 scale:int=2,
					 chnl:int=0,
                     show_canvas:bool=False):
      
	"""
    Stitch images together.

    Parameters:
    filepath: str, folder where images are stored.
    savepath: str, folder to store images.
    run_name: str, identifier for unique run/region
    scale: int, amount to scale down images by
    chnl: int, image channel to stitch
    show_canvas: bool, whether to show stitched image after stitching

    Returns:
    None
    
	"""

	print('Getting low resolution stitched image for run:', run_name)
	chnl_dict = {0: 'nuc',
			  1: 'WGA',
			  2: 'TEL',
			  3: 'CENPB'}

	file_list = os.listdir(filepath)

    #get dimensions for calculations
	dims = get_dims(filepath, file_list)
      
    #calculate x length and y length of stitched image
	max_y = 0
	max_x = 0
	for f in file_list:
		if f'_{run_name}_' in f:
			y = int(re.search('(?<=y=)\d+', f).group(0))
			if y > max_y:
				max_y = y
			x = int(re.search('(?<=x=)\d+', f).group(0))
			if x > max_x:
				max_x = x
                        
    #initialize canvas to insert images on
	canvas = np.zeros((int((max_x+dims[1])/scale), int((max_y+dims[1])/scale)), dtype=np.float64)
	scaled_dims = int(dims[1]/scale)


	for f in file_list:
		if f'_{run_name}_' in f:
            
            #get x and y coordinates
			y = int(int(re.search('(?<=y=)\d+', f).group(0))/scale)
			x = int(int(re.search('(?<=x=)\d+', f).group(0))/scale)
                  
            #read and scale image
			img = imread(os.path.join(filepath, f))
			min_axis = np.argmin(img.shape)
			img = np.moveaxis(img, min_axis, 0)[chnl]
	# 		img = np.clip(img, a_min=0, a_max=np.percentile(img, 95))
			new_img = img[::scale, ::scale]
                  
            #insert image on canvas
			canvas[x:x+scaled_dims, y:y+scaled_dims] = new_img

	canvas = (canvas/np.max(canvas)*255).astype(np.uint8)
      
    #show stitched image      
	if show_canvas:
		plt.imshow(canvas)
		plt.show()
          
    #save stitched image
	imsave(os.path.join(savepath, f'{run_name}_{chnl_dict[chnl]}.jpg'), canvas)



if __name__ == "__main__":

    filepath = '/content/Smooth_muscle/Smooth_muscle'
    savepath = './Images/Smooth_muscle'

    for p in range(127, 128):
        stitch_images(filepath, savepath, p, scale=1, chnl=0)

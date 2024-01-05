# -*- coding: utf-8 -*-
"""
Code to segment nuclei using StarDist
"""

import os
import numpy as np
# import cv2

from stardist.models import StarDist2D

# prints a list of available models
# StarDist2D.from_pretrained()

# creates a pretrained model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
from skimage.io import imread


from skimage.measure import regionprops_table
from skimage.transform import resize
from skimage.filters import gaussian, laplace

def segment_preprocess(nucleus_img, 
						   background_img=None,
						   scale:int=4):
	
		"""
		Process nuclei images by subtracting background fluorscence and applying a Gaussian blur.
		
		Parameters:
			nucleus_img: 2048 x 2048 nuclei image
			background_img: 2048 x 2048 background image		
			scale: Proportion to downsize. Default: 4
			
		Returns:
			blurred_img: 512 x 512 processed nuclei image
		
		"""
			
# 		norm_nuc_img = (nucleus_img/np.max(nucleus_img))[::scale, ::scale]	
# 		norm_bg_img = (background_img/np.max(background_img))[::scale, ::scale]	
# 		
# 		nuc_img = norm_nuc_img - norm_bg_img
		blurred_img = gaussian(nucleus_img[::scale, ::scale])
		blurred_img = np.clip(blurred_img, a_min=0, a_max=np.percentile(blurred_img, 98))
		blurred_img = blurred_img/np.max(blurred_img)
# 		blurred_img = gaussian(nuc_img)		
		
		return blurred_img



def segment_nuclei(img, 
					   model,
					   min_area=200):
	
			"""
		segment nuclei using StarDist and remove nuclei with < 200 pixel in area.
		
		Parameters:
			img: 512 x 512 downsized nuclei image
			model: Pretrained StarDist model
			min_area: minimum area to accept a nuclei. Default: 200
			
		Returns:
			labels: 512 x 512 labeled image
		
			"""
	
		
			labels, _ = model.predict_instances(img)
			unique, area = np.unique(labels, return_counts=True)
			for i, n in enumerate(unique):
	 			if n != 0 and area[i] < min_area:
	 				 labels[labels == n] = 0 			  
	
# 			plt.figure(figsize=(30,10))
# 			plt.subplot(1, 2, 1)
# 			plt.imshow(img, cmap='gray')
# 			plt.title('Original')
# 			plt.axis("off")
# 			plt.subplot(1, 2, 2)
# 			plt.title('Labelled image')
# 			plt.imshow(labels)
# 			plt.axis("off")
# 			plt.show()
# 			plt.close()
			
			return labels
		
def get_properties(labels, img, peak_img, downsize:bool=False):	

		"""
		Calculate feature properties for each image. Get properties from downsized 512x512 images due to computational limitations. 
		
		Parameters:
			labels: 512 x 512 labeled image
			img: 2048 x 2048 original nuclei image
			peak_img: 2048 x 2048 processed peak images
			downsize: bool. Whether to downsize the images
			
		Returns:
			feature_arr: (# of nuclei, 8) feature array
		
		"""

		
		if not downsize:
		    resized_label = resize(labels, (img.shape[0], img.shape[1]), preserve_range=True).astype(np.uint8)
		    peak_img_binned = peak_img
			
		else:
		    dim_size = img.shape[0]
		    img = img.reshape(dim_size//4, 4, dim_size//4, 4).mean(-1).mean(1)	
		    resized_label = labels	
		    peak_img_binned = peak_img.reshape(dim_size//4, 4, dim_size//4, 4).sum(-1).sum(1)

		props = regionprops_table(labels, 
                          properties=['label', 'centroid', 'axis_major_length',
                                      'area', 'perimeter', 'area_convex'])
		
		feature_arr = np.zeros((len(props['label']), 8))						
		lap_img = laplace(img)

		
		#get nuclei properties	
		feature_arr[:, 0] = props['centroid-0']			
		feature_arr[:, 1] = props['centroid-1']			
		feature_arr[:, 2] = props['area']
		feature_arr[:, 3] = props['axis_major_length']
		feature_arr[:, 4] = 4*np.pi*props['area']/((props['perimeter'])**2)
		feature_arr[:, 5] = props['area']/props['area_convex']
			
		for i, n in enumerate(props['label']):				
			feature_arr[i, 6] = np.var(lap_img[resized_label==n])
	
			#get peak count for FISH channels
			feature_arr[i, 7] = np.sum(peak_img_binned[resized_label==n])//255
	
		feature_arr = feature_arr[~np.all(feature_arr == 0, axis=1)]
		
		return feature_arr, 0

def remove_edge_labels(label_img):

    """
      Remove nuclei touching the edges.

      Parameters:
      label_img: ndarray of label mask of nuclei

      Return:
      label_img: ndarray of edited label mask
    """

    left_edge = np.unique(label_img[:, 0])
    right_edge = np.unique(label_img[:, -1])
    top_edge = np.unique(label_img[0, :])
    bottom_edge = np.unique(label_img[-1, :])

    remove_unique = np.concatenate((left_edge, right_edge, top_edge, bottom_edge))

    for n in remove_unique:
        if n != 0:
            label_img[label_img == n] = 0

    return label_img

def stitch_label_images(canvas,
                    sd_model,
                    patch,
                    save_path,
                    ):

    """
    Segment and save label masks.

    Parameters:
    canvas: ndarray of larger stitched image
    sd_model: StarDist model
    patch: int, patch number for saving images
    save_path: str, save path

    Return:
    new_canvas: ndarray, stitched label images
    """

    new_canvas = np.zeros_like(canvas).astype(np.uint16)

    #make image smaller to match expectations of the StarDist model
    new_canvas = new_canvas[::4, ::4]

    max_nuc = 0

    #get number of rows and columns
    num_rows = int(np.ceil(canvas.shape[0]/1024))
    num_cols = int(np.ceil(canvas.shape[1]/1024))

    for n in range(num_rows-1):
      for m in range(num_cols-1):         
         
          #get images of 2048 x 2048 at every 1024 interval
          temp_arr = canvas[n*1024:n*1024+2048, m*1024:m*1024+2048]
          prep_img = sf.segment_preprocess(temp_arr)

          #get image with labeled nuclei and remove nuclei touching the edge
          label_img = sf.segment_nuclei(prep_img, sd_model)
          label_img = sf.remove_edge_labels(label_img).astype(np.uint16)
          label_img = label_img + max_nuc
          label_img[label_img == np.min(label_img)] = 0

          labels = np.unique(label_img)

          #deconflict overlapping nuclei
          for l in labels:
            new_sum = np.sum(label_img[label_img == l])
            sub_canvas = new_canvas[n*256:n*256+512, m*256:m*256+512]
            canvas_sum = np.sum(sub_canvas[label_img == l])
            if new_sum*canvas_sum != 0:
               label_img[label_img == l] = 0

          #stitch processed label image
          new_canvas[n*256:n*256+512, m*256:m*256+512] += label_img
          max_nuc = np.max(label_img)

    new_canvas = new_canvas.astype(np.uint16)

    #save stitched label image
    np.save(os.path.join(save_path, f'{patch}_nuc_label.npy'), new_canvas)


def stitch_all_patches(file_path:str,
                    sd_model,
                    patch_min:int,
                    patch_max:int,
                    save_path:str,
                    ):

    """
    Get segmented labeled images for each nuclei and create a stitched labeled image.

    Parameters:
    file_path: str, folder where images are stored
    sd_model: pre-trained StarDist model
    patch_min: int, minimum patch number to process 
    patch_max: int, maximum patch number to process     
    save_path: str, save path

    Return:
    None
    """

    # get pre-trained stardist model
    file_list = os.listdir(file_path)
    sd_model = StarDist2D.from_pretrained('2D_versatile_fluo')

    #iterate over patch numbers
    for n in range(patch_min, patch_max):

        #create stitched label images
        for file_name in file_list:

            #process only stitched raw images from certain regions
            if 'stitch_raw' not in file_name:
               continue

            if n not in file_name:
               continue

            file_name = file_name
            canvas = np.load(os.path.join(file_path, file_name))
            stitch_label_images(canvas, sd_model, n, save_path)

    print('Done creating stitched labeled images')	
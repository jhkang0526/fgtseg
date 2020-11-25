'''
one nifti file load.
'''

import nibabel as nb
import os
import sys
import numpy as np
import math
import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from nilearn.image import resample_img

from typing import Dict, Sequence, Optional, Callable

from pathlib import Path

import warnings

warnings.filterwarnings(action='ignore') 

class dataset(Dataset):

    def __init__(self, 
                 path, 
                 mode='test'
                ):

        '''
        data loader
            path: path dict list from csv file
            mode: default 'test', chioce = ['test', 'train', 'valid'], but this code support only 'test'
            list: output
        '''
        self.path = path
        self.mode = str(mode)  
        self.list = [] 
       
        ## use in test
        if mode == 'test':
            self.create_test_data()

        
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, idx):
        return self.list[idx]

    def create_test_data(self):
        '''
        img: image array
        affine: low resolution affine
        affine_org: original affine
        shape_org: original image array shape
        '''
        img, affine, affine_org, shape_org = load_dataset_ts(self.path, self.mode)
        total = len(img)

        self.list.append(tuple((img, affine, affine_org, shape_org)))

#         print(str(idx)+' ', end='', flush=True)
            
        print("end", flush=True)

def load_data_ts(file_path,normalize = True):
    ## nifti file load
    volume_nifty = nb.load(file_path['img'])

    ### nifti file info
    volume_orig_affine = volume_nifty.affine
    volume_orig_shape = volume_nifty.get_fdata().shape
    
    ## check
    ### down sampling
    volume_nifty_re = resample_img(
        img=volume_nifty,
        target_affine=np.diag([1.5, 1.5, 3]),
        interpolation='continuous'
    )    
    
    volume = volume_nifty_re.get_fdata()
    
    if normalize:
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

#     print(volume.max())
    return volume, volume_nifty_re.affine, volume_orig_affine, volume_orig_shape

def load_dataset_ts(file_paths, mode,normalize=True):
    print(f"Loading and preprocessing {mode} data...")

    volume, affine, affine_orig, shape = load_data_ts(file_paths, normalize)

    return volume, affine, affine_orig, shape

###################################################################################
'''
Article{miscnn,
  title={MIScnn: A Framework for Medical Image Segmentation with Convolutional Neural Networks and Deep Learning},
  author={Dominik Müller and Frank Kramer},
  year={2019},
  eprint={1910.09308},
  archivePrefix={arXiv},
  primaryClass={eess.IV}
}

https://github.com/frankkramer-lab/MIScnn/blob/master/miscnn/utils/patch_operations.py
'''
def slice_3Dmatrix(array, window, overlap):
    # Calculate steps
    steps_x = int(math.ceil((len(array) - overlap[0]) /
                            float(window[0] - overlap[0])))
    steps_y = int(math.ceil((len(array[0]) - overlap[1]) /
                            float(window[1] - overlap[1])))
    steps_z = int(math.ceil((len(array[0][0]) - overlap[2]) /
                            float(window[2] - overlap[2])))

    # Iterate over it x,y,z
    patches = []
    for x in range(0, steps_x):
        for y in range(0, steps_y):
            for z in range(0, steps_z):
                # Define window edges
                x_start = x*window[0] - x*overlap[0]
                x_end = x_start + window[0]
                y_start = y*window[1] - y*overlap[1]
                y_end = y_start + window[1]
                z_start = z*window[2] - z*overlap[2]
                z_end = z_start + window[2]
                # Adjust ends
                if(x_end > len(array)):
                    # Create an overlapping patch for the last images / edges
                    # to ensure the fixed patch/window sizes
                    x_start = len(array) - window[0]
                    x_end = len(array)
                    # Fix for MRIs which are smaller than patch size
                    if x_start < 0 : x_start = 0
                if(y_end > len(array[0])):
                    y_start = len(array[0]) - window[1]
                    y_end = len(array[0])
                    # Fix for MRIs which are smaller than patch size
                    if y_start < 0 : y_start = 0
                if(z_end > len(array[0][0])):
                    z_start = len(array[0][0]) - window[2]
                    z_end = len(array[0][0])
                    # Fix for MRIs which are smaller than patch size
                    if z_start < 0 : z_start = 0
                # Cut window
                window_cut = array[x_start:x_end,y_start:y_end,z_start:z_end]
                # Add to result list
                patches.append(window_cut)
                
    return patches

def concat_3Dmatrices(patches, image_size, window, overlap):
    # Calculate steps
    steps_x = int(math.ceil((image_size[0] - overlap[0]) /
                            float(window[0] - overlap[0])))
    steps_y = int(math.ceil((image_size[1] - overlap[1]) /
                            float(window[1] - overlap[1])))
    steps_z = int(math.ceil((image_size[2] - overlap[2]) /
                            float(window[2] - overlap[2])))

    # Iterate over it x,y,z
    matrix_x = None
    matrix_y = None
    matrix_z = None
    pointer = 0
    for x in range(0, steps_x):
        for y in range(0, steps_y):
            for z in range(0, steps_z):
                # Calculate pointer from 3D steps to 1D list of patches
                pointer = z + y*steps_z + x*steps_y*steps_z
                # Connect current patch to temporary Matrix Z
                if z == 0:
                    matrix_z = patches[pointer]
                else:
                    matrix_p = patches[pointer]
                    # Handle z-axis overlap
                    slice_overlap = calculate_overlap(z, steps_z, overlap,
                                                      image_size, window, 2)
                    matrix_z, matrix_p = handle_overlap(matrix_z, matrix_p,
                                                        slice_overlap,
                                                        axis=2)
                    matrix_z = np.concatenate((matrix_z, matrix_p),
                                              axis=2)
            # Connect current tmp Matrix Z to tmp Matrix Y
            if y == 0:
                matrix_y = matrix_z
            else:
                # Handle y-axis overlap
                slice_overlap = calculate_overlap(y, steps_y, overlap,
                                                  image_size, window, 1)
                matrix_y, matrix_z = handle_overlap(matrix_y, matrix_z,
                                                    slice_overlap,
                                                    axis=1)
                matrix_y = np.concatenate((matrix_y, matrix_z), axis=1)
        # Connect current tmp Matrix Y to final Matrix X
        if x == 0:
            matrix_x = matrix_y
        else:
            # Handle x-axis overlap
            slice_overlap = calculate_overlap(x, steps_x, overlap,
                                              image_size, window, 0)
            matrix_x, matrix_y = handle_overlap(matrix_x, matrix_y,
                                                slice_overlap,
                                                axis=0)
            matrix_x = np.concatenate((matrix_x, matrix_y), axis=0)
    # Return final combined matrix
    return(matrix_x)

#-----------------------------------------------------#
#          Subroutines for the Concatenation          #
#-----------------------------------------------------#
# Calculate the overlap of the current matrix slice
def calculate_overlap(pointer, steps, overlap, image_size, window, axis):
            # Overlap: IF last axis-layer -> use special overlap size
            if pointer == steps-1 and not (image_size[axis]-overlap[axis]) \
                                            % (window[axis]-overlap[axis]) == 0:
                current_overlap = window[axis] - \
                                  (image_size[axis] - overlap[axis]) % \
                                  (window[axis] - overlap[axis])
            # Overlap: ELSE -> use default overlap size
            else:
                current_overlap = overlap[axis]
            # Return overlap
            return current_overlap

# Handle the overlap of two overlapping matrices
def handle_overlap(matrixA, matrixB, overlap, axis):
    # Access overllaping slice from matrix A
    idxA = [slice(None)] * matrixA.ndim
    matrixA_shape = matrixA.shape
    idxA[axis] = range(matrixA_shape[axis] - overlap, matrixA_shape[axis])
    sliceA = matrixA[tuple(idxA)]
    # Access overllaping slice from matrix B
    idxB = [slice(None)] * matrixB.ndim
    idxB[axis] = range(0, overlap)
    sliceB = matrixB[tuple(idxB)]
    # Calculate Average prediction values between the two matrices
    # and save them in matrix A
    matrixA[tuple(idxA)] = np.mean(np.array([sliceA, sliceB]), axis=0)
    # Remove overlap from matrix B
    matrixB = np.delete(matrixB, [range(0, overlap)], axis=axis)
    # Return processed matrices
    return matrixA, matrixB


                
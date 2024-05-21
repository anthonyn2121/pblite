#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""
import sys
import os

# Add the parent directory to the Python path to be able to use utils directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Code starts here:
import numpy as np
import cv2
import filter
from utils.image_utils import save_image, rotate_image, sobel_x, convolve2d, plot_subplot_images

def generate_DoG_bank(n_orientations:int, sigmas:list, sizes:list) -> list:
    '''! Generates a collection of oriented DoG filters

    @param n_orientations   Number of orientations that the DoG filter will be rotated
    @param sigmas           List of sigma values to include in the calculation of the gaussian kernel
    @param sizes            List of sizes of the gaussian kernel 
    
    @return List of np.arrays that describe a rotated DoG filter
    '''
    assert len(sigmas) == len(sizes), "Size of 'sigmas' and 'sizes' should be the same"

    DoG_bank = []
    for sigma, size in zip(sigmas, sizes):
        gaussian_kernel = filter.gaussian2d(size, sigma, sigma, 0, 0)
        DoG = convolve2d(gaussian_kernel, sobel_x())
        for i in range(n_orientations):
            angle = i * (360 / n_orientations)
            DoG_bank.append(rotate_image(DoG, angle))
    return DoG_bank


def generate_LM_bank(size:int, n_orientations:int, DoG_sigmas:list, LoG_sigmas:list, Gauss_sigmas:list):
    LM_bank = []
    orders = [1, 2]
    elongation_factor = 3

    ## Generate first and second order derivates of Gaussian - Total: 36
    for order in orders:
        for sigma in DoG_sigmas:
            gauss = filter.gaussian2d(size, 3* sigma, sigma, 0, order)
            for i in range(n_orientations):
                angle = i * (360 / n_orientations)
                LM_bank.append(rotate_image(gauss, angle))

    ## Generate LoG filters - Total: 8
    for sigma in LoG_sigmas:
        LM_bank.append(
            filter.laplacian_of_gaussian(size, sigma)
        )

    for sigma in LoG_sigmas:
        LM_bank.append(
            filter.laplacian_of_gaussian(size, 3 * sigma)
        )

    ## Generate Gaussian filters - Total: 4
    for sigma in Gauss_sigmas:
        LM_bank.append(
            filter.gaussian2d(size, sigma, sigma, 0, 0)
        )

    return LM_bank

def main():

    """
    Generate Difference of Gaussian Filter Bank: (DoG)
    Display all the filters in this filter bank and save image as DoG.png,
    use command "cv2.imwrite(...)"
    """
    orientations = 16
    sigmas = [1, 2]
    sizes = [17, 17]
    DoG_Bank = generate_DoG_bank(orientations, sigmas, sizes)
    for n, img in enumerate(DoG_Bank):
        save_image(img, "Phase1/DoG_Filters", f'DoG{n}.png')

    """
    Generate Leung-Malik Filter Bank: (LM)
    Display all the filters in this filter bank and save image as LM.png,
    use command "cv2.imwrite(...)"
    """
    LMS_bank = generate_LM_bank(49, 6, DoG_sigmas=[1, np.sqrt(2), 2],
                                LoG_sigmas=[1, np.sqrt(2), np.sqrt(3), 2],
                                Gauss_sigmas=[np.sqrt(2), 2, 2 * np.sqrt(2), 4])
    for n, img in enumerate(LMS_bank):
        save_image(img, "Phase1/LM_Filters", f'LM{n}.png')
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=4, ncols=12, figsize=(12, 4))
    for i, ax in enumerate(axes.flat):
        if i < len(LMS_bank):
            ax.imshow(LMS_bank[i], cmap='gray')
            ax.axis('off')
    plt.show()

    """
    Generate Gabor Filter Bank: (Gabor)
    Display all the filters in this filter bank and save image as Gabor.png,
    use command "cv2.imwrite(...)"
    """


    """
    Generate Half-disk masks
    Display all the Half-disk masks and save image as HDMasks.png,
    use command "cv2.imwrite(...)"
    """



    """
    Generate Texton Map
    Filter image using oriented gaussian filter bank
    """


    """
    Generate texture ID's using K-means clustering
    Display texton map and save image as TextonMap_ImageName.png,
    use command "cv2.imwrite('...)"
    """


    """
    Generate Texton Gradient (Tg)
    Perform Chi-square calculation on Texton Map
    Display Tg and save image as Tg_ImageName.png,
    use command "cv2.imwrite(...)"
    """


    """
    Generate Brightness Map
    Perform brightness binning 
    """


    """
    Generate Brightness Gradient (Bg)
    Perform Chi-square calculation on Brightness Map
    Display Bg and save image as Bg_ImageName.png,
    use command "cv2.imwrite(...)"
    """


    """
    Generate Color Map
    Perform color binning or clustering
    """


    """
    Generate Color Gradient (Cg)
    Perform Chi-square calculation on Color Map
    Display Cg and save image as Cg_ImageName.png,
    use command "cv2.imwrite(...)"
    """


    """
    Read Sobel Baseline
    use command "cv2.imread(...)"
    """


    """
    Read Canny Baseline
    use command "cv2.imread(...)"
    """


    """
    Combine responses to get pb-lite output
    Display PbLite and save image as PbLite_ImageName.png
    use command "cv2.imwrite(...)"
    """
    
if __name__ == '__main__':
    main()
 



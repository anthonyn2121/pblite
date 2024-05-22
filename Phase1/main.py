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

import sklearn.cluster

# Add the parent directory to the Python path to be able to use utils directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Code starts here:
import numpy as np
import cv2
import sklearn
import filter
from utils.image_utils import save_image, plot_image, rotate_image, sobel_x, convolve2d, plot_subplot_images

def convolve_image(image:np.array, filters:list) -> np.array:
    ''' Convolve image 2d with list of filters

    @param image    Np.array representation of image
    @param filters  List of filters to convolve image with

    @return masks  a (L x W x N) array where the depth of the array is one result of the convolved image
    '''
    masks = np.array(image)
    for i, filter in enumerate(filters):
        convolved_image = convolve2d(image, filter)
        masks = np.dstack((masks, convolved_image))
    return masks

def half_circle_mask(size:tuple, radius:int) -> np.array:
    ''' Generate semi-circle 'images' 
    
    @param size     Size of the NxN image
    @param radius   Radius of the circle

    @return The 'image' as an array
    '''
    length, width = size
    center_x, center_y = length//2, width//2
    ## Create array of distances from center
    mask = np.fromfunction(
        lambda x, y: np.sqrt((center_x - x)**2 + (center_y - y)**2),
        shape=size 
    )
    ## Make mask binary depending on distance to radius - Creating circle
    mask[mask <= radius] = 1
    mask[mask >= radius] = 0
    ## Cover half the mask to create a half circle
    mask[:, :(width//2)] = 0
    return mask 

def generate_DoG_bank(n_orientations:int, sigmas:list, sizes:list) -> list:
    ''' Generates a collection of oriented DoG filters

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
    ''' Generate a filter bank composed of Derivative of Gaussian(DoG), Laplacian of Gaussian(LoG),
        and regular Gaussian filters.

    @param size                 Size of all NxN kernels
    @param n_orientations       Number of orientations to rotate the DoG filters
    @param DoG_sigmas           Sigma values used for various DoG filters
    @param LoG_sigmas           Sigma values used for various LoG filtes
    @param Gauss_sigmas         Sigma values used for various Gaussian filters

    @return List fo np.arrays that describe the filters
    '''
    LM_bank = []
    orders = [1, 2]
    elongation_factor = 3

    ## Generate first and second order derivates of Gaussian - Total: 36
    for order in orders:
        for sigma in DoG_sigmas:
            gauss = filter.gaussian2d(size, elongation_factor * sigma, sigma, 0, order)
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
            filter.laplacian_of_gaussian(size, elongation_factor * sigma)
        )

    ## Generate Gaussian filters - Total: 4
    for sigma in Gauss_sigmas:
        LM_bank.append(
            filter.gaussian2d(size, sigma, sigma, 0, 0)
        )

    return LM_bank

def generate_gabor_filters(size:int, n_orientations:int, sigmas:list, theta:float, Lambda:int, psi:int, gamma:int) -> list:
    filters = []
    orientations = np.linspace(90, 270, n_orientations)
    for sigma in sigmas:
        g = filter.gabor(sigma, theta, Lambda, psi, gamma)
        for i in orientations:
            filters.append(rotate_image(g, i))
    return filters

def generate_halfcircle_filters(size:list, n_orientations:int, radius:list):
    filters = []
    for s in size:
        for r in radius:
            for n in range(0, n_orientations):
                mask = half_circle_mask((s, s), r)
                angle = n * (360 / n_orientations)
                filters.append(
                    rotate_image(mask, angle)
                )
    return filters

def generate_texton_map(image:np.array, filters:list, clusters:int):
    masks = convolve_image(image, filters)
    print("Shape of masks: ", masks.shape)
    length, width = image.shape
    tex = masks.reshape(((length*width),masks.shape[2]))
    print('After reshape: ', tex.shape)
    kmeans = sklearn.cluster.KMeans(n_clusters=clusters, n_init='auto')
    kmeans.fit(tex)
    map = kmeans.predict(tex)
    print("MAP DONE")
    map = np.reshape(map, (length, width))
    return map

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
    plot_subplot_images(LMS_bank, 'Phase1/LM_Filters/LMS.png', 12, 4, (12, 4))
    
    LML_bank = generate_LM_bank(49, 6, DoG_sigmas=[np.sqrt(2), 2, 2 * np.sqrt(2)],
                                LoG_sigmas=[2, 2*np.sqrt(2), 2 * np.sqrt(3), 4],
                                Gauss_sigmas=2 * np.array([np.sqrt(2), 2, 2 * np.sqrt(2), 4]))
    plot_subplot_images(LML_bank, 'Phase1/LM_Filters/LML.png', 12, 4, (12, 4))

    """
    Generate Gabor Filter Bank: (Gabor)
    Display all the filters in this filter bank and save image as Gabor.png,
    use command "cv2.imwrite(...)"
    """
    gabor_bank = generate_gabor_filters(49, 40, [3,5,7], theta = 0.25, Lambda = 1, psi = 1, gamma = 1)
    plot_subplot_images(gabor_bank, 'Phase1/Gabor_Filters/GB.png', nrows=5, ncols=8, figsize=(7, 7))
    
    """
    Generate Half-disk masks
    Display all the Half-disk masks and save image as HDMasks.png,
    use command "cv2.imwrite(...)"
    """
    halfcircle_bank = generate_halfcircle_filters(size=[10, 30, 50], n_orientations=5, radius=[3, 10, 14])
    for n, img in enumerate(halfcircle_bank):
        save_image(img, "Phase1/HalfCircle_Filters", f'HCM{n}.png')

    image = cv2.imread('Phase1/BSDS500/Images/1.jpg', cv2.IMREAD_GRAYSCALE)
    # plot_image(image, cmap='gray')
    """
    Generate Texton Map
    Filter image using oriented gaussian filter bank
    """
    # filter_bank = DoG_Bank + LMS_bank + LML_bank + gabor_bank

    filter_bank = DoG_Bank + LMS_bank + gabor_bank
    masks = convolve_image(image, filters=filter_bank)
    np.save('filter_bank.npy', np.array(filter_bank, dtype=object), allow_pickle=True)
    

    tg = generate_texton_map(image, filter_bank, 64)
    tg = 3 * tg
    plot_image(tg, cmap='gist_rainbow')

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
 



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

def generate_filter_bank():
    orientations = 16
    sigmas = [1, 2]
    sizes = [17, 17]
    # Generate Derivative of Gaussian(DoG) kernels
    DoG_Bank = generate_DoG_bank(orientations, sigmas, sizes)
    ## Save images
    # for n, img in enumerate(DoG_Bank):
    #     save_image(img, "Phase1/DoG_Filters", f'DoG{n}.png')

    # Generate Leung-Malik Filter Bank: (LM)
    LMS_bank = generate_LM_bank(49, 6, DoG_sigmas=[1, np.sqrt(2), 2],
                                LoG_sigmas=[1, np.sqrt(2), np.sqrt(3), 2],
                                Gauss_sigmas=[np.sqrt(2), 2, 2 * np.sqrt(2), 4])
    ## Save images
    # plot_subplot_images(LMS_bank, 'Phase1/LM_Filters/LMS.png', 12, 4, (12, 4))
    
    # LML_bank = generate_LM_bank(49, 6, DoG_sigmas=[np.sqrt(2), 2, 2 * np.sqrt(2)],
    #                             LoG_sigmas=[2, 2*np.sqrt(2), 2 * np.sqrt(3), 4],
    #                             Gauss_sigmas=2 * np.array([np.sqrt(2), 2, 2 * np.sqrt(2), 4]))
    ## Save images
    # plot_subplot_images(LML_bank, 'Phase1/LM_Filters/LML.png', 12, 4, (12, 4))

    # Generate Gabor Filter Bank: (Gabor)
    gabor_bank = generate_gabor_filters(49, 40, [3,5,7], theta = 0.25, Lambda = 1, psi = 1, gamma = 1)
    ## Save images
    # plot_subplot_images(gabor_bank, 'Phase1/Gabor_Filters/GB.png', nrows=5, ncols=8, figsize=(7, 7))

    filter_bank = DoG_Bank + LMS_bank + gabor_bank
    return filter_bank


def generate_halfcircle_filters(scales:list):
    half_discs = []
    angles = [0, 180, 30, 210, 45, 225, 60, 240, 90, 270, 120, 300, 135, 315, 150, 330]           #rotation angles (not equally spaced)
    no_of_disc = len(angles)
    for radius in scales:
        kernel_size = 2*radius + 1
        cc = radius
        kernel = np.zeros([kernel_size, kernel_size])
        for i in range(radius):
            for j in range(kernel_size):
                a = (i-cc)**2 + (j-cc)**2                                     #to create one disc
                if a <= radius**2:
                    kernel[i,j] = 1
        
        for i in range(0, no_of_disc):                                       #rotate to make other discs
            mask = rotate_image(kernel, angles[i])
            mask[mask<=0.5] = 0
            mask[mask>0.5] = 1
            half_discs.append(mask)
    return half_discs

def generate_texton_map(image:np.array, filters:np.array, clusters:int):
    length, width = image.shape
    tex = filters.reshape(((length*width), filters.shape[2]))
    kmeans = sklearn.cluster.KMeans(n_clusters=clusters, n_init='auto')
    kmeans.fit(tex)
    map = kmeans.predict(tex)
    map = np.reshape(map, (length, width))
    return map

def generate_brightness_map(image:np.array, clusters:int):
    length, width = image.shape
    bg = image.reshape((length*width, 1))
    kmeans = sklearn.cluster.KMeans(n_clusters=clusters, n_init='auto')
    kmeans.fit(bg)
    map = kmeans.predict(bg)
    return map.reshape((length, width))

def generate_color_map(image:np.array, clusters:int):
    length, width, depth = image.shape
    cg =  image.reshape((length*width, depth))
    kmeans = sklearn.cluster.KMeans(n_clusters=clusters, n_init='auto')
    kmeans.fit(cg)
    map = kmeans.predict(cg)
    return map.reshape((length, width))
    
def chi_square_dist(map, n_bins, left_mask, right_mask):
    chi_sqr_dist = np.zeros(map.shape)
    for i in range(0, n_bins):
        tmp = np.zeros(map.shape)
        tmp[map == i] = 1
        gi = cv2.filter2D(tmp, -1, kernel=left_mask)
        hi = cv2.filter2D(tmp, -1, kernel=right_mask)
        
        chi_sqr_dist += ((gi - hi)**2 / (gi + hi + 0.0001))
    return chi_sqr_dist/2

def gradient(map, n_bins, hdmasks):
    i = 0
    gradient = np.array(map)
    while i < len(hdmasks) - 1:
        g = chi_square_dist(map, n_bins, hdmasks[i], hdmasks[i+1])
        gradient = np.dstack((gradient, g))
        i += 2
    return np.mean(gradient, axis=2)

def main():
    halfcircle_bank = generate_halfcircle_filters(scales=[3, 10, 14])
    for n, img in enumerate(halfcircle_bank):
        save_image(img, "Phase1/HalfCircle_Filters", f'HCM{n}.png')

    # i = 2
    for i in range(1, 11):
        image = cv2.imread(f'Phase1/BSDS500/Images/{i}.jpg', cv2.IMREAD_COLOR)
        gray_image = cv2.imread(f'Phase1/BSDS500/Images/{i}.jpg', cv2.IMREAD_GRAYSCALE)

        if not os.path.isfile(f'Phase1/data/masks{i}.npy'):
            filter_bank = generate_filter_bank()
            masks = convolve_image(gray_image, filters=filter_bank)
            np.save(f'Phase1/data/masks{i}.npy', masks, allow_pickle=True)
        elif os.path.isfile(f'Phase1/data/masks{i}.npy'):
            masks = np.load(f'Phase1/data/masks{i}.npy', allow_pickle=True)

        t_map = generate_texton_map(gray_image, masks, 48)
        t_map = 3 * t_map

        b_map = generate_brightness_map(gray_image, 16) 

        c_map = generate_color_map(image, 16)

        tg = gradient(t_map, 64, halfcircle_bank)
        save_image(tg, 'Phase1/results', f'texture_gradient{i}.png')
        bg = gradient(b_map, 8, halfcircle_bank)
        save_image(bg, 'Phase1/results', f'brightness_gradient{i}.png')
        cg = gradient(c_map, 8, halfcircle_bank)
        save_image(cg, 'Phase1/results', f'color_gradient{i}.png')

        sobel_baseline = cv2.imread(f'Phase1/BSDS500/SobelBaseline/{i}.png', cv2.IMREAD_GRAYSCALE)
        canny_baseline = cv2.imread(f'Phase1/BSDS500/CannyBaseline/{i}.png', cv2.IMREAD_GRAYSCALE)

        output = np.multiply((tg + bg + cg)/3, (0.5 * canny_baseline + 0.5 * sobel_baseline)) 
        save_image(output, 'Phase1/results/', f'output{i}.png')
        print(f'Output generated for image {i}')    

if __name__ == '__main__':
    main()
 


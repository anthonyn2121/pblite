import numpy as np

def generate_gaussian_kernel(size:int, sigma:int) -> np.array:
    '''! Generates a 2D Gaussian kernel or mask of NxN size 

    @param size     Size of the kernel 
    @param sigma  Standard deviation used in the x-direction

    @return image mask
    '''
    center_coord = (size - 1) / 2 ## coordinates used to center the kernel

    ## There is an extra subtraction step here used to center the kernel
    ## '((x - (size - 1)/2)**2 + (y - (size - 1)/2)**2)'
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - center_coord)**2 + (y - center_coord)**2) / (2 * sigma**2)),
        (size, size)
    )

    return kernel / np.sum(kernel)  ## normalized 

def gaussian_derivative(size:int, sigma:int, order:int, axis:int=0) -> np.array:
    '''! Generates 1st or 2nd derivative of the gaussian along a specific axis
    
    @param size     Size of kernel
    @param sigma    Standard deviation value
    @param order    Order of derivative to take

    @return 1-directional DoG
    '''
    x = np.arange(-size // 2, size // 2)
    G = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x**2 / (2 * sigma**2)))
    if order == 1:
        G = -x / sigma**2 * G
    elif order == 2:
        G = (x**2 - sigma**2) * G / sigma**4 

    G /= np.abs(G).sum()

    if axis == 0:
        return G
    if axis == 1:
        return G.reshape(-1, 1)
    
def laplacian_of_gaussian(size:int, sigma:int) -> np.array:
    '''! Generates the Laplacian of Gaussian (LoG) filter

    @param size     Size of the kernel 
    @param sigma  Standard deviation used in the x-direction

    @return image mask
    '''
    kernel = np.fromfunction(
        lambda x, y: (x**2 + y**2 - 2 * sigma**2) * np.exp(-(x**2 + y**2) / (2 * sigma**2)),
        (size, size)
    )

    kernel /= (2 * np.pi * sigma**6)
    return kernel
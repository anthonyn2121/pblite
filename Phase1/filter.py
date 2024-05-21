import numpy as np
from utils.image_utils import convolve2d

def gaussian1d(x:np.array, sigma:int, order:int) -> np.array:
    ''' Generates 1st or 2nd derivative of the gaussian
    
    @param size     Size of kernel
    @param sigma    Standard deviation value
    @param order    Order of derivative to take

    @return 1-directional DoG
    '''
    G = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x**2 / (2 * sigma**2)))
    if order == 1:
        G = -x / sigma**2 * G
    elif order == 2:
        G = (x**2 - sigma**2) * G / sigma**4 
    return G
    
def gaussian2d(size:int, sigma:int) -> np.array:
    ''' Generates a 2D Gaussian kernel or mask of NxN size where 
        the sigma value of x and y are the same

    @param size     Size of the kernel 
    @param sigma    Standard deviation used in the x-direction

    @return image mask
    '''
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(-((x)**2 + (y)**2) / (2 * sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)  ## normalized 

def gaussian2d(size:int, sigma_x:int, sigma_y:int, dx:int, dy:int) -> np.array:
    ''' Generates a 2D Guassian kernel or mask of NxN size where
        the sigma value of x and y are different and/or of higher order of derivative 
    
    @param size     Size of the kernel
    @param sigma_x  Sigma value of the x-direction gaussian
    @param sigma_y  Sigma value of the y-direction gaussian
    @param order    Order of derivative to take

    @return image mask
    '''
    x, y = np.meshgrid(np.linspace(-size//2, size//2, size),
                       np.linspace(-size//2, size//2, size))
    x = x.flatten()
    y = y.flatten()
    gx = gaussian1d(x, sigma_x, dx)
    gy = gaussian1d(y, sigma_y, dy)
    g = gx * gy
    return np.reshape(g, (size, size))

def laplacian_of_gaussian(size:int, sigma:int) -> np.array:
    ''' Generates the Laplacian of Gaussian (LoG) filter

    @param size   Size of the kernel 
    @param sigma  Standard deviation used in the x-direction

    @return image mask
    '''
    kernel = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])
    
    g = gaussian2d(size, sigma, sigma, 0, 0)
    log = convolve2d(g, kernel)
    return log
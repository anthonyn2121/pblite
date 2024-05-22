import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import scipy.ndimage

def plot_image(image, title=None, cmap=None) : 
    ''' Plot an image with matplotlib
    
    @param image    An nd.array or PIL image
    @param title(optional)    Title of the image
    '''
    if isinstance(image, np.ndarray):
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(np.asarray(image), cmap=cmap)

    if title:
        plt.title(title)
    plt.show()

def plot_subplot_images(images:list, title:str, nrows:int, ncols:int, figsize:tuple=(10, 10)):
    """
    Display a list of images in a subplot.

    Parameters:
    - images: List of np.array, each array representing an image.
    - cols: Number of columns in the subplot grid.
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(title)
    plt.close()

def save_image(image, directory, path):
    if not os.path.exists(directory):
        os.makedirs(directory)

    new_path = os.path.join(directory, path)

    matplotlib.image.imsave(new_path, image, cmap='gray')

def rotate_image(image, angle):
    return scipy.ndimage.rotate(image, angle, reshape=False)

def sobel_x() -> np.array:
    '''! Returns the sobel filter that when conlved with an image, is used to calculate the approximate
         derivates of the image in the x-direction
    
    @return Sobel filter in x-direction, Gx
    '''

    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])

def sobel_y() -> np.array:
    '''! Returns the sobel filter that when conlved with an image, is used to calculate the approximate
         derivates of the image in the y-direction
    
    @return Sobel filter in y-direction, Gy
    '''

    return np.array([[1, 2, 1],
                     [0, 0, 0],
                     [-1, -2, -1]])

def convolve2d(image:np.array, filter:np.array) -> np.array:
    '''! Convolution operator on an image using a filter. Pads kernel with 0s to preserve image size
    
    @param image    A 2D image 
    @param filter

    @return Convolved image
    '''
    img_height, img_width = image.shape
    filter_height, filter_width = filter.shape

    pad_height = filter_height // 2 
    pad_width = filter_width // 2

    padded_image = np.pad(image, ((pad_height, ), (pad_width, )), mode='constant')

    output = np.zeros_like(image)
    for i in range(img_height):
        for j in range(img_width):
            region = padded_image[i:(i + filter_height), j:(j + filter_width)]
            output[i, j] = np.sum(region * filter)
    
    return output


# A Simplified Reimplementation of "Contour Detection and Hierarchical Image Segmentation"

Boundary detection is an important, well-studied computer vision problem. Classical edge detection algorithms 
such as Sobel or Canny detection algorithms identifies where the image brightness changes sharply or experiences
discontinuities. With this method, we also consider texture, color, and brightness discontinuities in addition to intensity
continuities to find a per-pixel probability of a boundary. Qualitatively, this should supress the "false positives" where 
the classical methods produce in textured regions.


## Method
1. Creating a filter bank composed of:
    - Derivative of Gaussian(DoG) filters
    - Leung-Malik filters 
    - Gabor Filters
2. Creating a texton, brightness, color map of the image through KMeans Clustering
3. Calculating the gradients of the 3 previous maps by convolving the maps with oriented half-discs and calculating the 
   chi-squared difference between histograms computed from the binary half-discs
4. Combine the information from the features with a baseline method (either the Sobel, Canny, or both outputs)


## Results
The pblite edge-detection output is better when compared to the sobel or canny edge detection algorithm outputs, although there are still *some* "false positive" artifacts left in the image. This can be further improved by taking a larger weight on the canny output than the sobel output. 
import cv2
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Creates background mask or background image depending on what is desired
def create_background(fg_mask, img, color = False):
    """
    Inverts the foreground mask to create the background image.

    Parameters
    ----------
    color : bool, optional
        If set to true, the function will return the background in color; otherwise it will be a binary image.
    fg_mask : Numpy array
        Binary image containing white bounding boxes around foreground objects.
    img : Numpy array
        Original image of the foreground mask.

    Returns
    -------
    bg : Numpy array
        Background image in color or as binary image.
    
    Notes
    -----
    create_background on its own does not decide when it is best to update the background. Using this function without deciding when to update the background will negate background subtraction.
    """
    bg = cv2.bitwise_not(fg_mask)
    if color is True:
        bg = cv2.bitwise_and(img, img, mask=bg)     # use this line if color is desired over binary mask
    return bg

# expands bounding box to include background pixels for better feature detection
def expand_bounding_box(ystart, xstart, ystop, xstop, img_xsize, img_ysize, advancement = 150):
    """
    expand_bounding_box takes bounding boxes and expands them in all directions by the desired amount. 
    
    Parameters
    ----------
    ystart : int
        The top left most y coorindate for the bounding box.
    xstart : int
        The top left most x coordinate for the bounding box.
    ystop : int
        The bottom right most y coordinate for the bounding box.
    xstop : int
        The bottom right most x coordinate for the bounding box.
    img_xsize : int
        Image Width.
    img_ysize : int
        Image Height.
    advancement : int
        How many pixels to expand the bounding box by.

    Returns
    -------
    ystart : int
        New top left most y coordinate.
    xstart : int
        New top left most x coordinate.
    ystop : int
        New bottom right most x coordinate.
    xstop : int
        New bottom right most x coordinate.
    Notes
    -----
    This allows the foreground mask to include some background pixles in addition to the foreground. This allows for better feature detection when using openMVG.
    
    """
    ystart = max(0, ystart-advancement)
    xstart = max(0, xstart-advancement)
    ystop = min(ystop+advancement, img_ysize)
    xstop = min(xstop+advancement, img_xsize)
    return ystart, xstart, ystop, xstop

def kmeans_helper(image, advancement, grid_width = 45):
    """
    Cluster nonzero pixels and return bounding boxes around each cluster.

    Parameters
    ----------
    image : Numpy array
        Binary image from frame subtraction.
    grid_width : int 
        Determines the size of the grid squares to be used for grid search algorithm. Higher value = less time, less accuracy. Lower value = more time, more accuracy.

    Returns
    -------
    success : bool
        True, if clustering was successful.
    foreground_mask : Numpy array
        Binary image containing bounding boxes around foreground clusters. If success = false, this will be type None.
    Notes
    -----
    To reduce computational complexity, a grid search algorithm is used before kmeans clustering to reduce the amount of points. From here the points from grid search are used for clustering. Then bounding boxes are drawn around each of those clusters. This comes with the trade off of potentially losing some accuracy.
    """
    #initialize success
    success = True
    
    #find grid bondary points
    data = np.nonzero(image)
    topLeft = ( data[1].min(), data[0].min() )
    bottomRight = ( data[1].max(), data[0].max() ) 

    #creating the grid
    #grid_width = 45     #width of each grid square
    centroids = []      #centroid for each grid
    end_x = bottomRight[0] - topLeft[0]
    end_y = bottomRight[1] - topLeft[1]
    start_grid = time.time()

    #height, width for image array
    ys_to_search = np.linspace(0,end_y,int(end_y/grid_width),dtype=int)
    xs_to_search = np.linspace(0,end_x,int(end_x/grid_width),dtype=int)
    for i in range(len(xs_to_search)-1):
        for j in range(len(ys_to_search)-1):
            #obtain new grid slice
            grid = image[(ys_to_search[j] + topLeft[1]):(ys_to_search[j+1] + topLeft[1]), (xs_to_search[i] + topLeft[0]):(xs_to_search[i+1] 
                                                                                                                          + topLeft[0])]
            
            #find non-zero points
            nonzero_xs, nonzero_ys = np.nonzero(grid)
            
            #need to bring nonzero pixels into image coordinates
            nonzero_xs += xs_to_search[i] + topLeft[0]
            nonzero_ys += ys_to_search[j] + topLeft[1]
            if len(nonzero_ys) > 1 or len(nonzero_xs) > 1:
                x_avg = np.mean(nonzero_xs)
                y_avg = np.mean(nonzero_ys)
                centroids.append([x_avg,y_avg])
                
    # print("Size of grids: ", grid_width, "x", grid_width )
    # print("Total Number of centroids after grid algorithm: ",  len(centroids))
    # print("Time taken to form grid and calculate centroids: ", time.time() - start_grid)
    centroids = np.asarray(centroids).reshape(-1,2)

    #calculate centroids
    try:
        #run silhouette algorithm (2 to 11 clusters)
        best_labels = None
        best_sil_score = -1
        start_Ctime = time.time() 
        for k in range(2,11):
            kmeans = KMeans(n_clusters=k, init='k-means++')
            kmeans.fit(centroids)
            silhouette_avg = silhouette_score(centroids, kmeans.labels_)
            if silhouette_avg > best_sil_score:
                best_centroids = kmeans.cluster_centers_
                best_labels = kmeans.labels_
                best_sil_score = silhouette_avg
        # print("Time taken to run ", k, " kmeans: ", time.time() - start_Ctime)
        clusters = [centroids[best_labels == label] for label in np.unique(best_labels)]
        # print('Number of clusters is: ', len(clusters))

        # establish bounding box for each cluster
        img = np.zeros((image.shape[0],image.shape[1]), np.uint8)
        for c in clusters:
            # find bounding values of cluster (min values)
            (xstart, ystart), (xstop, ystop) = c.min(0), c.max(0) 
            # expand bounding box to include feature points (do not exceed image bounds)
            ystart, xstart, ystop, xstop = expand_bounding_box(int(ystart), int(xstart), int(ystop), int(xstop), image.shape[1], image.shape[0], advancement = advancement)
            foreground_mask = cv2.rectangle(img, (xstart, ystart), (xstop, ystop), color = (255,255,255), thickness=-1)

    except:
        success = False
        foreground_mask = None
        
    return success, foreground_mask


def create_fg_mask(fg_binary, image, advancement, color=False):
    """
    Create foreground image using foreground binary image from background subtraction.

    Parameters
    ----------
    fg_binary : Numpy array
        Binary image obtained from frame subtraction.
    image : Numpy array
        Original image of foreground binary.
    color : bool, optional
        If set to true, the function will return the foreground in color; otherwise it will be a binary image.

    Returns
    -------
    fg_mask : Numpy array
        Foreground image as either binary image or color image.

    """
    success, fg = kmeans_helper(fg_binary, advancement)
    if color is True:
        fg = cv2.bitwise_and(image, image, mask=fg)
    return fg


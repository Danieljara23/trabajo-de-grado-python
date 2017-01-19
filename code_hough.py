# **********************************************************************
# Algoritmo para clasificacion de esquejes- Daniel Jaramillo Grisales
#             Facultad de Ingenieria Electronica
#                 Universidad de Antioquia
# ***********************************************************************

# ****************Libraries needed***************************************
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from skimage.morphology import erosion, dilation
from skimage.morphology import rectangle, disk
from skimage import morphology
from skimage.transform import (hough_line, hough_line_peaks,probabilistic_hough_line)
from skimage.feature import canny
from skimage import measure
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
from skimage.filter import sobel_h, sobel_v
from skimage.color import rgb2gray
from scipy import ndimage
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.measure import regionprops
import scipy.misc

path=r"G:\Nueva carpeta\Baltica_01_13_2017_C6_5L8_5H0_9"
list_all=os.listdir(path)

def main():
    # Starting system
    loop_flag = True

    if loop_flag == True:
        for fl in list_all:
            if fl.endswith(".TIFF"):
                print '______________________________________________________'
                print fl
                image = cv2.imread(path + "\\" + fl)
                gray_image = rgb2gray(image)
                thresh_mask = segmentation(image)
                plt.imshow(thresh_mask)
                plt.title('Esta es la imagen binarizada')
                plt.show()

                stem_binary = get_stem(thresh_mask)
                degrees = get_orientation(stem_binary,thresh_mask)
                # hough(image, thresh_mask)
                # find_corners(thresh_mask)
                # gradient_function(gray_image)


def segmentation(img):
    ret, thresh_mask = cv2.threshold(img[:, :, 0], 50, 255, cv2.THRESH_BINARY_INV)
    im2, contours, hierarchy = cv2.findContours(thresh_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print ("contours")
    # print(contours)
    if not contours:
        print ("Contours does not find anything")
    else:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # cnt = contours[0]
        thresh_mask[...] = 0
        cv2.drawContours(thresh_mask, [contours[0]], 0, 255, cv2.FILLED)
        # Esqueje binarizado hasta este punto
        # plt.imshow(thresh_mask)
        # plt.title('Esta es la imagen binarizada')
        # plt.show()

        thresh_mask = erosion(thresh_mask, rectangle(18, 1))
        thresh_mask = dilation(thresh_mask, rectangle(18, 1))
        thresh_mask = erosion(thresh_mask, rectangle(1, 18))
        thresh_mask = dilation(thresh_mask, rectangle(1, 18))

        # plt.imshow(thresh_mask)
        # plt.title('Esta es la imagen binarizada despues de dilatacion-erosion')
        # plt.show()

        return thresh_mask

def get_stem(thresh_mask):
    bw = np.asarray(thresh_mask)

    bw[bw == 1] = 255
    bw_a = 1 - bw
    # Se aplica la funcion distancia
    h_2 = ndimage.distance_transform_edt(1 - bw_a)
    print(h_2)

    h_2[h_2 < 20] = 0
    h_2[h_2 > 0] = 255
    plt.imshow(h_2)
    plt.title('after 0')
    plt.show()


    h_2 = cv2.convertScaleAbs(h_2)
    h_2 = area_filter(h_2)

    g = dilation(h_2, disk(15))
    g = dilation(g, disk(15))
    plt.imshow(g,'gray')
    plt.title('g')
    plt.show()

    h = bw_a
    h[g>5.5] = 0
    h[h != 2] =0
    plt.imshow(h,'gray')
    plt.title('h')
    plt.show()

    #-----------now i get stem and maybe some leafs----------------
    b = h
    b = area_filter(b)

    plt.imshow(b, 'gray')
    plt.title('b')
    plt.show()
    # bw_b = 1 - b
    # h_2 = ndimage.distance_transform_edt(1 - bw_b)
    return b

def get_orientation(stem, binary_cut):
    bw = np.asarray(stem)

    bw[bw == 1] = 255
    bw_a = 1 - bw
    # Se aplica la funcion distancia
    h_2 = ndimage.distance_transform_edt(1 - bw_a)
    plt.imshow(h_2)
    plt.title('h-orientation')
    plt.show()
    # print(h_2)

    h_2[h_2 < 12] = 0
    h_2[h_2 > 0] = 255
    # plt.imshow(h_2)
    # plt.title('almost thin')
    # plt.show()

    h_2 = cv2.convertScaleAbs(h_2)
    h_2 = area_filter(h_2)
    h_2[h_2 != 0] = 1
    thin_line = morphology.skeletonize(h_2)

    # plt.imshow(thin_line)
    # plt.title('thin')
    # plt.show()

    # --
    ind = h_2 > 0
    ind_2 = ind.shape[1]

    if ind_2 > 6:
        for i in range(1,2):
            thin_line[ind[i]] = 0

        for i in range(ind_2 -1, ind_2 - 2):
            thin_line[ind[i]] = 0

    plt.imshow(thin_line)
    plt.title('thin-post')
    plt.show()

    print (ind)

    label_img = label(thin_line, connectivity=thin_line.ndim)
    props = regionprops(label_img)
    # centroid of first labeled object
    stem_orientation = np.rad2deg(props[0].orientation)
    print("The orientation is:")
    print(stem_orientation)

    stem_rotated = scipy.misc.imrotate(stem,-stem_orientation)
    plt.imshow(stem_rotated)
    plt.title('Stem rotated')
    plt.show()

    final_binary_image_rotated = scipy.misc.imrotate(binary_cut, -stem_orientation)
    plt.imshow(final_binary_image_rotated)
    plt.title('Esqueje rotado')
    plt.show()

    return h_2


def area_filter(b):
    b_ret = b
    # b_ret = b_ret.reshape(b_ret.shape[0], b_ret.shape[1])
    im_x, contours, hierarchy = cv2.findContours(b_ret, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print ("Contours does not find anything")
        b_ret[:,:] = 0
    else:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        cnt = contours[0]
        b_ret[...] = 0
        cv2.drawContours(b_ret, [cnt], 0, 255, cv2.FILLED)

    return b_ret

def show_images(images,titles=None):
    """Display a list of images"""
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n) # Make subplot
        if image.ndim == 2: # Is image grayscale?
            plt.gray() # Only place in this blog you can't replace 'gray' with 'grey'
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()



if __name__ == '__main__':
    main()

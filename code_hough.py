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
from skimage.color import rgb2gray
from scipy import ndimage
from skimage.measure import label
from skimage.measure import regionprops
import scipy.misc
import math
import skimage.io
from mpl_toolkits.axes_grid import AxesGrid
from skimage.io import MultiImage

path=r"G:\Nueva carpeta\Baltica_01_13_2017_C6_5L8_5H0_9"
list_all=os.listdir(path)

def main():
    # Starting system
    loop_flag = False
    small_cm = 8
    large_cm = 9
    hoja_base_cm = 1
    #small_px, large_px, hojabase_px = convert_umbrals(small_cm, large_cm, hoja_base_cm)


    if loop_flag == True:
        for fl in list_all:
            if fl.endswith(".TIFF"):
                print '______________________________________________________'
                print fl
                image = cv2.imread(path + "\\" + fl)
                gray_image = rgb2gray(image)
                thresh_mask = segmentation(image)

                #plt.imshow(thresh_mask)
                #plt.title('Esta es la imagen binarizada')
                #plt.show()

                stem_binary = get_stem(thresh_mask)
                degrees = get_orientation(stem_binary,thresh_mask)
                correct_stem = rotate(degrees, thresh_mask, stem_binary)
                correct_stem, size= well_position(correct_stem)
                x_1, y_1, x_2, y_2 = get_h_position(correct_stem,fl)
                hoja_base = is_hoja_base(x_1,y_1, x_2, y_2, hoja_base_cm)
                category = clasification(small_cm, large_cm, size,hoja_base)
    else:
        print '______________________________________________________'
        fl = "Foto_1118_clase_2.TIFF"
        print fl
        image = cv2.imread(path + "\\" + fl)
        gray_image = rgb2gray(image)
        thresh_mask = segmentation(image)

        plt.imshow(thresh_mask)
        plt.title('Esta es la imagen binarizada')
        plt.show()

        stem_binary = get_stem(thresh_mask)
        degrees = get_orientation(stem_binary, thresh_mask)
        correct_stem = rotate(degrees, thresh_mask, stem_binary)
        correct_stem, size = well_position(correct_stem)
        x_1, y_1, x_2, y_2 = get_h_position(correct_stem, fl)
        hoja_base = is_hoja_base(x_1, y_1, x_2, y_2, hoja_base_cm)
        category = clasification(small_cm, large_cm, size, hoja_base)

def convert_umbrals(small_cm, large_cm, hoja_base_cm):


    factor = 11.5 / 960  # ************Factor px to cm
    small_px = small_cm / factor
    large_px = large_cm / factor
    hojabase_px = hoja_base_cm / factor

    return small_px, large_px, hojabase_px


def color_spaces(r,g,b):
    cmyk_scale = 100

    if (r == 0) and (g == 0) and (b == 0):
        # black
        return 0, 0, 0, cmyk_scale

    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / 255.
    m = 1 - g / 255.
    y = 1 - b / 255.

    # extract out k [0,1]
    min_cmy = min(c, m, y)
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy

    # rescale to the range [0,cmyk_scale]
    return c*cmyk_scale, m*cmyk_scale, y*cmyk_scale, k*cmyk_scale


def segmentation(img):
    ret, thresh_mask = cv2.threshold(img[:, :, 0], 50, 255, cv2.THRESH_BINARY_INV)
    im2, contours, hierarchy = cv2.findContours(thresh_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print ("contours")
    if not contours:
        print ("Contours does not find anything")
    else:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        thresh_mask[...] = 0
        cv2.drawContours(thresh_mask, [contours[0]], 0, 255, cv2.FILLED)
        # Esqueje binarizado hasta este punto
        #plt.imshow(thresh_mask)
        #plt.title('Esta es la imagen binarizada')
        #plt.show()

        thresh_mask = erosion(thresh_mask, rectangle(18, 1))
        thresh_mask = dilation(thresh_mask, rectangle(18, 1))
        thresh_mask = erosion(thresh_mask, rectangle(1, 18))
        thresh_mask = dilation(thresh_mask, rectangle(1, 18))

        # plt.imshow(thresh_mask)
        # plt.title('Esta es la imagen binarizada despues de dilatacion-erosion')
        # plt.show()

        return thresh_mask

def is_hoja_base(x_1, y_1, x_2, y_2, hoja_base_cm):
    factor = 11.5 / 960
    distance = math.sqrt(math.pow(x_2 - x_1,2) + math.pow(y_2 - y_1,2))
    distance_cm = factor * distance
    print ("La hoja esta a: "+str(distance_cm)+"cm")
    if distance_cm < hoja_base_cm:
        print ("El esqueje presenta una hoja en base")
        hoja_base = 1
    else:
        print ("No presenta hoja en base")
        hoja_base = 0

    return hoja_base


def get_stem(thresh_mask):
    bw = np.asarray(thresh_mask)

    bw[bw == 1] = 255
    bw_a = 1 - bw
    # Se aplica la funcion distancia
    h_2 = ndimage.distance_transform_edt(1 - bw_a)
    #print(h_2)
    #plt.imshow(h_2)
    #plt.title('h_2')
    #plt.show()


    h_2[h_2 < 20] = 0
    h_2[h_2 > 0] = 255
    #plt.imshow(h_2)
    #plt.title('after 0')
    #plt.show()


    h_2 = cv2.convertScaleAbs(h_2)
    h_2 = area_filter(h_2)

    g = dilation(h_2, disk(15))
    g = dilation(g, disk(15))
    #plt.imshow(g,'gray')
    #plt.title('g')
    #plt.show()

    h = bw_a
    h[g>5.5] = 0
    h[h != 2] =0
    #plt.imshow(h,'gray')
    #plt.title('h')
    #plt.show()

    #-----------now i get stem and maybe some leafs----------------
    b = h
    b = area_filter(b)

    #plt.imshow(b, 'gray')
    #plt.title('b')
    #plt.show()
    return b

def get_orientation(stem, binary_cut):
    bw = np.asarray(stem)

    bw[bw == 1] = 255
    bw_a = 1 - bw
    # Se aplica la funcion distancia
    h_2 = ndimage.distance_transform_edt(1 - bw_a)
    #plt.imshow(h_2)
    #plt.title('h-orientation')
    #plt.show()
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

    #plt.imshow(thin_line)
    #plt.title('thin-post')
    #plt.show()

    #print (ind)

    label_img = label(thin_line, connectivity=thin_line.ndim)
    props = regionprops(label_img)
    # centroid of first labeled object
    stem_orientation = np.rad2deg(props[0].orientation)
    #print("The orientation is:")
    #print(stem_orientation)


    return stem_orientation

def well_position(bw_esqueje):
    bw_esqueje_aux = bw_esqueje
    bw_esqueje_aux[bw_esqueje_aux > 0] = 1
    label_img = label(bw_esqueje_aux, connectivity=bw_esqueje_aux.ndim)
    props = regionprops(label_img)


    bw_sliced = props[0].image[:,:]
    size = bw_sliced.shape
    size = size[1]
    #print ("---------")
    #print(size)
    #print ("---------")
    #print(size)
    #plt.imshow(bw_sliced)
    #plt.title('Sliced')
    #plt.show()

    rows, cols = bw_sliced.shape
    a = 0
    # ***************************Get stem's side***********************
    for c in range(cols):
        a = np.append(a, (bw_sliced[:, c] > 100).sum())
    part20 = cols * 0.2

    image20 = bw_sliced[:,:int(part20)]
    image80 = bw_sliced[:,-int(part20):]

    image20 = np.array(image20)
    image80 = np.array(image80)

    size_20 = np.sum(image20,axis=0)
    size_80 = np.sum(image80, axis=0)
    size_20 = size_20.max()
    size_80 = size_80.max()
    #print (size_20)
    #print (size_80)

    if size_20 > size_80:
        print ("Tallo a la derecha")
        bw_esqueje = cv2.flip(bw_esqueje, 1)
    elif size_80 > size_20:
        print ("Tallo a la izquierda")


    #plt.imshow(bw_esqueje)
    #plt.title('Esqueje bien posicionado')
    #plt.show()
    return bw_esqueje, size


def rotate(stem_orientation, binary_cut, stem):

    stem_rotated = scipy.misc.imrotate(stem,-stem_orientation)
    #plt.imshow(stem_rotated,'gray')
    #plt.title('Stem rotated')
    #plt.show()

    final_binary_image_rotated = scipy.misc.imrotate(binary_cut, -stem_orientation)
    #plt.imshow(final_binary_image_rotated,'gray')
    #plt.title('Esqueje rotado')
    #plt.show()

    return final_binary_image_rotated

def get_h_position(binary_image,fl):
    bw = np.asarray(binary_image)
    flip_bw = np.rot90(bw, 1)
    flip_bw = cv2.flip(flip_bw, 0)
    #plt.imshow(flip_bw, 'gray')
    #plt.title('fliped to search first pixel')
    #plt.show()
    (i, j) = flip_bw.nonzero()
    x_1 = i[0]
    y_1 = j[0]

    bw[bw == 1] = 255
    bw_a = 1 - bw
    # Se aplica la funcion distancia
    h_2 = ndimage.distance_transform_edt(1 - bw_a)
    #print(h_2)
    #plt.imshow(h_2)
    #plt.title('h_2 - Esqueje derecho')
    #plt.show()


    h_2[h_2 != 20] = 0
    h_2[h_2 > 0] = 1


    flip_bw_2 = np.rot90(h_2, 1)
    flip_bw_2 = cv2.flip(flip_bw_2, 0)
    #plt.imshow(flip_bw_2, 'gray')
    #plt.title('fliped to search another white pixel')
    #plt.show()
    (i, j) = flip_bw_2.nonzero()
    x_2 = i[0]
    y_2 = j[0]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(bw, cmap=plt.cm.gray)
    ax.plot(x_1, y_1,'ro')
    ax.plot(x_2, y_2, 'ro')
    fig.savefig("Resultados/"+fl[:9]+".TIFF")

    #plt.imshow(bw,'gray')
    #plt.plot(x_1,y_1,'ro')
    #plt.plot(x_2, y_2, 'ro')
    #plt.title('Dots')
    #plt.show()

    return x_1, y_1, x_2, y_2




def area_filter(b):
    b_ret = b
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

def clasification(small_cm, large_cm, size,hoja_base):
    factor = 11.5 / 960
    length_cm = factor * size
    print ("El esqueje mide: " + str(length_cm) + "cm")
    if length_cm < small_cm :
        print ("Corto")
        category = '1'
    elif (length_cm > small_cm and length_cm < large_cm) and (hoja_base != 1):
        print ("Ideal")
        category = 4
    elif length_cm > large_cm and hoja_base == 0:
        print ("Largo")
        category = '2'
    elif hoja_base == 1:
        print ("Hoja en base")
        category = '3'

    return category

if __name__ == '__main__':
    main()

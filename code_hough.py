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
from skimage import morphology
from skimage.color import rgb2gray
from scipy import ndimage
from skimage.measure import label
from skimage.measure import regionprops
import scipy.misc
import math
import matplotlib.image as mpimg

path = r"G:\Nueva carpeta\Baltica_01_13_2017_C6_5L8_5H0_9"
list_all = os.listdir(path)


def main():
    # Starting system
    loop_flag = False
    small_cm = 8
    large_cm = 9
    hoja_base_cm = 1

    if loop_flag:
        for fl in list_all:
            if fl.endswith(".TIFF"):
                print '______________________________________________________'
                print fl
                image = cv2.imread(path + "\\" + fl)
                thresh_mask, presence = segmentation(image)
                if presence == True:

                    stem_binary = get_stem(thresh_mask)
                    degrees = get_orientation(stem_binary)
                    correct_stem = rotate(degrees, thresh_mask)
                    correct_stem, size = well_position(correct_stem)
                    x_1, y_1, x_2, y_2, bw = get_h_position(correct_stem, fl, image)
                    hoja_base = is_hoja_base(x_1, y_1, x_2, y_2, hoja_base_cm)
                    category, length_cm = clasification(small_cm, large_cm, size, hoja_base)
                elif not presence:
                    category = '4'
    else:
        print '______________________________________________________'
        fl = "Foto_1022_clase_3.TIFF"
        print fl
        image = cv2.imread(path + "\\" + fl)
        thresh_mask, presence = segmentation(image)
        print (presence)
        if presence:

            stem_binary = get_stem(thresh_mask)
            degrees = get_orientation(stem_binary)
            correct_stem = rotate(degrees, thresh_mask)
            correct_stem, size = well_position(correct_stem)
            x_1, y_1, x_2, y_2, bw = get_h_position(correct_stem, fl, image)
            hoja_base = is_hoja_base(x_1, y_1, x_2, y_2, hoja_base_cm)
            category , length_cm = clasification(small_cm, large_cm, size, hoja_base)
            plt.imshow(bw, 'gray')
            plt.plot(x_1, y_1, 'ro')
            plt.plot(x_2, y_2, 'ro')
            plt.axis('off')
            plt.text(200, 800, r"Longitud del esqueje=" + str(length_cm)+"cm", color = 'white')
            plt.text(200, 850, r"Clasificacion: "+category, color='white')
            plt.grid(True)
            plt.title('Final image', fontsize=20)
            plt.show()
        elif not presence:
            category = '4'


def convert_umbrals(small_cm, large_cm, hoja_base_cm):  #Function to converto umbrals in cm to px
    factor = 11.5 / 960  # ************Factor px to cm
    small_px = small_cm / factor
    large_px = large_cm / factor
    hojabase_px = hoja_base_cm / factor

    return small_px, large_px, hojabase_px


def segmentation(img):      #Function to segmentate cut
    blur = cv2.GaussianBlur(img, (15, 15), 0)
    image_array = np.asarray(blur)

    blue = image_array[:, :, 0]

    plt.imshow(blue, 'gray')
    plt.show()
    ret, thresh_mask = cv2.threshold(blue, 50, 255, cv2.THRESH_BINARY_INV)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh_mask = cv2.dilate(thresh_mask, se, iterations=2)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh_mask = cv2.erode(thresh_mask, se, iterations=2)

    im2, contours, hierarchy = cv2.findContours(thresh_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print ("contours")
    if not contours:
        print ("Contours does not find anything")
        presence = False
    else:
        presence = True
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        biggest = contours[0]
        area_biggest = cv2.contourArea(biggest)

        if area_biggest < 200:
            presence = False
        else:
            thresh_mask[...] = 0
            cv2.drawContours(thresh_mask, [contours[0]], 0, 255, cv2.FILLED)

            se = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 1))
            thresh_mask = cv2.erode(thresh_mask, se, iterations=1)
            thresh_mask = cv2.dilate(thresh_mask, se, iterations=1)
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 18))
            thresh_mask = cv2.erode(thresh_mask, se, iterations=1)
            thresh_mask = cv2.dilate(thresh_mask, se, iterations=1)



    plt.imshow(thresh_mask,'gray')
    plt.show()

    return thresh_mask, presence


def is_hoja_base(x_1, y_1, x_2, y_2, hoja_base_cm): #Function to define if a cut have a leaf in near to the base
    factor = 11.5 / 960
    distance = math.sqrt(math.pow(x_2 - x_1, 2) + math.pow(y_2 - y_1, 2))
    distance_cm = factor * distance
    print ("La hoja esta a: " + str(distance_cm) + "cm")
    if distance_cm < hoja_base_cm:
        print ("El esqueje presenta una hoja en base")
        hoja_base = 1
    else:
        print ("No presenta hoja en base")
        hoja_base = 0

    return hoja_base


def get_stem(thresh_mask):              #Function to get just the cut's stem
    bw = np.asarray(thresh_mask)

    bw[bw == 1] = 255
    bw_a = 1 - bw
    # Distance Transform
    h_2 = ndimage.distance_transform_edt(1 - bw_a)
    plt.imshow(h_2)
    plt.set_cmap('nipy_spectral')
    plt.colorbar()
    plt.show()

    h_2[h_2 < 20] = 0
    h_2[h_2 > 0] = 255

    h_2 = cv2.convertScaleAbs(h_2)
    h_2 = area_filter(h_2)

    g= cv2.dilate(h_2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)), iterations = 2)
    #kernel = np.ones((15, 15), np.uint8)
    #g = cv2.dilate(h_2, kernel, iterations=4)

    h = bw_a
    h[g > 5.5] = 0
    h[h != 2] = 0

    # -----------now i get stem and maybe some leafs----------------
    b = h
    b = area_filter(b)

    return b


def get_orientation(stem):          #Function to get the stem's orientation
    bw = np.asarray(stem)

    bw[bw == 1] = 255
    bw_a = 1 - bw
    #Distance Transform
    h_2 = ndimage.distance_transform_edt(1 - bw_a)


    h_2[h_2 < 12] = 0
    h_2[h_2 > 0] = 255

    h_2 = cv2.convertScaleAbs(h_2)
    h_2 = area_filter(h_2)
    h_2[h_2 != 0] = 1

    plt.imshow(h_2,'gray')
    plt.show()
    thin_line = morphology.skeletonize(h_2)

    ind = h_2 > 0
    ind_2 = ind.shape[1]

    if ind_2 > 6:
        for i in range(1, 2):
            thin_line[ind[i]] = 0

        for i in range(ind_2 - 1, ind_2 - 2):
            thin_line[ind[i]] = 0

    label_img = label(thin_line, connectivity=thin_line.ndim)
    props = regionprops(label_img)
    # centroid of first labeled object
    if not props:
        stem_orientation = 90
    else:
        stem_orientation = np.rad2deg(props[0].orientation)

    return stem_orientation


def well_position(bw_esqueje):              #Function to properly align the cut
    bw_esqueje_aux = bw_esqueje
    bw_esqueje_aux[bw_esqueje_aux > 0] = 1
    label_img = label(bw_esqueje_aux, connectivity=bw_esqueje_aux.ndim)
    props = regionprops(label_img)

    bw_sliced = props[0].image[:, :]
    size = bw_sliced.shape
    size = size[1]

    rows, cols = bw_sliced.shape
    a = 0
    # ***************************Get stem's side***********************
    for c in range(cols):
        a = np.append(a, (bw_sliced[:, c] > 100).sum())
    part30 = cols * 0.3

    image30 = bw_sliced[:, :int(part30)]
    image70 = bw_sliced[:, -int(part30):]

    image30 = np.array(image30)
    image70 = np.array(image70)

    size_30 = np.sum(image30, axis=0)
    size_70 = np.sum(image70, axis=0)
    size_30 = size_30.max()
    size_70 = size_70.max()

    if size_30 > size_70:
        print ("Tallo a la derecha")
        bw_esqueje = cv2.flip(bw_esqueje, 1)
    elif size_70 > size_30:
        print ("Tallo a la izquierda")

    return bw_esqueje, size


def rotate(stem_orientation, binary_cut):               #Function to rotate the binary image
    # Rotating cut as much degrees as stem orientation says
    final_binary_image_rotated = scipy.misc.imrotate(binary_cut, -stem_orientation)
    plt.imshow(final_binary_image_rotated)
    plt.show()
    return final_binary_image_rotated


def get_h_position(binary_image, fl, image):            #Function to find the leaf in the cut

    kernel = np.ones((3, 70), np.uint8)
    binary_image = cv2.erode(binary_image, kernel)
    binary_image = cv2.dilate(binary_image, kernel)

    bw = np.asarray(binary_image)
    flip_bw = np.rot90(bw, 1)
    (i, j) = flip_bw.nonzero()
    x_rect = i[0]
    flip_bw = cv2.flip(flip_bw, 0)

    (i, j) = flip_bw.nonzero()
    x_1 = i[0]
    y_1 = j[0]

    bw[bw == 1] = 255
    bw_a = 1 - bw
    # Se aplica la funcion distancia
    h_2 = ndimage.distance_transform_edt(1 - bw_a)
    plt.imshow(h_2)
    plt.set_cmap('nipy_spectral')
    plt.colorbar()
    plt.show()
    h_2[h_2 != 20] = 0
    h_2[h_2 > 0] = 1


    flip_bw_2 = np.rot90(h_2, 1)
    flip_bw_2 = cv2.flip(flip_bw_2, 0)

    (i, j) = flip_bw_2.nonzero()
    x_2 = i[0]
    y_2 = j[0]

    bw_lines = bw
    bw_lines[:, x_1] = 255
    bw_lines[:, -x_rect] = 255


    image = image[..., ::-1]
    fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

    ax.imshow(image)
    ax.axis('off')
    ax.set_title('Original image', fontsize=20)

    ax2.imshow(bw_lines, 'gray')
    ax2.plot(x_1, y_1, 'ro')
    ax2.plot(x_2, y_2, 'ro')
    ax2.axis('off')
    ax2.set_title('Final image', fontsize=20)

    fig.savefig("Resultados_3/" + fl[:9] + ".TIFF")

    return x_1, y_1, x_2, y_2, bw_lines


def area_filter(b):         #Function to get just the biggest object of the binary image
    b_ret = b
    im_x, contours, hierarchy = cv2.findContours(b_ret, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print ("Contours does not find anything")
        b_ret[:, :] = 0
    else:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        cnt = contours[0]
        b_ret[...] = 0
        cv2.drawContours(b_ret, [cnt], 0, 255, cv2.FILLED)

    return b_ret


def clasification(small_cm, large_cm, size, hoja_base):             #Function to classify the cut
    factor = 11.5 / 960
    length_cm = factor * size
    print ("El esqueje mide: " + str(length_cm) + "cm")
    if length_cm < small_cm:
        print ("Corto")
        category = 'Corto'
    elif (length_cm > small_cm and length_cm < large_cm) and (hoja_base != 1):
        print ("Ideal")
        category = 'Ideal'
    elif length_cm > large_cm and hoja_base == 0:
        print ("Largo")
        category = 'Largo'
    elif hoja_base == 1:
        print ("Hoja en base")
        category = 'Hoja en Base'

    return category, length_cm


if __name__ == '__main__':
    main()

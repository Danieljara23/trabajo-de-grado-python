# **********************************************************************
# Algoritmo para clasificacion de esquejes- Daniel Jaramillo Grisales
#             Facultad de Ingenieria Electronica
#                 Universidad de Antioquia
# ***********************************************************************

# ****************Libraries needed***************************************
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import signal
from skimage.morphology import skeletonize_3d
from skimage.morphology import erosion, dilation
from skimage.morphology import square
from skimage.morphology import rectangle
from skimage.morphology import medial_axis
# import scipy.misc
import os
path=r"G:\Nueva carpeta\Baltica_01_13_2017_C6_5L8_5H0_9"
list_all=os.listdir(path)

def main():
    # ***************Variables needed along the algorithm**************
    small_cm = 8
    large_cm = 9
    hoja_base_cm = 1

    factor = 11.5 / 960  # ************Factor px to cm
    small_px = small_cm / factor
    large_px = large_cm / factor
    hojabase_px = hoja_base_cm / factor

    print large_px
    print small_px

    # ******************************Image reading*****************************************
    for fl in list_all:
        if fl.endswith(".TIFF"):
            print '______________________________________________________'
            print fl
            img = cv2.imread(path + "\\" + fl)
            # img = cv2.imread('G:\Nueva carpeta\Baltica_01_13_2017_C6_5L8_5H0_9\Foto_1028_clase_2.TIFF')
            mask, mask2, cols, rows, res_rotated = segmentation(img)
            if cols < 200:
                category = '0'
                print("nada")
            else:
                final_image, skeleton, final_category, hojamm, x1, x2, y1 = classification(mask2, cols, rows, res_rotated, factor,small_px, large_px, hojabase_px)

                fig = plt.figure()
                a = fig.add_subplot(1, 2, 1)
                plt.imshow(final_image, 'gray')
                plt.plot(x1, y1, 'ro')
                plt.plot(x2, y1, 'ro')
                plt.title('Esqueje final')

                a = fig.add_subplot(1, 2, 2)
                plt.imshow(skeleton, 'gray')
                plt.title('Esqueleto')
                plt.show()

            # print(final_category)
            print("It's over")

            # print("La distancia entre la base y la primera hoja es:"+str(hojamm))


def segmentation(img):
    # Threshold
    ret, thresh = cv2.threshold(img[:, :, 0], 50, 255, cv2.THRESH_BINARY_INV)

    # ----------------
    thresh = erosion(thresh, rectangle(18,1))
    thresh = dilation(thresh, rectangle(18, 1))
    thresh = erosion(thresh, rectangle(1, 18))
    thresh = dilation(thresh, rectangle(1, 18))
    # se = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 1))
    # thresh = cv2.erode(thresh, se, iterations=1)
    # thresh = cv2.dilate(thresh, se, iterations=1)
    # se = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 18))
    # thresh = cv2.erode(thresh, se, iterations=1)
    # thresh = cv2.dilate(thresh, se, iterations=1)
    # ---------------
    # plt.imshow(thresh, 'gray')
    # plt.title('After dilate & erode')
    # plt.show()

    # ---------------------------------------------------------------------------
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print ("contours")
    print(contours)
    if not contours:
        thresh2 = img
        thresh3 = img
        cols = 100
        rows = 100
        res_rotated = img
    else:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        cnt = contours[0]
        thresh[...] = 0
        cv2.drawContours(thresh, [cnt], 0, 255, cv2.FILLED)
        res = cv2.bitwise_and(img, img, mask=thresh)

        # getting edges-rotating edges
        center_contour, size_contour, theta = cv2.fitEllipse(cnt)  # cv2.minAreaRect(cnt)
        res_rotated = rotate_and_scale(res, 1, theta)

        # Thresholding and getting edges again
        ret, thresh2 = cv2.threshold(res_rotated[:, :, 1], 5, 255, cv2.THRESH_BINARY)

        im2, contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)
        cnt2 = contours2[0]

        thresh2[...] = 0
        cv2.drawContours(thresh2, [cnt2], 0, 255, cv2.FILLED)

        x, y, w, h = cv2.boundingRect(cnt2)
        dst_roi = res_rotated[y:y + h, x:x + w]
        thresh3 = thresh2[y:y + h, x:x + w]

        cols, rows = dst_roi.shape[:2]
        if rows < cols:
            res_rotated = rotate_and_scale(dst_roi, 1, 90)
            thresh3 = rotate_and_scale(thresh3, 1, 90)
        cols, rows = dst_roi.shape[:2]

    return thresh2, thresh3, cols, rows, res_rotated

def rotate_and_scale(img, scale_factor=0.5, degrees_ccw=30):
    (old_y, old_x) = img.shape[:2]
    m = cv2.getRotationMatrix2D(center=(old_x / 2, old_y / 2), angle=degrees_ccw, scale=scale_factor)

    new_x, new_y = old_x * scale_factor, old_y * scale_factor
    r = np.deg2rad(degrees_ccw)
    new_x, new_y = (abs(np.sin(r) * new_y) + abs(np.cos(r) * new_x), abs(np.sin(r) * new_x) + abs(np.cos(r) * new_y))

    (tx, ty) = ((new_x - old_x) / 2, (new_y - old_y) / 2)
    m[0, 2] += tx
    m[1, 2] += ty

    rotated_img = cv2.warpAffine(img, m, dsize=(int(new_x), int(new_y)))
    return rotated_img


def classification(mask2, cols, rows, res_rotated, factor, small_px, large_px, h_base_px):
    category = '0'
    number_neighboring_pixels = 3



    # plt.imshow(mask2, 'gray')
    # plt.title('thresh3-mask2')
    # plt.show()

    # **************************Pre-proccess*******************
    b, g, r = cv2.split(res_rotated)
    res_rotated = cv2.merge([r, g, b])
    # plt.imshow(res_rotated, 'gray')
    # plt.title('FINAL')
    # plt.show()

    row, col = mask2.shape

    a = 0
    # ***************************Get stem's side***********************
    for c in range(col):
        a = np.append(a, (mask2[:, c] > 100).sum())
    part20 = cols * 0.2

    image20 = a[:int(part20)]
    image80 = a[-int(part20):]
    image20 = np.array(image20)
    image80 = np.array(image80)

    mean20 = image20.mean()
    mean80 = image80.mean()

    print 'mean20'
    print mean20
    print 'mean80'
    print mean80

    if mean20 < mean80:
        print 'Tallo a la izquierda'
    else:
        print 'Tallo a la derecha'
        mask2 = cv2.flip(mask2, 1)
        res_rotated = cv2.flip(res_rotated, 1)
# **********To detect Hoja en Base*******************************

    # plt.imshow(mask2, 'gray')
    # plt.title('Well positioned')
    # plt.show()

    ret, binarize_image = cv2.threshold(mask2, 0, 1, cv2.THRESH_BINARY)

    binarize_image = erosion(binarize_image, square(3))
    binarize_image = erosion(binarize_image, square(3))
    binarize_image = erosion(binarize_image, square(3))

    binarize_image = dilation(binarize_image, square(3))
    binarize_image = dilation(binarize_image, square(3))
    binarize_image = dilation(binarize_image, square(3))
    # Get Skeleton of full image
    # skeleton = medial_axis(binarize_image, return_distance=False)
    skeleton = skeletonize_3d(binarize_image)
    # Showing results of skeletonization
    # plt.imshow(skeleton, 'gray')
    # plt.title('Skeleton')
    # plt.show()

    bw_conv = signal.convolve2d(skeleton, np.ones((3, 3)), mode='same')
    # plt.imshow(bw_conv, 'gray')
    # plt.title('bw_conv')
    # plt.show()

    bw_conv = (bw_conv == number_neighboring_pixels + 1) & binarize_image
    # plt.imshow(bw_conv, 'gray')
    # plt.title('bw_conv')
    # plt.show()

    bw_sum = bw_conv.sum(axis=0)
    r = bw_sum.ravel().nonzero()

    # r[0][5] = 3
    print("r = ")
    print(r)

    r = np.array(r)
    #  if r.item(0) > 50:
    #     r = r[:-1]
    #    np.insert(r,0,9)
    # print("r = ")
    # print(r)
    v = np.ediff1d(r)
    print("este es v: ")
    print (v)

    v = v.tolist()
    max_difference = max(v)
    index_x1 = v.index(max_difference)
    x1_index = index_x1+1

    print("este es x1_index")
    print(x1_index)
    r = np.asarray(r)
    print (r)
    x1 = r[0][x1_index]
    x2 = r[0][x1_index - 1]

    y1 = bw_conv[:, x1]
    y1_index = y1.nonzero()
    y1 = y1_index[0][0]
    print ("la mxima diferencia es:")
    print (max_difference)
    print ("x1 es:"+str(x1))
    print ("x2 es:"+str(x2))
    print ("y1 es:"+str(y1))
    hojamm = (11.5 * max_difference / 345) * 10
    # ----------------------CLASIFICACION FINAL ----------------------
    flag_h_base = False
    if max_difference < h_base_px:
        category = '3'
        flag_h_base = True

    if cols < small_px:
        print 'corto'
        category = '1'
    elif (cols > large_px) & (flag_h_base == False):
        print 'largo'
        category = '2'
    elif (cols > small_px) & (cols < large_px):
        print 'ideal'
        category = 4
    elif cols < 200:
        print 'nada'
        category = '0'

    return res_rotated,skeleton,  category, hojamm, x1, x2, y1


if __name__ == '__main__':
    main()

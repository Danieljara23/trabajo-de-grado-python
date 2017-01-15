# **********************************************************************
# Algoritmo para clasificacion de esquejes- Daniel Jaramillo Grisales
#             Facultad de Ingenieria Electronica
#                 Universidad de Antioquia
# ***********************************************************************

# ****************Libraries needed***************************************
import numpy as np
# import Tkinter as Tki
import cv2
from matplotlib import pyplot as plt
# import os
import time


def main():
    # ***************Variables needed along the algorithm**************
    corto_cm = 8
    largo_cm = 9
    hoja_base_cm = 1

    factor = 11.5 / 960  # ************Factor px to cm
    global factor

    corto_px = corto_cm / factor
    global corto_px

    largo_px = largo_cm / factor
    global largo_px

    hojabase_px = hoja_base_cm / factor
    global hojabase_px

    print largo_px
    print corto_px
    print '______________________________________________________'

    # ******************************Image reading*****************************************
    img = cv2.imread('C:\Users\Daniel\Documents\MATLAB\esquejes-2016-08-30\Gui_Final\Baltica_10_03_2016\Foto_1005.TIFF')
    classification, mask, cnt, cols, rows = segmentacion(img)
    row, col = thresh3.shape

    a = 0
    global a
    # ***************************Get stem's side***********************
    for c in range(col):
        a = np.append(a, (thresh3[:, c] > 100).sum())
    part20 = cols * 0.2
    part80 = cols * 0.8
    sizea = a.size

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
        roi_tallo = image20
        roi_tallo = roi_tallo[:int(hojabase_px)]
        roi_tallo = np.array(roi_tallo)
        roi_tallo = roi_tallo[roi_tallo > 15]
        print roi_tallo
    else:
        print 'Tallo a la derecha'
        roi_tallo = image80[::-1]
        roi_tallo = roi_tallo[:int(hojabase_px)]
        roi_tallo = np.array(roi_tallo)
        roi_tallo = roi_tallo[roi_tallo > 15]
        print roi_tallo
    # **********To detect Hoja en Base*******************************

    meanroi = np.argmax(np.bincount(roi_tallo))
    valuesgtmean = len(roi_tallo[roi_tallo > meanroi + 2])
    print valuesgtmean
    if valuesgtmean > 15:
        print "Hoja en base"
        classification = '3'
    else:
        print "No es hoja en base"

    plt.plot(a)
    plt.ylabel('perfil')
    plt.show()
    print classification
    print '______________________________________________________'


def segmentacion(img):
    t = time.time()
    # Threshold
    ret, thresh = cv2.threshold(img[:, :, 0], 50, 255, cv2.THRESH_BINARY_INV)

    # ----------------
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 1))
    thresh = cv2.erode(thresh, se, iterations=1)
    thresh = cv2.dilate(thresh, se, iterations=1)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 18))
    thresh = cv2.erode(thresh, se, iterations=1)
    thresh = cv2.dilate(thresh, se, iterations=1)
    # ---------------

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = contours[0]
    thresh[...] = 0
    cv2.drawContours(thresh, [cnt], 0, 255, cv2.FILLED)
    res = cv2.bitwise_and(img, img, mask=thresh)

    # getting edges-rotating edges
    center_contour, size_contour, theta = cv2.fitEllipse(cnt)  # cv2.minAreaRect(cnt)
    res_rotated = rotate_and_scale(res, 1, theta)

    # Thresholding and getting of edges again
    ret, thresh2 = cv2.threshold(res_rotated[:, :, 1], 5, 255, cv2.THRESH_BINARY)

    im2, contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)
    cnt2 = contours2[0]

    thresh2[...] = 0
    cv2.drawContours(thresh2, [cnt2], 0, 255, cv2.FILLED)

    x, y, w, h = cv2.boundingRect(cnt2)
    dst_roi = res_rotated[y:y + h, x:x + w]
    thresh3 = thresh2[y:y + h, x:x + w]
    global thresh3
    cols, rows = dst_roi.shape[:2]
    if rows < cols:
        res_rotated = rotate_and_scale(dst_roi, 1, 90)
        thresh3 = rotate_and_scale(thresh3, 1, 90)
    cols, rows = dst_roi.shape[:2]

    # ***********************************Final Clasification*******************************
    clasification = '0'
    if cols < corto_px:
        print 'corto'
        clasification = '1'
    elif cols > largo_px:
        print 'largo'
        clasification = '2'
    elif (cols > corto_px) & (cols < largo_px):
        print 'ideal'
        clasification = 4
    elif cols < 200:
        print 'nada'
        clasification = '0'
    # --------------------------------------------

    # --------------------------------------------

    plt.imshow(thresh3, 'gray')
    plt.title('thresh3')
    plt.show()

    # **************************Pre-proccess*******************
    b, g, r = cv2.split(res_rotated)
    res_rotated = cv2.merge([r, g, b])
    plt.imshow(res_rotated, 'gray')
    plt.title('FINAL')
    plt.show()
    elapsed = time.time() - t
    print elapsed
    return clasification, thresh2, cnt2, cols, rows


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


if __name__ == '__main__':
    main()

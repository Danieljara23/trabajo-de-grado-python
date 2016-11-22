# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 00:52:16 2016

@author: Sebastian Guzman
"""
import numpy as np
import Tkinter as Tki
import cv2
from matplotlib import pyplot as plt
#from scipy import stats


import os

import time

#pathCortos = r"C:\Users\Sebastian Guzman\Google Drive\Baltica_21_09_2016\cortos\orientacion2"
#pathLargos = r"C:\Users\Sebastian Guzman\Google Drive\Baltica_21_09_2016\Largos\orientacion3"
#pathHB = r"C:\Users\Sebastian Guzman\Google Drive\Baltica_21_09_2016\HBase\orientacion3"
#path = pathHB
#listEsqueje = os.listdir(path)


def main():
    corto_cm = 8
    largo_cm = 9
    Hoja_base_cm = 1
    global factor
    factor = 11.5 / 960
    global corto_px
    corto_px = corto_cm / factor
    global largo_px
    largo_px = largo_cm / factor
    global hojabase_px
    hojabase_px = Hoja_base_cm / factor
    print largo_px
    print corto_px
    #for fl in listEsqueje:
    #    if fl.endswith(".TIFF"):
    print '______________________________________________________'
    #print fl
    #img = cv2.imread(path + "\\" + fl)
    img = cv2.imread('C:\Users\Daniel\Documents\MATLAB\esquejes-2016-08-30\Gui_Final\Baltica_10_03_2016\Foto_1005.TIFF')
    classification, mask, cnt,cols,rows = segmentacion(img)
    row, col = thresh3.shape

    global a
    a = 0

    for c in range(col):
        a = np.append(a, (thresh3[:, c] > 100).sum())
    part20 = cols * 0.2
    # print part20
    part80 = cols * 0.8
    # print part80
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
        print  roi_tallo

    meanroi = np.argmax(np.bincount(roi_tallo))
    print "valorr que mas se repite"
    print meanroi
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
    t = time.time()  # TIC
    global thresh3
    # UMBRALIZADO
    ret, thresh = cv2.threshold(img[:, :, 0], 50, 255, cv2.THRESH_BINARY_INV)

    #----------------
    SE = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 1))
    thresh = cv2.erode(thresh, SE, iterations=1)
    thresh = cv2.dilate(thresh, SE, iterations=1)
    SE = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 18))
    thresh = cv2.erode(thresh, SE, iterations=1)
    thresh = cv2.dilate(thresh, SE, iterations=1)
    #---------------

    #plt.imshow(thresh,'gray')
    #plt.title('thresh')
    #plt.show()
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = contours[0]
    thresh[...] = 0
    cv2.drawContours(thresh, [cnt], 0, 255, cv2.FILLED)
    res = cv2.bitwise_and(img, img, mask=thresh)

    #    plt.imshow(res,'gray')
    #    plt.title('res')
    #    plt.show()
    #


    # OBTENIENDO CONTORNOS Y ROTANDO
    centerContour, sizeContour, theta = cv2.fitEllipse(cnt)  # cv2.minAreaRect(cnt)
    resRotated = rotateAndScale(res, 1, theta)

    # UMBRALIZA Y OBTIENE NUEVAMENTE EL CONTORNO
    ret, thresh2 = cv2.threshold(resRotated[:, :, 1], 5, 255, cv2.THRESH_BINARY)
    #    plt.imshow(thresh2,'gray')
    #    plt.title('thresh2')
    #    plt.show()

    im2, contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)
    cnt2 = contours2[0]

    thresh2[...] = 0
    cv2.drawContours(thresh2, [cnt2], 0, 255, cv2.FILLED)
    #    plt.imshow(thresh2,'gray')
    #    plt.title('thresh2')
    #    plt.show()

    x, y, w, h = cv2.boundingRect(cnt2)
    # cv2.rectangle(resRotated,(x,y),(x+w,y+h),(0,255,0),7)
    dst_roi = resRotated[y:y + h, x:x + w]
    thresh3 = thresh2[y:y + h, x:x + w]
    #    plt.imshow(dst_roi,'gray')
    #    plt.title('dest_roi')
    #    plt.show()

    cols, rows = dst_roi.shape[:2]
    if rows < cols:
        resRotated = rotateAndScale(dst_roi, 1, 90)
        thresh3 = rotateAndScale(thresh3, 1, 90)
    cols, rows = dst_roi.shape[:2]
    print rows
    print 'Imprimí Rows'
    print cols
    clasificacion = '0'
    if cols < corto_px :
        print 'corto'
        clasificacion = '1'
    elif cols > largo_px:
        print 'largo'
        clasificacion = '2'
    elif ((cols > corto_px) & (cols < largo_px)):
        print 'ideal'
        clasificacion = 4
    elif cols < 200:
        print 'nada'
        clasificacion = '0'
    #--------------------------------------------

    #--------------------------------------------

    plt.imshow(thresh3, 'gray')
    plt.title('thresh3')
    plt.show()

    b, g, r = cv2.split(resRotated)  # get b,g,r
    resRotated = cv2.merge([r, g, b])  # switch it to rgb
    plt.imshow(resRotated, 'gray')
    plt.title('FINAL')
    plt.show()
    elapsed = time.time() - t
    print elapsed
    return clasificacion, thresh2, cnt2,cols,rows


def rotateAndScale(img, scaleFactor=0.5, degreesCCW=30):
    (oldY, oldX) = img.shape[:2]  # note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv2.getRotationMatrix2D(center=(oldX / 2, oldY / 2), angle=degreesCCW,
                                scale=scaleFactor)  # rotate about center of image.

    # choose a new image size.
    newX, newY = oldX * scaleFactor, oldY * scaleFactor
    # include this if you want to prevent corners being cut off
    r = np.deg2rad(degreesCCW)
    newX, newY = (abs(np.sin(r) * newY) + abs(np.cos(r) * newX), abs(np.sin(r) * newX) + abs(np.cos(r) * newY))

    # the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    # So I will find the translation that moves the result to the center of that region.
    (tx, ty) = ((newX - oldX) / 2, (newY - oldY) / 2)
    M[0, 2] += tx  # third column of matrix holds translation, which takes effect after rotation.
    M[1, 2] += ty

    rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX), int(newY)))
    return rotatedImg


if __name__ == '__main__':
    main()
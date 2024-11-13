import cv2
import numpy as np
import matplotlib.pyplot as plt 


def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

monedas = cv2.imread('./monedas.jpg',cv2.IMREAD_GRAYSCALE)

len(np.unique(monedas))

# --- CANNY ---------------------------------------------------------------------------------------
monedas_blur = cv2.GaussianBlur(monedas, ksize=(3, 3), sigmaX=1.5)
gcan1 = cv2.Canny(monedas_blur, threshold1=0.04*255, threshold2=0.1*255)
gcan2 = cv2.Canny(monedas_blur, threshold1=0.4*255, threshold2=0.5*255)
gcan3 = cv2.Canny(monedas_blur, threshold1=0.4*255, threshold2=0.75*255)
imshow(gcan1)


plt.figure()
ax = plt.subplot(221)
imshow(monedas, new_fig=False, title="Imagen Original")
plt.subplot(222, sharex=ax, sharey=ax), imshow(gcan1, new_fig=False, title="Canny - U1=4% | U2=10%")
plt.subplot(223, sharex=ax, sharey=ax), imshow(gcan2, new_fig=False, title="Canny - U1=40% | U2=50%")
plt.subplot(224, sharex=ax, sharey=ax), imshow(gcan3, new_fig=False, title="Canny - U1=40% | U2=75%")
plt.show(block=False)

# --- Hough Circulos --------------------------------------------------------------------------------
# Tutorial: https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html
# https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
img = monedas
img = cv2.medianBlur(img,5)
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=50, minRadius=150, maxRadius=-1)  # Circulos chicos
# circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT, 1.2, 20, param1=50, param2=30, minRadius=0, maxRadius=50) # Circulos grandes + otros
# circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT, 1.2, 20, param1=50, param2=50, minRadius=0, maxRadius=50) # Circulos grandes
circles = np.uint16(np.around(circles))
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for i in circles[0,:]:
    cv2.circle(cimg, (i[0],i[1]), i[2], (0,255,0), 2)   # draw the outer circle
    cv2.circle(cimg, (i[0],i[1]), 2, (0,0,255), 2)      # draw the center of the circle

2+2
imshow(cimg)

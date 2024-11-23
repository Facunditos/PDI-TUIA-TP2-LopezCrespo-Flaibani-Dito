import cv2
import numpy as np
import matplotlib.pyplot as plt

######################################## FUNCIONES ################################################################################

# Defininimos funcion para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
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

#  Reconstrucción Morgológica 
def imreconstruct(marker, mask, kernel=None):
    if kernel==None:
        kernel = np.ones((3,3), np.uint8)
    while True:
        expanded = cv2.dilate(marker, kernel)                               # Dilatacion
        expanded_intersection = cv2.bitwise_and(src1=expanded, src2=mask)   # Interseccion
        if (marker == expanded_intersection).all():                         # Finalizacion?
            break                                                           #
        marker = expanded_intersection        
    return expanded_intersection

# Utilizando reconstrucción morfológica
# NO rellena los huecos que tocan los bordes
def imfillhole(img):
    # img: Imagen binaria de entrada. Valores permitidos: 0 (False), 255 (True).
    mask = np.zeros_like(img)                                                   # Genero mascara para...
    mask = cv2.copyMakeBorder(mask[1:-1,1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255)) # ... seleccionar los bordes.
    marker = cv2.bitwise_not(img, mask=mask)                # El marcador lo defino como el complemento de los bordes.
    img_c = cv2.bitwise_not(img)                            # La mascara la defino como el complemento de la imagen.
    img_r = imreconstruct(marker=marker, mask=img_c)        # Calculo la reconstrucción R_{f^c}(f_m)
    img_fh = cv2.bitwise_not(img_r)                         # La imagen con sus huecos rellenos es igual al complemento de la reconstruccion.
    return img_fh



## FALTA ##
#def contar_dados(contorno, imagen) -> int:


#def contar_moneda(area):

###################################################################################################################################################

# Definir el path de la imagen de las monedas
path_imagen_monedas = r'C:\Users\Usuario\Documents\TECNICATURA EN IA\PROCESAMIENTO DE IMAGENES\PDI_2024\TP2\monedas.jpg' 
#img = cv2.imread("./monedas.jpg", cv2.IMREAD_GRAYSCALE)   #./monedas.jpg le indica a cv2.imread que busque la imagen en la carpeta actual.

# Cargar la imagen desde el path especificado
img = cv2.imread(path_imagen_monedas, cv2.IMREAD_GRAYSCALE)

#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
"""
Imagen a color pero transformo el espacio de color de BGR (azul, verde, rojo) a RGB (rojo, verde, azul).

Ya que OpenCV carga las imágenes en formato BGR por defecto. Este cambio a RGB es necesario 
para  mostrar la imagen usando matplotlib (espera las imágenes en formato RGB)"""

#-------------------- BLUR PARA HOMOGENEIZAR FONDO ------------------

img_blur = cv2.medianBlur(img, 9, 2)
# plt.figure()
# ax1 = plt.subplot(121); plt.imshow(img, cmap="gray"), plt.title("Imagen")
# plt.subplot(122, sharex=ax1, sharey=ax1), plt.imshow(img_blur, cmap="gray"), plt.title("Imagen + blur")
# plt.show(block=False)

# --------------------------------- CANNY --------------------------------

# img_canny_CV2 = cv2.Canny(img_blur, 50, 115, apertureSize=3, L2gradient=True)
img_canny_CV2 = cv2.Canny(img_blur, 10, 54, apertureSize=3, L2gradient=True)
#imshow(img_canny_CV2)


f=img_canny_CV2.copy()
plt.figure()
ax1 = plt.subplot(121); plt.imshow(img, cmap="gray"), plt.title("Imagen")
plt.subplot(122, sharex=ax1, sharey=ax1), plt.imshow(f, cmap="gray"), plt.title("Imagen + blur + canny")
plt.show(block=False)

# --------------------------- DILATACION Y CLAUSURA -------------------

k = 22
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))

fd = cv2.dilate(f, kernel)
fc = cv2.morphologyEx(fd, cv2.MORPH_CLOSE, (7,7))

#fe = cv2.erode(fd, 3)
imshow(fc, title= 'Dilatacion + Clausura')


#----------------------RELLENADO DE HUECOS FUNCION + APERTURA---------------------------------

rellenada=imfillhole(fc)
imshow(rellenada, title='Rellenada')

rellenada2=rellenada.copy()

kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(121,121))

rConClausura = cv2.morphologyEx(rellenada2, cv2.MORPH_OPEN, kernel3)

imshow(rConClausura,title='Rellenada + Clausura')







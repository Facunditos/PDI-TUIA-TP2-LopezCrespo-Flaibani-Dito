import cv2
import numpy as np
import matplotlib.pyplot as plt

#-------------------
# Funciones
#-------------------

'''
Muestra imágenes por pantalla
'''
# Defininimos función para mostrar imágenes
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

# ---------------------------------------------------------------------------------------
# --- Reconstrucción Morgológica --------------------------------------------------------
# ---------------------------------------------------------------------------------------
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

# --- Version 1 ------------------------------------------------
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

# --- Version 2 ------------------------------------------------
# Utilizando cv2.floodFill()
# SI rellena los huecos que tocan los bordes
def imfillhole_v2(img):
    img_flood_fill = img.copy().astype("uint8")             # Genero la imagen de salida
    h, w = img.shape[:2]                                    # Genero una máscara necesaria para cv2.floodFill()
    mask = np.zeros((h+2, w+2), np.uint8)                   # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#floodfill
    cv2.floodFill(img_flood_fill, mask, (0,0), 255)         # Relleno o inundo la imagen.
    img_flood_fill_inv = cv2.bitwise_not(img_flood_fill)    # Tomo el complemento de la imagen inundada --> Obtenog SOLO los huecos rellenos.
    img_fh = img | img_flood_fill_inv                       # La salida es un OR entre la imagen original y los huecos rellenos.
    return img_fh



'''
Programa principal
'''
img = cv2.imread('monedas.jpg',cv2.IMREAD_GRAYSCALE)
imshow(img)
# Filtro de Mediaba
blurred = cv2.medianBlur(img, 9)
# imshow(blurred)
# Detectar bordes con Canny
edges= cv2.Canny(blurred, 10, 50, apertureSize=3, L2gradient=True)
imshow(edges)

# Morfologia

# Apertura vs Cierre
# Apertura (erosión → dilatación):
# Elimina ruido y suaviza bordes. Útil para eliminar elementos pequeños.
# Una forma de "limpiar" la imagen de detalles muy pequeños o ruido.

# Cierre (dilatación → erosión):
# Cierra huecos pequeños y une componentes. Útil para hacer los objetos más completos.
# una manera de "sellar" huecos y unir componentes dentro de un área conectada.

# rellena solo las imágenes cerradas
# prueba1 = imfillhole_v2(edges)
# imshow(prueba1)

# # Solo clausura... agranda un poquito
# prueba2 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, (40,40))
# imshow(prueba2)

# # Solo apertura...no sirve, achica la imagen
# prueba3 = cv2.morphologyEx(edges, cv2.MORPH_OPEN, (40,40))
# imshow(prueba3)

# dilatar
k1 = 17#20
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k1, k1))
dilated = cv2.dilate(edges, kernel1)
imshow(dilated)
# clausura
k2 = 3#7
closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, (k2, k2))
imshow(closed)
# rellenar huecos
filled=imfillhole_v2(closed)
imshow(filled)
# apertura
# saca el ruido...no sería necesario. Se pueden ignorar las formas pequeñas en componentes conectados
k3 = 121
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k3, k3))
filled_open = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel3)
imshow(filled_open)

# Componentes conectados
# Configuración de parámetros
img_seg = img.copy()
connectivity = 8
min_area = 100

# Encontrar los componentes conectados
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled_open, connectivity, cv2.CV_32S)

# Crear una imagen de salida para los componentes que cumplen con el área mínima
output_img = np.zeros_like(filled_open)

# Colorear las áreas que cumplen con el área mínima
labels_colored = np.uint8(255 / num_labels * labels)
im_color = cv2.applyColorMap(labels_colored, cv2.COLORMAP_JET)

# Iterar sobre los componentes y procesar solo los que cumplen el área mínima
for i in range(1, num_labels):  # Ignorar el componente 0 (fondo)
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= min_area:
        # Marcar los píxeles del componente en la imagen de salida
        output_img[labels == i] = 255  # Solo copiar los componentes que cumplen el área mínima

        # Dibujar el centroide
        cv2.circle(im_color, tuple(np.int32(centroids[i])), 9, color=(255, 255, 255), thickness=-1)

        # Dibujar el cuadro delimitador (rectángulo)
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        cv2.rectangle(img_seg, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

imshow(output_img)
imshow(img_seg)
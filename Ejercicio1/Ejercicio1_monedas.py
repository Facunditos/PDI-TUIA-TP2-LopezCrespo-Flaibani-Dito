import cv2
import matplotlib.pyplot as plt
import numpy as np

"""
Resumen del Proceso:
1. Cargar la imagen a color.
2. Convertirla a escala de grises para facilitar la segmentación.
3. Aplicar un filtro de suavizado (como el Gaussiano) para reducir el ruido.
4. Segmentar usando un umbral para separar los objetos del fondo.
5. Usar operaciones morfológicas para mejorar la segmentación si es necesario.
6. Encontrar los contornos de los objetos.
7. Dibujar los contornos sobre la imagen original para visualizarlos.
"""
#-------------------
# Funciones
#-------------------
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

def imfillhole(img):
    # img: Imagen binaria de entrada. Valores permitidos: 0 (False), 255 (True).
    mask = np.zeros_like(img)                                                   # Genero mascara para...
    mask = cv2.copyMakeBorder(mask[1:-1,1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255)) # ... seleccionar los bordes.
    marker = cv2.bitwise_not(img, mask=mask)                # El marcador lo defino como el complemento de los bordes.
    img_c = cv2.bitwise_not(img)                            # La mascara la defino como el complemento de la imagen.
    img_r = imreconstruct(marker=marker, mask=img_c)        # Calculo la reconstrucción R_{f^c}(f_m)
    img_fh = cv2.bitwise_not(img_r)                         # La imagen con sus huecos rellenos es igual al complemento de la reconstruccion.
    return img_fh


def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False)-> None:
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
                                                

###################################################  PRUEBAS VARIAS ########################################################

image = cv2.imread('Ejercicio1\\monedas.jpg')              # Cargar la imagen a color por si luego la necesito
#img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Aunque la imagen es a color, para segmentar, convertir a escala de grises 
                                               # (ayudará a trabajar con las variaciones de intensidad de los píxeles y separar los objetos del fondo.
img1 = cv2.imread('Ejercicio1\\monedas.jpg',cv2.IMREAD_GRAYSCALE)  # OTRA OPCIÓN DONDE no se cargan los canales de color, solo el valor de luminosidad (escala de grises), lo que hace que la imagen ocupe menos memoria                                                 
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convierte de BGR a RGB, ya que matplotlib espera imágenes en RGB para visualizarlas correctamente.

fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Crea subplots : 1 fila, 2 columnas

# Mostrar la imagen original en color
axs[0].imshow(image_rgb)
axs[0].set_title('Imagen a Color')
axs[0].axis('off')  # No mostrar los ejes

# Mostrar la imagen en escala de grises
axs[1].imshow(img1, cmap='gray')   # aca mapea como minimo el minimo valor de la imagen y maximo el maximo ( que puede no ser 255)
axs[1].set_title('Imagen en Escala de Grises')
axs[1].axis('off')  # No mostrar los ejes

plt.tight_layout()
plt.show()      

# --- PRUEBAS DE Muestreo imagen con Matplotlib -------------------------------------------------plt.figure(1)
h = plt.imshow(img1, cmap='gray') 
plt.title('Imagen')
plt.xlabel('X')
plt.ylabel('Y')
plt.xticks([])
plt.yticks([])
plt.show()


h = plt.imshow(img1, cmap='gray', vmin=0, vmax=10) # satura
plt.colorbar(h)
plt.show()
"""vmin=0 y vmax=10: cualquier valor en la imagen que sea menor o igual a "0" se mostrará como negro,
 y >= a 10 se mostrará como blanco. Los valores entre 0 y 10 se mapearán a niveles de gris intermedios."""

plt.subplot(121)  # 1 fila 2 columnas 1er grafico
h = plt.imshow(img1, cmap='gray') # sin setaer vmin y vmax --> normalizando: maximizo el contraste de la imagen solo para visualizarla
plt.colorbar(h)
plt.title('Imagen - normalizada')  # Normalizada:significa que el negro es el minimo valor de la imagen y el blanco el maximo,( ej: que quede en blanco un valor de pixel 101 )
plt.subplot(122)  # 1 fila 2 columnas 2do grafico
h = plt.imshow(img1, cmap='gray', vmin=0, vmax=255) # Saturo: seteo minimo negro,maximo blanco (Cualquier valor fuera del rango se satura.)
plt.colorbar(h)
plt.title('Imagen - sin normalizar')
plt.show()

# Info
type(img1)
img1.dtype
img1.shape  # (2681, 3871)
w,h = img1.shape

# Stats
img1.min()  # 0
img1.max()  # 255
pix_vals = np.unique(img1)
N_pix_vals = len(np.unique(img1))

##################### Transformación y Filtrado en Imagen : Recorte, Escalado, y Rotación ###############################
""" ESTO NO LO ESTOY USANDO POR AHORA

# Recorte: (Slicing): subconjunto de la imagen img1, desde el píxel (50, 50) hasta el píxel (200, 200) en las dimensiones (altura, anchura).
# 50:200 para las filas (altura): selecciona píxeles en la dirección vertical desde la fila 50 hasta la fila 200.
# 50:200 para las columnas (anchura): selecciona píxeles en la dirección horizontal desde la columna 50 hasta la columna 200.
# útil cuando quieres trabajar solo con una región específica de la imagen

recorte = img1[50:200, 50:200]

# Escalado
escalado = cv2.resize(img1, (200, 200))

# Rotación
centro = (img1.shape[1] // 2, img1.shape[0] // 2)
matriz_rotacion = cv2.getRotationMatrix2D(centro, 45, 1.0)
rotada = cv2.warpAffine(img1, matriz_rotacion, (img1.shape[1], img1.shape[0]))

# Mostrar imagenes
cv2.imshow("Recorte", recorte)
cv2.imshow("Escalado", escalado)
cv2.imshow("Rotación", rotada)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

######################## Ecualización de histograma ###############################################################

# La transformación de intensidad mejora el contraste de la imagen, destacando características clave. 
# La ecualización del histograma es particularmente útil para mejorar el detalle en imágenes de bajo contraste.
ecualizada = cv2.equalizeHist(img1)  # cv2.equalizeHist solo funciona en imágenes en escala de grises
# Mostrar con opencv
cv2.imshow("Ecualizada", ecualizada)   
cv2.waitKey(0)                         # Cerrar ventanas

# Mostrar la imagen ecualizada con matplotlib
plt.imshow(ecualizada, cmap='gray')  # plt.imshow es que permite visualizar la imagen en un Jupyter Notebook o entorno interactivo sin necesidad de una ventana emergente como cv2.imshow.
plt.title("Imagen Ecualizada")
plt.axis("off")  # Ocultar ejes
plt.show()

"""
La ecualización de histograma es una técnica en procesamiento de imágenes utilizada para mejorar el contraste de una imagen.
OBJ: distribuir de manera más uniforme los niveles de intensidad (brillo) en toda la imagen, de forma que los detalles en 
áreas oscuras o claras sean más visibles.

¿Cómo Funciona?
ajusta el brillo de los píxeles en la imagen para que el histograma (distribución de niveles de gris) sea lo más uniforme
posible. Esto aumenta el contraste en áreas de intensidad media, destacando detalles que podrían estar ocultos en imágenes 
de bajo contraste.

¿Cuándo Usar la Ecualización?
*Imágenes con bajo contraste: Es ideal para imágenes donde los detalles se ven oscuros o tienen un rango de brillo limitado, como en imágenes de rayos X o fotos en condiciones de poca luz.
*Destacar detalles ocultos: Útil cuando se desea revelar detalles en sombras o en áreas muy brillantes.
*Preparación para análisis: Antes de aplicar otros algoritmos de procesamiento (como detección de bordes o segmentación), 
la ecualización mejora el contraste y facilita la detección de características importantes.
"""

######################### Filtrado Espacial (lineal y no lineal) ################################################
# Los filtros espaciales ayudan a eliminar ruido (filtros no lineales como el filtro de mediana) o a 
# resaltar bordes y detalles (filtros lineales como el filtro Laplaciano).

# Filtro de mediana
mediana = cv2.medianBlur(ecualizada, 5)

# Filtro Laplaciano
#laplaciano = cv2.Laplacian(ecualizada, cv2.CV_64F)

# Mostrar
plt.imshow(mediana, cmap='gray')  # plt.imshow es que permite visualizar la imagen en un Jupyter Notebook o entorno interactivo sin necesidad de una ventana emergente como cv2.imshow.
plt.title("Imagen filtro de mediana")
plt.axis("off")  # Ocultar ejes
plt.show()

# OTRA ALTERNATIVA -->filtrado Gaussiano, PASA-BAJOS, que suaviza la imagen sin eliminar tantos detalles finos como el filtro de mediana.
gaussiano = cv2.GaussianBlur(ecualizada, (5, 5), 0) # El tamaño del kernel es importante, generalmente debe ser un valor impar como (5,5), (3,3), etc.

# Mostrar la imagen original y la imagen filtrada con plt.imshow
plt.figure(figsize=(10, 5))

# Imagen original
plt.subplot(1, 2, 1)
plt.imshow(ecualizada, cmap='gray')  # Mostrar la imagen en escala de grises
plt.title("Imagen Original")
plt.axis("off")

# Imagen con filtro Gaussiano
plt.subplot(1, 2, 2)
plt.imshow(gaussiano, cmap= 'gray') 
plt.title("Filtro Gaussiano")
plt.axis("off")

plt.show()

# Guardar la imagen filtrada
cv2.imwrite('imagen_filtrada_Gaussiano.jpg', gaussiano)

# con Matplotlib
# Crear los subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 fila, 3 columnas

# Mostrar las imágenes en subplots
axs[0].imshow(img1, cmap = "gray")
axs[0].set_title('Imagen Original')
axs[0].axis('off')  # No mostrar los ejes

axs[1].imshow(ecualizada, cmap = "gray")
axs[1].set_title('Imagen Ecualizada')
axs[1].axis('off')  # No mostrar los ejes

axs[2].imshow(gaussiano, cmap = "gray")
axs[2].set_title('Imagen Filtrada')
axs[2].axis('off')  # No mostrar los ejes

plt.tight_layout()
plt.show()

#################### Segmentación usando un Umbral (Thresholding) ####################################
"""
Para segmentar la imagen, se utiliza un umbral (thresholding). Si el valor de cada pixel es mayor que el umbral 120, 
se establece el pixel en blanco (255), y si es menor, se establece en negro (0).
"""
# Usar un umbral para separar los objetos del fondo
_, threshold_image = cv2.threshold(gaussiano, 120, 255, cv2.THRESH_BINARY)

##################### Aplicar Operaciones de Morfología #############################################
"""
Si los objetos están fragmentados o si hay ruido en la segmentación,  aplicar operaciones morfológicas
como la dilatación o la erosión para mejorar el resultado
"""
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # elemento estructural rectangulo
morph_image = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel)  # CLAUSURA --> DILATA + EROSIONA

##################### Encontrar Contornos de los Objetos ###########################################
"""Una vez segmentada la imagen, usar la función findContours() para detectar los contornos de los objetos."""
contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

##################### Dibujar los Contornos en la Imagen Original #################################
"""dibujar los contornos detectados sobre la imagen original para visualizar los objetos segmentados."""

# Dibujar los contornos sobre la imagen original
image_with_contours= cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # Dibujar en verde
plt.imshow(image_with_contours)
plt.show()

# Contar los objetos (monedas y dados)
num_objects = len(contours) # 1666

# Mostrar la imagen original con los contornos dibujados
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_with_contours, cmap = "gray")
plt.title(f'Contornos Detectados - {num_objects} objetos')
plt.axis('off')

plt.tight_layout()
plt.show()

# Opcionalmente, puedes imprimir el número de objetos encontrados
print(f"Número de objetos encontrados: {num_objects}")

#######################################################################################################

################### segmentación de las monedas y los dados, ##########################################
#  filtrado Gaussiano,
#  la detección de bordes (Canny), 
#  la transformada de Hough para detectar círculos (posibles monedas), 
# y el etiquetado de componentes conectados para identificar los puntos en cada dado. 
######################################################################################################

img = cv2.imread(f"Ejercicio1\\monedas.jpg") # Cargar la imagen
img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#gray = cv2.equalizeHist(img_gris, cv2.COLOR_BGR2GRAY)

# Aplicar filtro Gaussiano para suavizar la imagen----------------------------------------
#gaussiano = cv2.GaussianBlur(img_gris, (3, 3), 2) # 2) es el parámetro sigmaX, que controla el grado de suavizado  (Ajustar nivel de desenfoque aplicado)
gaussiano = cv2.GaussianBlur(img_gris, (5, 5), 0)

# Detección de bordes con Canny ------------------------------------------------------------
bordes = cv2.Canny(gaussiano, 50, 100)

# Detectar círculos (monedas) con la Transformada de Hough-----------------------------------
#circulos = cv2.HoughCircles(bordes, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=100, param2=40, minRadius=20, maxRadius=40)
#circulos = cv2.HoughCircles(bordes, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=120, param2=40, minRadius=20, maxRadius=50)
circulos = cv2.HoughCircles(
    bordes,
    cv2.HOUGH_GRADIENT,
    dp=1,            # Resuelve los círculos a la resolución original de la imagen
    minDist=50,      # Reduce la distancia mínima entre los centros de los círculos
    param1=100,       # Umbral para Canny (borde más sensible)
    param2=40,       # Umbral de acumulación (reduce para más detección)
    minRadius=20,    # Tamaño mínimo de las monedas
    maxRadius=60     # Tamaño máximo de las monedas
)

# Dibujo de círculos detectados en una copia de la imagen original
resultado_circulos = img_gris.copy()
if circulos is not None:
    circulos = np.round(circulos[0, :]).astype("int")
    for (x, y, r) in circulos:
        cv2.circle(resultado_circulos, (x, y), r, (0, 255, 0), 4)

# UMBRALADO - BINARIZACIÓN - Segmentación de los puntos en los dados usando COMPONENTES CONECTADAS ---------------------
_, binarizada = cv2.threshold(bordes, 150, 255, cv2.THRESH_BINARY)
#num_labels, labels_im = cv2.connectedComponents(bordes)

connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bordes, connectivity, cv2.CV_32S)

# Visualización con plt.imshow
plt.figure(figsize=(15, 10))

# Imagen original
plt.subplot(2, 2, 1)
#plt.imshow(cv2.cvtColor(img_gris, cv2.COLOR_BGR2RGB))
plt.imshow(img_gris)
plt.title("Imagen Original")
plt.axis("off")

# Imagen con filtro Gaussiano
plt.subplot(2, 2, 2)
plt.imshow(gaussiano, cmap="gray")
plt.title("Filtro Gaussiano")
plt.axis("off")

# Bordes detectados con Canny 
plt.subplot(2, 2, 3)
plt.imshow(bordes, cmap="gray")
plt.title("Bordes con Canny")
plt.axis("off")

# Monedas detectadas con Transformada de Hough
plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(resultado_circulos, cv2.COLOR_BGR2RGB))
plt.title("Círculos Detectados (Monedas)")
plt.axis("off")

plt.show()

# Conteo de puntos en cada dado ---------------------------------------------------------------
# Asumimos que los dados se detectan como regiones etiquetadas en 'labels_im'
contornos, _ = cv2.findContours(binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i, contorno in enumerate(contornos):
    # Calcular el área y descartar regiones pequeñas
    if cv2.contourArea(contorno) > 100:  # Ajustar según tamaño de puntos
        (x, y, w, h) = cv2.boundingRect(contorno)
        cv2.rectangle(img_gris, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Mostrar imagen final con regiones de los puntos en cada dado
plt.imshow(cv2.cvtColor(img_gris, cv2.COLOR_BGR2RGB))
plt.title("Identificación de Puntos en los Dados")
plt.axis("off")
plt.show()

def contar_dados(contorno, imagen) -> int:
    mask = np.zeros_like(imagen)
    # Dibuja el contorno en la máscara
    cv2.drawContours(mask, [contorno], -1, 255, thickness=cv2.FILLED)
    # Aplica la máscara a la imagen original
    dado_recortado = cv2.bitwise_and(imagen, imagen, mask=mask)
    dado_recortado_u =  cv2.threshold(dado_recortado, 168, 255, cv2.THRESH_BINARY_INV)[1]
    dado_recortado_c = cv2.Canny(dado_recortado_u, 0, 255, apertureSize=3, L2gradient=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    dado_recortado_para_components = cv2.dilate(dado_recortado_c, kernel)
    # Encuentra componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dado_recortado_para_components, 8)
    # Especifica el umbral de área
    area_threshold = (700, 3000)  # UMBRAL DE AREA
    # Filtra las componentes conectadas basadas en el umbral de área
    filtered_labels = []
    filtered_stats = []
    filtered_centroids = []
    for label in range(1, num_labels):  # comienza desde 1 para excluir el fondo (etiqueta 0)
        area = stats[label, cv2.CC_STAT_AREA]
        if area > area_threshold[0] and area < area_threshold[1]:
            filtered_labels.append(label)
            filtered_stats.append(stats[label])
            filtered_centroids.append(centroids[label])
    return len(filtered_centroids)


def contar_moneda(area):
    if area > 69000 and area < 80000:
        return 10
    elif area > 80000 and area < 110000:
        return 1
    else:
        return 50


###################################################################
# Programa principal

img = cv2.imread('Ejercicio1\\monedas.jpg', cv2.IMREAD_GRAYSCALE)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

# --------------------------- DILATACION Y CLAUSURA -------------------

k = 22
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
fd = cv2.dilate(f, kernel)
fc = cv2.morphologyEx(fd, cv2.MORPH_CLOSE, (7,7))

# fe = cv2.erode(fd, 3)
#imshow(fc, title= 'Dilatacion + Clausura')

#----------------------RELLENADO DE HUECOS FUNCION + APERTURA---------------------------------

rellenada=imfillhole(fc)
#imshow(rellenada, title='Rellenada')

rellenada2=rellenada.copy()
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(121,121))
rConClausura = cv2.morphologyEx(rellenada2, cv2.MORPH_OPEN, kernel3)
#imshow(rConClausura,title='Rellenada + Clausura')

#-------------------------CONTORNOS PARA SEPARAR Y CLASIFICAR ELEMENTOS------------------------

imgContour = rConClausura.copy()

# Encuentra componentes conectados
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imgContour, 8)

# Especifica el umbral de área
area_threshold = 300  # UMBRAL DE AREA

# Filtra las componentes conectadas basadas en el umbral de área
filtered_labels = []
filtered_stats = []
filtered_centroids = []
for label in range(1, num_labels):  # comienza desde 1 para excluir el fondo (etiqueta 0)
    area = stats[label, cv2.CC_STAT_AREA]
    if area > area_threshold:
        filtered_labels.append(label)
        filtered_stats.append(stats[label])
        filtered_centroids.append(centroids[label])
# Convierte las listas filtradas a matrices
filtered_labels = np.array(filtered_labels)
filtered_stats = np.array(filtered_stats)
filtered_centroids = np.array(filtered_centroids)


####################################################CLASIFICACION##########################################

cnt, _ = cv2.findContours(rConClausura, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
objetos = (labels).astype(np.uint8)
cnt, _ = cv2.findContours(objetos, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

labeled_shapes = np.zeros_like(rConClausura)

dados = []
monedas = []
factdeforma = []

for i in range(1, num_labels):
    objeto = (labels == i).astype(np.uint8)
    cont, _ = cv2.findContours(objeto, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # cv2.findContours(image, mode, method)
    # objeto: imagen binaria donde se han segmentado los objetos a través de técnicas como la umbralización, el binarizado o la detección de bordes. Debe ser una imagen de tipo CV_8UC1 (imagen en escala de grises con valores entre 0 y 255, donde 0 es el fondo y 255 es el objeto).
    # salida: contours, hierarchy = cv2.findContours(...)
    # la jerarquía describe las relaciones entre los contornos (por ejemplo, si un contorno está contenido dentro de otro). Si se usa cv2.RETR_LIST, la jerarquía no tiene relación entre los contornos, 
    # por lo que puede no ser útil, por lo que generalmente se usa un _ (guion bajo) para descartarlo.
    if cont:
        for c in cont:
            dados_d={}
            monedas_d={}
            # Calcula el factor de forma (fp) que es la inversa del área del contorno dividido por la longitud del perímetro del contorno.
            area = cv2.contourArea(c) # calcula el área de un contorno cerrado,
            p = cv2.arcLength(c, True) # calcula la longitud del contorno, distancia total alrededor del perímetro de la figura representada por c. 
            fp = 1 / (area / p ** 2) 
            #print(i, cont)
            factdeforma.append(fp) # Usando la inversa del factor de forma, filtrás dado si esa inversa es mayor a 14 y menos a 15
            if 14 <= fp < 15:      # 1/Fp = 12.57
                monedas_d['area']=area
                monedas_d['img']=objeto
                monedas_d['contorno']=c
                monedas_d['valor']=contar_moneda(area)
                monedas.append(monedas_d)
            else:
                dados_d['fp']=fp
                dados_d['img']=objeto
                dados_d['contorno']=c
                dados_d['valor']=contar_dados(c, img)
                dados.append(dados_d)
"""
Fp: Facto de Forma = Área / Perímetro^2

Este factor de forma es útil para comparar el tamaño de un objeto con respecto a su forma, y puede ser útil para 
detectar objetos de diferentes tamaños y formas.

*** Círculo *******************************
A = pi.r^2
P = pi.(2*r)
Fp = A/P^2 	= (pi.r^2) / (pi.(2*r))^2
			= (pi.r^2) / (pi^2.4.r^2)
			= 1 / (pi*4)
			= 0.0796
			
--> 1/Fp = 12.57		la inversa del factor de forma, pero usaremos entre 14 y 15 de rango.	

============================================================			

Contorno  	--> cv2.findContours() --> cnt
Area		--> cv2.contourArea(cnt) 
Perímetro	--> cv2.arcLength(cnt, True)
"""

#DEBUG

# for x in range(len(monedas)):
#     print(monedas[x]['valor'])


# Dibuja los bounding boxes en la imagen

for i in range(19):
    x, y, w, h, area = filtered_stats[i]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Muestra el resultado
plt.imshow(img)
plt.title('Original + BB')
plt.show()


###################################################################
# Resultado Final

img = cv2.imread('Ejercicio1\\monedas.jpg',cv2.IMREAD_COLOR)

# Crear una figura con 2 filas y 2 columnas para mostrar las imágenes
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Subfigura 1: Monedas con valor 50
axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Monedas con Valor .50')
for moneda in monedas:
    if moneda['valor'] == 50:
        x, y, w, h = cv2.boundingRect(moneda['contorno'])
        axs[0, 0].add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none'))
        axs[0, 0].text(x, y, f"50", color='g', fontsize=10)
axs[0, 0].text(2500,2500,f"Cantidad de monedas: {len([m for m in monedas if m['valor'] == 50])}", color='g', fontsize=10, ha='center')

# Subfigura 2: Monedas con valor 1
axs[0, 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0, 1].set_title('Monedas con Valor 1')
for moneda in monedas:
    if moneda['valor'] == 1:
        x, y, w, h = cv2.boundingRect(moneda['contorno'])
        axs[0, 1].add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none'))
        axs[0, 1].text(x, y, f"1", color='r', fontsize=10)
axs[0, 1].text(2500,2500,f"Cantidad de monedas: {len([m for m in monedas if m['valor'] == 1])}", color='r', fontsize=10, ha='center')

# Subfigura 3: Monedas con valor 10
axs[1, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title('Monedas con Valor 10')
for moneda in monedas:
    if moneda['valor'] == 10:
        x, y, w, h = cv2.boundingRect(moneda['contorno'])
        axs[1, 0].add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='b', facecolor='none'))
        axs[1, 0].text(x, y, f"10", color='b', fontsize=10)
axs[1, 0].text(2500,2500,f"Cantidad de monedas: {len([m for m in monedas if m['valor'] == 10])}", color='b', fontsize=10, ha='center')
# Subfigura 4: Dados
axs[1, 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title('Dados')
for dado in dados:
    x, y, w, h = cv2.boundingRect(dado['contorno'])
    axs[1, 1].add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='m', facecolor='none'))
    axs[1, 1].text(x, y, f"{dado['valor']}", color='m', fontsize=10)

plt.show()


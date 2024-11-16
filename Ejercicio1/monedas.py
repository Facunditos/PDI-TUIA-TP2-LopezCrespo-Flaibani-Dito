import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import KMeans

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

def preprocess_image(img):
    # Filtro de Mediaba
    blurred = cv2.medianBlur(img, 9)
    # imshow(blurred)
    # Detectar bordes con Canny
    edges= cv2.Canny(blurred, 10, 50, apertureSize=3, L2gradient=True)
    # imshow(edges)
    # Morfologia
    # dilatar
    k1 = 17#20
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k1, k1))
    dilated = cv2.dilate(edges, kernel1)
    # imshow(dilated)
    # clausura
    k2 = 3#7
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, (k2, k2))
    # imshow(closed)
    # rellenar huecos
    filled=imfillhole_v2(closed)
    # imshow(filled)
    # apertura
    # saca el ruido...no sería necesario. Se pueden ignorar las formas pequeñas en componentes conectados
    k3 = 121
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k3, k3))
    filled_open = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel3)
    # imshow(filled_open)
    return filled_open

def count_dados(img, coords):
    x, y, w, h = coords
    crop_img = img[y:y+h, x:x+w]
    circles = cv2.HoughCircles(crop_img,cv2.HOUGH_GRADIENT, 1, 10, param1=50, param2=40, minRadius=20, maxRadius=30)
    return circles.shape[1]

def detect_components(img):
    # img_seg = img.copy()
    connectivity = 8
    # Encontrar los componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
    # Crear una imagen de salida para los componentes que cumplen con el área mínima
    # output_img = np.zeros_like(img)
    monedas = []
    dados = []
    # Iterar sobre los componentes y procesar solo los que cumplen el área mínima
    for i in range(1, num_labels):  # Ignorar el componente 0 (fondo)
        # Marcar los píxeles del componente en la imagen de salida
        # output_img[labels == i] = 255  # Solo copiar los componentes que cumplen el área mínima
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        obj = (labels == i).astype(np.uint8)
        contours, hierarchy = cv2.findContours(obj, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Calcular el área del contorno
            area = cv2.contourArea(contour)
            print(f"Área del contorno: {area}")
            # Calcular el perímetro del contorno
            perimeter = cv2.arcLength(contour, closed=True)
            figura ={
                'coords': (x, y, w, h),
                'area': area,
                'perimeter': perimeter,
            }
            print(f"Perímetro del contorno: {perimeter}")
            if 14 < perimeter**2/area < 15: #12.57
                monedas.append(figura)
            else:
                figura['valor'] = count_dados(img, figura['coords'])
                dados.append(figura)
    return monedas, dados


def classification(monedas):
    # Preparar los datos
    data = np.array([[moneda['area'], moneda['perimeter']] for moneda in monedas])
    # Aplicar k-means para agrupar en 3 categorías
    kmeans = KMeans(n_clusters=3, random_state=42).fit(data)
    labels = kmeans.labels_
    # Asignar los resultados a las monedas
    for i, moneda in enumerate(monedas):
        moneda['cluster'] = int(labels[i])
    # Identificar el tamaño promedio de cada clúster
    cluster_sizes = []
    for cluster in range(3):
        monedas_cluster = [moneda for moneda in monedas if moneda['cluster'] == cluster]
        promedio_area = np.mean([moneda['area'] for moneda in monedas_cluster])
        cluster_sizes.append((cluster, promedio_area))
    # Ordenar clústeres por tamaño promedio de área
    cluster_sizes.sort(key=lambda x: x[1])  # Orden ascendente por tamaño promedio
    # Asignar valores a los clústeres
    valor_por_cluster = {
        cluster_sizes[0][0]: '10 centavos',  # Clúster más pequeño
        cluster_sizes[1][0]: '1 peso',      # Clúster mediano
        cluster_sizes[2][0]: '50 centavos'  # Clúster más grande
    }
    # Agregar el valor a cada moneda
    for moneda in monedas:
        moneda['valor'] = valor_por_cluster[moneda['cluster']]
    # # Imprimir los resultados
    # for moneda in monedas:
    #     print(f"Área: {moneda['area']:.2f}, Perímetro: {moneda['perimeter']:.2f}, Valor: {moneda['valor']}")

def show_results():
    # Definir los colores normalizados (0-1) para matplotlib
    colores = {
        '10 centavos': (0, 0, 255/255),  # Rojo
        '1 peso': (0, 255/255, 0),      # Verde
        '50 centavos': (255/255, 0, 0), # Azul
        'dados': (0, 140/255, 140/255)      # Naranja
    }
    # Copiar la imagen original para dibujar
    img_result = img_color.copy()
    # Crear la figura y los ejes para mostrar la imagen
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_result)
    # Dibujar los rectángulos sobre las monedas
    for moneda in monedas:
        x, y, w, h = moneda['coords']
        valor = moneda['valor']
        color = colores[valor]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 15, valor, fontsize=12, color=color, weight='bold')
    # Dibujar los rectángulos para los dados
    for dado in dados:
        x, y, w, h = dado['coords']
        valor = dado['valor']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=colores['dados'], facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 15, "Dado: "+str(valor), fontsize=12, color=colores['dados'], weight='bold')
    # Agregar la leyenda
    leyenda_pos = (20, 50)  # Posición inicial de la leyenda
    linea_espaciado = 150    # Espaciado entre líneas de la leyenda
    for i, (valor, color) in enumerate(colores.items()):
        # Cuadro de color para leyenda
        ax.add_patch(patches.Rectangle((leyenda_pos[0], leyenda_pos[1] + i * linea_espaciado),
            20, 20, linewidth=2, edgecolor=color, facecolor=color))
        # Agregar texto a la leyenda
        ax.text(leyenda_pos[0] + 30, leyenda_pos[1] + i * linea_espaciado + 10, valor, fontsize=12, color=color)
    ax.axis('off')
    plt.show(block=False)


# ---------------------------------------------------------------------------------------
# --- Programa Principal ----------------------------------------------------------------
# ---------------------------------------------------------------------------------------
image = cv2.imread('monedas.jpg')
img_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)#cv2.imread('monedas.jpg',cv2.IMREAD_GRAYSCALE)
# imshow(img)
img_filled = preprocess_image(img)
monedas, dados = detect_components(img_filled)
classification(monedas)
show_results()
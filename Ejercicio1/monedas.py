import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import KMeans

#-------------------
# Funciones
#-------------------
'''
Muestra imágenes por pantalla.
'''
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

'''
Reconstrucción Morgológica.
'''
def imreconstruct(marker: np.ndarray, mask: np.ndarray, kernel=None)-> np.ndarray:
    if kernel==None:
        kernel = np.ones((3,3), np.uint8)
    while True:
        expanded = cv2.dilate(marker, kernel)                               # Dilatacion
        expanded_intersection = cv2.bitwise_and(src1=expanded, src2=mask)   # Interseccion
        if (marker == expanded_intersection).all():                         # Finalizacion?
            break                                                           #
        marker = expanded_intersection
    return expanded_intersection

'''
Version 1
Utilizando reconstrucción morfológica
NO rellena los huecos que tocan los bordes
'''
def imfillhole(img: np.ndarray)-> np.ndarray:
    # img: Imagen binaria de entrada. Valores permitidos: 0 (False), 255 (True).
    mask = np.zeros_like(img)                                                   # Genero mascara para...
    mask = cv2.copyMakeBorder(mask[1:-1,1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255)) # ... seleccionar los bordes.
    marker = cv2.bitwise_not(img, mask=mask)                # El marcador lo defino como el complemento de los bordes.
    img_c = cv2.bitwise_not(img)                            # La mascara la defino como el complemento de la imagen.
    img_r = imreconstruct(marker=marker, mask=img_c)        # Calculo la reconstrucción R_{f^c}(f_m)
    img_fh = cv2.bitwise_not(img_r)                         # La imagen con sus huecos rellenos es igual al complemento de la reconstruccion.
    return img_fh

'''
Version 2
Utilizando cv2.floodFill()
SI rellena los huecos que tocan los bordes
'''
def imfillhole_v2(img: np.ndarray)-> np.ndarray:
    img_flood_fill = img.copy().astype("uint8")             # Genero la imagen de salida
    h, w = img.shape[:2]                                    # Genero una máscara necesaria para cv2.floodFill()
    mask = np.zeros((h+2, w+2), np.uint8)                   # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#floodfill
    cv2.floodFill(img_flood_fill, mask, (0,0), 255)         # Relleno o inundo la imagen.
    img_flood_fill_inv = cv2.bitwise_not(img_flood_fill)    # Tomo el complemento de la imagen inundada --> Obtenog SOLO los huecos rellenos.
    img_fh = img | img_flood_fill_inv                       # La salida es un OR entre la imagen original y los huecos rellenos.
    return img_fh

'''
Pre-procesa una imagen a escala de grises.
Técnicas:
Filtro Mediana - Canny
Dilatación - Clausura
Rellenar Huecos - Apertura
'''
def preprocess_image(img: np.ndarray)-> np.ndarray:
    # Filtro de Mediana
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
    k3 = 121
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k3, k3))
    filled_open = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel3)
    # imshow(filled_open)
    # show_preprocessing_results(blurred, edges, dilated, closed, filled, filled_open)
    return filled_open

'''
Muestra las etapas de pre-procesamiento.
'''
def show_preprocessing_results(blurred, edges, dilated, closed, filled, filled_open)-> None:
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    axs = axs.ravel()  # Convertir en un arreglo para iterar fácilmente
    images = [blurred, edges, dilated, closed, filled, filled_open]
    titles = [
        "Filtro de Mediana",
        "Canny",
        "Dilatación",
        "Clausura",
        "Rellenar Huecos",
        "Apertura"
    ]
    for i in range(6):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].set_title(titles[i])
        axs[i].axis('off')  
    plt.tight_layout()
    plt.show(block=False)

'''
Cuenta el puntaje de un dado.
Técnica: HoughCircles
'''
def count_dados(img: np.ndarray, coords: tuple) -> int:
    x, y, w, h = coords
    crop_img = img[y:y+h, x:x+w]
    # Detectar círculos usando HoughCircles
    # circles = cv2.HoughCircles(crop_img,cv2.HOUGH_GRADIENT, 1, 10, param1=50, param2=40, minRadius=20, maxRadius=30)
    circles = cv2.HoughCircles(
        crop_img,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10,
        param1=50,
        param2=40,
        minRadius=20,
        maxRadius=30
    )
    # Dibujar los círculos detectados en la imagen
    # output_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)  # Convertir a color para dibujar círculos
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for circle in circles[0, :]:
    #         center = (circle[0], circle[1])  # Coordenadas del centro
    #         radius = circle[2]              # Radio del círculo
    #         print(f"Dibujando círculo en {center} con radio {radius}.")
    #         cv2.circle(output_img, center, radius, (0, 255, 0), 2)  # Dibujar contorno
    #         cv2.circle(output_img, center, 2, (0, 0, 255), 3)       # Dibujar centro
    # plt.figure(figsize=(8, 6))
    # plt.title("Área de los Dados con Círculos Detectados")
    # plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))  # Convertir a RGB para matplotlib
    # plt.axis("off")
    # plt.show(block=False)
    return circles.shape[1] if circles is not None else 0

'''
Detecta los Componentes Conectados en la imagen pre-procesada.
Técnica: Componentes Conectados - Contornos
'''
def detect_components(img_gray: np.ndarray, img: np.ndarray)-> tuple:
    # img_seg = img_gray.copy()
    connectivity = 8
    # Encontrar los componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
    monedas = []
    dados = []
    # Iterar sobre los componentes
    for i in range(1, num_labels):  # Ignorar el componente 0 (fondo)
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        obj = (labels == i).astype(np.uint8)
        contours, hierarchy = cv2.findContours(obj, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Calcular el área del contorno
            area = cv2.contourArea(contour)
            print(f"Área del contorno: {area}")
            # Calcular el perímetro del contorno
            perimeter = cv2.arcLength(contour, closed=True)
            print(f"Perímetro del contorno: {perimeter}")
            # Detectar círculos. Relación: P^2 /A = 12.57 (da un nro mayor)
            if 14 < perimeter**2/area < 15: #12.57
                moneda = {
                'coords': (x, y, w, h),
                'area': area,
                'perimeter': perimeter,
                'value': "",
                'cluster': 0,
                }
                monedas.append(moneda)
                # cv2.rectangle(img_seg, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                dado = {
                'coords': (x, y, w, h),
                'value': 0,
                }
                dado['value'] = count_dados(img_gray, dado['coords'])
                dados.append(dado)
                # cv2.rectangle(img_seg, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Mostrar la imagen segmentada con los bounding boxes
    # imshow(img_seg)
    return monedas, dados

'''
Clasifica las monedas por sus dimensiones.
Técnica: KMeans
'''
def classification(monedas: list)-> None:
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
        cluster_sizes[0][0]: '10 Cts.', # Clúster más pequeño
        cluster_sizes[1][0]: '1 Peso',  # Clúster mediano
        cluster_sizes[2][0]: '50 Cts.'  # Clúster más grande
    }
    # Agregar el valor a cada moneda
    for moneda in monedas:
        moneda['value'] = valor_por_cluster[moneda['cluster']]

    # # Imprimir los resultados
    # for moneda in monedas:
    #     print(f"Área: {moneda['area']:.2f}, Perímetro: {moneda['perimeter']:.2f}, Valor: {moneda['value']}")

    # Visualizar los clústeres
    # plt.figure(figsize=(10, 6))
    # # Colores para los clústeres
    # colors = ['red', 'green', 'blue']
    # for cluster in range(3):
    #     cluster_points = data[labels == cluster]
    #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
    #                 color=colors[cluster], label=f"Cluster {cluster} ({valor_por_cluster[cluster]})")
    # # Mostrar centros de los clústeres
    # centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='yellow', marker='x', s=200, label='Centros de clúster')
    # # Etiquetas y leyenda
    # plt.title("Clusters de Monedas según Área y Perímetro")
    # plt.xlabel("Área")
    # plt.ylabel("Perímetro")
    # plt.legend()
    # plt.grid(True)
    # plt.show(block=False)

'''
Muestra las monedas y dados clasificados por colores.
'''
def show_results(monedas: list, dados: list)-> None:
    # Definir los colores normalizados (0-1) para matplotlib
    colores = {
        '10 Cts.': (0, 0, 255/255),     # Rojo
        '1 Peso': (0, 255/255, 0),      # Verde
        '50 Cts.': (255/255, 0, 0),     # Azul
        'Dados': (255/255, 255/255, 0)  # Amarillo
    }
    # Copiar la imagen original para dibujar
    img_result = img_color.copy()
    # Crear la figura y los ejes para mostrar la imagen
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_result)
    # Dibujar los rectángulos sobre las monedas
    for moneda in monedas:
        x, y, w, h = moneda['coords']
        value = moneda['value']
        color = colores[value]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 15, str(value), fontsize=12, color=color, weight='bold')
    # Dibujar los rectángulos para los dados
    for dado in dados:
        x, y, w, h = dado['coords']
        value = dado['value']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=colores['Dados'], facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 15, "Dado: "+str(value), fontsize=12, color=colores['Dados'], weight='bold')
    # Agregar la leyenda
    leyenda_pos = (20, 50)  # Posición inicial de la leyenda
    linea_espaciado = 150    # Espaciado entre líneas de la leyenda
    for i, (value, color) in enumerate(colores.items()):
        # Cuadro de color para leyenda
        ax.add_patch(patches.Rectangle((leyenda_pos[0], leyenda_pos[1] + i * linea_espaciado),
            20, 20, linewidth=2, edgecolor=color, facecolor=color))
        # Agregar texto a la leyenda
        ax.text(leyenda_pos[0] + 30, leyenda_pos[1] + i * linea_espaciado + 10, str(value), fontsize=12, color=color)
    ax.axis('off')
    plt.show(block=False)


#-------------------
# Programa Principal
#-------------------
'''
Lee el archivo.
Muestra el resultado.
'''
image = cv2.imread('monedas.jpg')
# imshow(image)
img_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# imshow(img_color)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
# imshow(img)
img_filled = preprocess_image(img_gray)
# imshow(img_filled)
monedas, dados = detect_components(img_gray, img_filled)
classification(monedas)
show_results(monedas, dados)
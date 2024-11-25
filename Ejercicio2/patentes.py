import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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
Binariza la imagen original(en escala de grises) con un umbral.
Aplica una transformación Top Hat a la imagen original para lograr más nitidez.
Devuelve la intersección entre ambas imágenes binarias.
'''
# En la imagen 03 el caracter toca el borde inferior izquierdo después de la binarización
# Si se aumenta el umbral se dincontinúa la letra D
# La imagen resultante de la transformación Top Hat no toca ese borde, pero los caracteres se engrosan
# y tocan en otros lugares el borde.
# la intersección de las dos imágenes soluciona el problema
# y mejora las demás imágenes, por lo que se aplica a todas por igual
def preprocess_image(img: np.ndarray, umbral: int)-> np.ndarray:
    # imagen binaria
    th, binary_img = cv2.threshold(img.astype(np.uint8), umbral, 1, cv2.THRESH_BINARY)
    # imshow(binary_img)
    # top hat
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat_img = cv2.morphologyEx(img, kernel=se, op=cv2.MORPH_TOPHAT)
    # binaria de top hat
    th1, binary_tophat_img = cv2.threshold(tophat_img.astype(np.uint8), 55, 1, cv2.THRESH_BINARY)
    # # Binaria de Top Hat con Otsu
    # _, binary_tophat_img = cv2.threshold(
    #     tophat_img.astype(np.uint8),
    #     0,  # Ignorado con THRESH_OTSU
    #     1,  # Valor máximo del píxel
    #     cv2.THRESH_BINARY + cv2.THRESH_OTSU
    # )
    # Interseccion
    intersection = np.bitwise_and(binary_img, binary_tophat_img)
    # show_preprocessing_results(binary_img, tophat_img, binary_tophat_img, intersection)
    return intersection

'''
Muestra las etapas de pre-procesamiento.
'''
def show_preprocessing_results(binary_img, tophat_img, binary_tophat_img, intersection)-> None:
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axs = axs.ravel()  # Convertir en un arreglo para iterar fácilmente
    images = [binary_img, tophat_img, binary_tophat_img, intersection]
    titles = [
        "Original Binaria",
        "Top Hat",
        "Top Hat Binaria",
        "Intersección"
    ]
    for i in range(4):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].set_title(titles[i])
        axs[i].axis('off')
    plt.tight_layout()
    plt.show(block=False)

'''
Filtra de la imagen todos los componentes conectados que cuyas áreas están fuera de los límites establecidos
y que no tengan una relación de aspecto largo/ancho entre 1,5 y 3 (formato de los caracteres).
'''
def filter_area_aspect(img: np.ndarray)-> np.ndarray:
    connectivity = 8
    min_area = 16
    max_area = 200
    # Encontrar los componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
    # Crear una imagen de salida para los componentes que cumplen con el área mínima y relación de aspecto
    output_img = np.zeros_like(img)
    # Iterar sobre los componentes y procesar solo los que cumplen el área mínima y relación de aspecto
    for i in range(1, num_labels):  # Ignorar el componente 0 (fondo)
        area = stats[i, cv2.CC_STAT_AREA]
        if (area > min_area) & (area < max_area):
            h, w = stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_WIDTH]
            h_w_ratio = h / w
            # if 1.5 <= h_w_ratio <= 3:
            if 1.5 <= h_w_ratio <= 3:
                output_img[labels == i] = 255
    # imshow(output_img)
    return output_img

'''
Busca dentro de la imagen una secuencia de un grupo de 6 caracteres,
separados en dos grupos de 3 caracteres cada uno.
Patrón buscado XXX XXX
Previamente filtra los componentes cuya coordenada y está alejada de los del resto del grupo.
'''
def detect_patente(img: np.ndarray) -> tuple:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8, ltype=cv2.CV_32S)
    # Filtrar componentes válidos y calcular alturas
    possible_characters = []
    heights = []
    for i in range(1, num_labels):  # Ignorar el fondo
        x, y, w, h, area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        possible_characters.append((x, y, w, h, centroids[i]))
        heights.append(h)
    # Ordenar caracteres por coordenada X (izquierda a derecha)
    possible_characters.sort(key=lambda char: char[0])
    # Calcular promedios de las alturas de los caracteres
    mean_height = np.mean(heights) if heights else 0
    # Valores dinámicos basados en la altura promedio y las relaciones de distancia del formato real
    max_distance_y = mean_height #altura promedio
    # El espaciado entre grupos es aprox. la mitad de la altura de los caracteres (0.5*altura)
    # Se le da un margen mayor porque mean_height incluye otros componentes espurios.
    e_min = mean_height * 0.4
    e_max = mean_height * 1.6
    c_min = e_min/2
    c_max = e_max/2
    # print(e_min,e_max,max_distance_y)
    # Crear grupos iniciales por proximidad en Y
    groups = []
    for character in possible_characters:
        appended = False
        for group in groups:
            if all(abs(character[1] - member[1]) <= max_distance_y for member in group):
                group.append(character)
                appended = True
                break
        if not appended:
            groups.append([character])
    # Buscar el patrón `XXX XXX`
    for group in groups:
        if len(group) < 6:
            continue
        for i in range(len(group) - 5):
            subgroup = group[i:i + 6]
            distances = [subgroup[j + 1][0] - subgroup[j][0] for j in range(5)]  # Distancias entre centros
            # alturas = [char[1] for char in subgroup]
            # print(distances)
            # print(alturas)
            if all(c_min <= dist <= c_max for dist in distances[:2] + distances[3:]) and \
                e_min <= distances[2] <= e_max:
                x_start = subgroup[0][0]
                x_end = subgroup[-1][0] + subgroup[-1][2]
                y_start = min([char[1] for char in subgroup])
                y_end = max([char[1] + char[3] for char in subgroup])
                return x_start, x_end, y_start, y_end, subgroup
    return None

'''
Muestra la imagen original con la patente recuadrada y
un crop de la zona de la patente con los carateres recuadrados.
'''
def show_results(img: np.ndarray, patente: dict, margin: int)-> None:
    img_with_patente = img.copy()
    # Definir las coordenadas de la patente con margen
    x_start, y_start, x_end, y_end = patente["x_start"], patente["y_start"], patente["x_end"], patente["y_end"]
    x_start_margin = max(0, x_start - margin)
    y_start_margin = max(0, y_start - margin)
    x_end_margin = x_end + margin
    y_end_margin = y_end + margin
    # Dibujar el rectángulo alrededor de la patente con el margen
    cv2.rectangle(img_with_patente, (x_start_margin, y_start_margin), (x_end_margin, y_end_margin), (0, 0, 255), 2)
    # Recortar la zona de la patente con margen
    img_cropped = img[y_start_margin:y_end_margin, x_start_margin:x_end_margin].copy()
    # Dibujar rectángulos alrededor de los caracteres en la zona recortada
    for char in patente["characters"]:
        char_x, char_y, char_w, char_h = char[0] - x_start_margin, char[1] - y_start_margin, char[2], char[3]
        cv2.rectangle(img_cropped, (char_x, char_y), (char_x + char_w, char_y + char_h), (0, 0, 255), 1)
    # Crear la figura con subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.sca(axes[0])
    imshow(img_with_patente, new_fig=False, title="Patente en la Imagen", color_img=True, ticks=False)
    plt.sca(axes[1])
    imshow(img_cropped, new_fig=False, title="Caracteres en la Patente", color_img=True, ticks=False)
    plt.tight_layout()
    plt.show(block=False)

'''
Realiza un proceso iterativo con distintos umbrales de binarización
hasta obtener una imagen donde puede reconocerse el patrón XXX XXX de una patente
'''
def identify_patente(img: np.ndarray)-> dict:
    # Realiza el proceso para distintos umbrales hasta que encuentra el patrón XXX XXX y sale
    for umbral in range(110,135,2):
        img_cleaned = preprocess_image(img, umbral)
        img_filtered = filter_area_aspect(img_cleaned)
        result = detect_patente(img_filtered)
        # Visualizar el resultado
        if result is not None:
            x_start, x_end, y_start, y_end, characters = result
            # Almacenar los datos en el diccionario
            detected_patente = {
                "x_start": x_start,
                "x_end": x_end,
                "y_start": y_start,
                "y_end": y_end,
                "characters": characters
            }
            print(umbral)
            return detected_patente
    return None

#-------------------
# Programa Principal
#-------------------
'''
Identifica una patente
Lee un archivos .png de la carpeta patentes.
Muestra los resultados.
'''
img = cv2.imread("Patentes\img02.png")
img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
detected_patente = identify_patente(img_gray)
if detected_patente is not None:
    show_results(img_color, detected_patente, 5)


'''
Identifica todas las patentes en la carpeta "patentes"
Lee los archivos .png de la carpeta patentes.
Muestra los resultados.
'''
path_patentes = 'Patentes'
files = [os.path.join(path_patentes, f) for f in os.listdir(path_patentes) if f.endswith('.png')]
# Diccionario para almacenar la información de las patentes detectadas
detected_patentes = []
for file in files:
    img = cv2.imread(file)
    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_patente = identify_patente(img_gray)
    show_results(img_color, detected_patente, 5)

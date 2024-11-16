import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#-------------------
# Funciones
#-------------------

'''
Muestra imágenes por pantalla
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
Binariza la imagen original(en escala de grises) con un umbral
Aplica una transformación Top Hat a la imagen original para lograr más nitidez
Devuelve la intersección entre ambas imágenes binarias
'''
# En la imagen 04  el caracter toca el borde inferior izquierdo después de la binarización
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
    g3 = cv2.morphologyEx(img, kernel=se, op=cv2.MORPH_TOPHAT)
    # binaria de top hat
    th1, binary_img1 = cv2.threshold(g3.astype(np.uint8), 53, 1, cv2.THRESH_BINARY)
    # Interseccion
    intersection = np.bitwise_and(binary_img, binary_img1)
    return intersection

'''
Filtra de la imagen todos los componentes conectados que cuyas áreas están fuera de los límites establecidos
y que no tengan una relación de aspecto largo/ancho entre 1,5 y 3 (formato de los caracteres)
'''
def filter_area_aspect(img: np.ndarray)-> np.ndarray:
    connectivity = 8
    min_area = 16
    max_area = 100
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
            if 1.5 <= h_w_ratio <= 3:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                output_img[labels == i] = 255
    return output_img

'''
Busca dentro de la imagen una secuencia de un grupo de 6 caracteres,
separados en dos grupos de 3 caracteres cada uno.
Patrón buscado XXX XXX
Previamente filtra los componentes cuya coordenada y está alejada de los del resto del grupo
'''
def detect_patente(img: np.ndarray)-> tuple:
    c_min = 5
    c_max = 16
    e_min = 10
    e_max = 30
    max_distance_y = 15  # Diferencia máxima permitida en y para formar un grupo
    # Aplicar Connected Components para encontrar posibles caracteres
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8, ltype=cv2.CV_32S)
    # Filtrar componentes que parezcan caracteres según su relación de aspecto
    possible_characters = []
    for i in range(1, num_labels):  # Empezamos en 1 para ignorar el fondo
        h, w = stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_WIDTH]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        possible_characters.append((x, y, w, h, centroids[i]))
    # Ordenar los posibles caracteres de izquierda a derecha
    possible_characters.sort()
    # Crear grupos iniciales por proximidad en y
    groups = []
    for character in possible_characters:
        appended = False
        for group in groups:
            # antes de agregar un caracter al grupo verifica que su coordenada y no esté
            # alejada de los miembros del grupo una distancia mayor a max_distance_y (15)
            if all(abs(character[1] - member[1]) <= max_distance_y for member in group):
                group.append(character)
                appended = True
                break
        if not appended:
            groups.append([character])
    # Buscar el patrón `XXX XXX` en cada grupo formado
    for group in groups:
        # Asegurarse de que el grupo tenga al menos 6 caracteres para buscar el patrón
        if len(group) < 6:
            continue
        # Intentar encontrar el patrón dentro del grupo
        # si por ejemplo el grupo tiene 10 componentes, armará los siguientes subgrupos en las 10-5 iteraciones
        # iteracion 0: 012345
        # iteracion 1: 123456
        # iteracion 2: 234567
        # iteracion 3: 345678
        # iteracion 4: 456789
        # en la iteración que encuentra el patrón retorna (x_start, x_end, y_start, y_end, subgroup)
        for i in range(len(group) - 5):
            subgroup = group[i:i + 6]
            distances = [subgroup[j + 1][0] - subgroup[j][0] for j in range(5)]  # Distances entre centros de caracteres
            alturas = [character[1] for character in subgroup]
            # Verificar si cumplen con el patrón de la patente
            if all(c_min <= dist <= c_max for dist in distances[:2] + distances[3:]) and \
            e_min <= distances[2] <= e_max and (max(alturas) - min(alturas)) < max_distance_y:
                # Obtener los límites de la patente
                x_start = subgroup[0][0]
                x_end = subgroup[-1][0] + subgroup[-1][2]  # x + ancho del último componente
                # Calcular el punto más alto y más bajo en el conjunto de caracteres
                y_start = min([character[1] for character in subgroup])  # y mínimo
                y_end = max([character[1] + character[3] for character in subgroup])  # y + altura del componente
                return x_start, x_end, y_start, y_end, subgroup
    return None

'''
Muestra la imagen original con la patente recuadrada y
un crop de la zona de la patente con los carateres recuadrados
'''
def show_results(img: np.ndarray, margin: int)-> None:
    img_with_patente = img.copy()
    # Definir las coordenadas de la patente con margen
    x_start, y_start, x_end, y_end = patente["x_start"], patente["y_start"], patente["x_end"], patente["y_end"]
    x_start_margin = max(0, x_start - margin)
    y_start_margin = max(0, y_start - margin)
    x_end_margin = x_end + margin
    y_end_margin = y_end + margin
    # Dibujar el rectángulo alrededor de la patente con el margen
    cv2.rectangle(img_with_patente, (x_start_margin, y_start_margin), (x_end_margin, y_end_margin), (255, 0, 0), 2)
    # Recortar la zona de la patente con margen
    img_cropped = img[y_start_margin:y_end_margin, x_start_margin:x_end_margin].copy()
    # Dibujar rectángulos alrededor de los caracteres en la zona recortada
    for char in patente["characters"]:
        char_x, char_y, char_w, char_h = char[0] - x_start_margin, char[1] - y_start_margin, char[2], char[3]
        cv2.rectangle(img_cropped, (char_x, char_y), (char_x + char_w, char_y + char_h), (255, 0, 0), 1)
    # Convertir las imágenes a formato RGB para Matplotlib
    img_with_patente_rgb = cv2.cvtColor(img_with_patente, cv2.COLOR_BGR2RGB)
    img_cropped_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
    # Crear subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_with_patente_rgb)
    axes[0].set_title("Area de la Patente en la Imagen")
    axes[0].axis("off")
    axes[1].imshow(img_cropped_rgb)
    axes[1].set_title("Caracteres en la Patente")
    axes[1].axis("off")
    # Mostrar los subplots
    plt.tight_layout()
    plt.show(block=False)
    # plt.figure()
    # # plt.title = "Detección de patentes"
    # ax = plt.subplot(121), imshow(img_with_patente_rgb, new_fig=False, title="Patente en la Imagen", colorbar=False)
    # plt.subplot(122), imshow(img_cropped_rgb, new_fig=False, title="Caracteres en la Patente", colorbar=False)
    # plt.show(block=False)

'''
Programa principal
Lee los archivos .png de la carpeta patentes.
Los resultados se guardan en una lista de diccionarios con información de cada patente.
Se muestran los resultados
'''
path_patentes = 'patentes'
patentes = [os.path.join(path_patentes, f) for f in os.listdir(path_patentes) if f.endswith('.png')]
# Diccionario para almacenar la información de las patentes detectadas
detected_patentes = []
for patente in patentes:
    img_color  = cv2.imread(patente)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_patente = img_color.copy()
    for umbral in range(110,151):
        img_cleaned = preprocess_image(img_gray, umbral)
        img_filtered = filter_area_aspect(img_cleaned)
        result = detect_patente(img_filtered)
        # Visualizar el resultado
        if result is not None:
            x_start, x_end, y_start, y_end, characters = result
            # print(f"Patente detectada en el área: ({x_start}, {y_start}) a ({x_end}, {y_end})")
            # Dibujar una caja alrededor de la patente en la imagen
            # cv2.rectangle(img_patente, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            # imshow(img_patente)
            # Almacenar los datos en el diccionario
            detected_patente = {
                "file": patente,
                "x_start": x_start,
                "x_end": x_end,
                "y_start": y_start,
                "y_end": y_end,
                "characters": characters
            }
            detected_patentes.append(detected_patente)
            break

for patente in detected_patentes:
    img = cv2.imread(patente["file"])
    show_results(img, 5)

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

""" 
# OTSU NO FUNCIONA --> QUISIMOS AUTOMATIZAR LA BUSQUEDA DEL UMBRAL, además realiza EQUALIZACION .
def preprocess_image_with_otsu(img: np.ndarray) -> np.ndarray:
    # ECUALIZACION DEL HISTOGRAMA
    img= cv2.equalizeHist(img)
    # Aplicar Otsu para encontrar el umbral óptimo
    _, binary_img = cv2.threshold(img.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #método cv2.THRESH_OTSU le dice a OpenCV que utilice el algoritmo de Otsu para calcular automáticamente el umbral óptimo.
    
    # Aplicar la operación Top-Hat con el elemento estructurante elíptico
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat_img = cv2.morphologyEx(img, kernel=se, op=cv2.MORPH_TOPHAT)
    
    # Aplicar Otsu también en la imagen Top-Hat para obtener el umbral óptimo
    _, binary_tophat_img = cv2.threshold(tophat_img.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Intersección de las dos imágenes binarias obtenidas
    intersection = np.bitwise_and(binary_img, binary_tophat_img)
    
    return intersection
"""
def preprocess_image(img: np.ndarray, umbral: int)-> np.ndarray:
    # imagen binaria
    th, binary_img = cv2.threshold(img.astype(np.uint8), umbral, 1, cv2.THRESH_BINARY)
    # imshow(binary_img)
    # top hat
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat_img = cv2.morphologyEx(img, kernel=se, op=cv2.MORPH_TOPHAT)
    # binaria de top hat
    th1, binary_tophat_img = cv2.threshold(tophat_img.astype(np.uint8), 53, 1, cv2.THRESH_BINARY)
    # Interseccion
    intersection = np.bitwise_and(binary_img, binary_tophat_img)
    # show_preprocessing_results(binary_img, tophat_img, binary_tophat_img, intersection)
    return intersection

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
    plt.show()


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
                output_img[labels == i] = 255
    return output_img


def detect_patente(img: np.ndarray)-> tuple:
    c_min = 5   # mínima distancia entre caracteres de un grupo
    c_max = 16  # máxima distancia entre caracteres de un grupo
    e_min = 10  # mínima distancia entre los dos grupos
    e_max = 30  # máxima distancia entre los dos grupos
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
            # Antes de agregar un caracter al grupo verifica que su coordenada "y" no esté
            # alejada de los miembros del grupo una distancia mayor a max_distance_y (15 pixeles)
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


def show_results(img: np.ndarray, patente: dict, margin: int)-> None: # entradas: imagen donde detecto la patente, 
    img_with_patente = img.copy()                                     # patente: Un diccionario con información sobre la patente detectada. Este diccionario contiene las coordenadas del área donde se encuentra la patente y la lista de caracteres detectados dentro de esa área.
    # Definir las coordenadas de la patente con margen                # margen valor entero
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
    # Crear subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_with_patente)
    axes[0].set_title("Area de la Patente en la Imagen")
    axes[0].axis("off")
    axes[1].imshow(img_cropped)
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

def identify_patente(img):
    # Realiza el proceso para distintos umbrales hasta que encuentra el patrón XXX XXX y sale
    for umbral in range(114,151):
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
            return detected_patente
"""
# otsu no funciona, no usar.
def identify_patente(img_gray):
    intersection= preprocess_image_with_otsu(img_gray)
    
    # Aplicar el umbral de Otsu automáticamente
    _, img_cleaned = cv2.threshold(intersection, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Continuar con el resto del proceso
    img_filtered = filter_area_aspect(img_cleaned)
    result = detect_patente(img_filtered)

    if result is not None:
        x_start, x_end, y_start, y_end, characters = result
        detected_patente = {
            "x_start": x_start,
            "x_end": x_end,
            "y_start": y_start,
            "y_end": y_end,
            "characters": characters
        }
        return detected_patente
"""
#-------------------------------------------------------------------------------------------------------------------
# leo imagen de ruta absoluta
img= cv2.imread(f"C:\\Users\\Usuario\\Documents\\TECNICATURA EN IA\\PROCESAMIENTO DE IMAGENES\\PDI_2024\\TP2\\Ejercicio2\\Patentes\\img01.png")
# Leo imagenes de ruta relativa
#img = cv2.imread("Ejercicio2\\Patentes\\img01.png")
#img = cv2.imread("patentes\img01.png")
img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       

# PRE-PROCESAMEINTO -------------------------------------------------------------------------------
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


"""
# cv2.threshold(src, thresh, maxval, type)
    src: La imagen de entrada (en este caso, img).
    thresh: El valor de umbral para la binarización.
    maxval: El valor máximo que se asignará a los píxeles superiores al umbral.
    type: El tipo de umbralización (en tu caso, cv2.THRESH_BINARY + cv2.THRESH_OTSU).
    Cuando usas cv2.THRESH_BINARY + cv2.THRESH_OTSU, el umbral se determina automáticamente, pero aún necesitas 
    proporcionar un valor para maxval, que es el valor que se asigna a los píxeles que superen el umbral.
"""
# imagen binaria = Aplica la umbralización de Otsu con el valor máximo establecido en 255
th, binary_img = cv2.threshold(img.astype(np.uint8), thresh= 0, maxval = 255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# imshow(binary_img)         # Recordar pasar imagen en escala de grisis a threshold
                            # OpenCV que use el método de Otsu para det. el umbral automáticamente, junto con la umbralización binaria.
th  # 128 equalizada  
    # 114 si no hago equalizada. En este caso modificar el nombre de la imagen que le paso a threshold por img_gray

# top hat   -----------------------------------------------------------------------
"""
La transformada Top-Hat es una operación morfológica que destaca los detalles brillantes (pequeñas características) de 
una imagen respecto a un fondo más oscuro. Se utiliza principalmente para mejorar el contraste y resaltar características
 de interés en las imágenes.

Transformada Top-Hat (T-Hat) se define como la diferencia entre la imagen original y su erosión seguida de dilatación
(en términos de operaciones morfológicas). Este proceso permite resaltar las pequeñas estructuras brillantes sobre un 
fondo más oscuro.

*Fórmula de la Transformada Top-Hat" --> Top-Hat= I(x,y) - Opening(I(x,y))
                                            I(x,y) es la imagen original.
                                                        Opening es una operación morfológica = primero erosión y luego dilatación. 
        La apertura elimina los detalles pequeños y resalta las características más grandes.

*Usos de la Transformada Top-Hat    
        Extracción de detalles pequeños: Destaca detalles brillantes que son pequeños y que pueden ser difíciles de ver sobre un fondo oscuro o uniforme.
        Mejora de contraste: Ayuda a aumentar el contraste entre objetos pequeños brillantes y el fondo de la imagen.
        Eliminación de ruido: Al restar la imagen con apertura (que elimina las características pequeñas), la transformada Top-Hat elimina el ruido que podría estar presente.
        Análisis de textura:  extracción de características texturales, en especial cuando los objetos de interés tienen formas pequeñas o texturas finas.
        Segmentación de objetos: En imágenes donde el objeto de interés tiene una característica pequeña y brillante, la transformada Top-Hat es útil para segmentarlo.

*Implementación en OpenCV --> función cv2.morphologyEx() con el tipo de operación cv2.MORPH_TOPHAT.

"""  
# Transformada Top-Hat: Resalta las características brillantes pequeñas sobre un fondo oscuro.     
# Definir el kernel de la operación morfológica (puedes experimentar con su tamaño)  

# Define el tamaño del kernel como una tupla (ancho, alto)
kernel_size = (7, 7)  # Un kernel 7 x 7  para aplicar a las operaciones de erosión y dilataci        
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size) # elemento estructurante elíptico
tophat_img = cv2.morphologyEx(img, kernel=se, op=cv2.MORPH_TOPHAT) # NO ES BINARIA

"""
TOP-HAT : No devuelve una imagen binaria por defecto.
    Devuelve una imagen de escala de grises (si la imagen original es en escala de grises), donde los detalles pequeños
            brillantes están resaltados. 
   Estas áreas brillantes se corresponden con las diferencias entre la imagen original y su imagen después de la 
   apertura/OPENING = EROSION  + DILATAR.
   
   TOP HAT = IMAGEN ORIGINAL - OPENING
"""

# Calcular binaria de top hat
#th1, binary_tophat_img = cv2.threshold(tophat_img.astype(np.uint8), 53, 1, cv2.THRESH_BINARY)
th1, binary_tophat_img = cv2.threshold(tophat_img.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#th1   # 35 umbral
"""
Para usar Otsu's Thresholding en lugar de un valor de umbral fijo (hardcodeado) en la función cv2.threshold(), solo necesitas modificar el argumento que corresponde 
al umbral, y en lugar de pasar un valor específico como 53, debes usar el tipo de umbral cv2.THRESH_OTSU.

El método de Otsu calcula automáticamente el umbral óptimo basado en la histograma de la imagen, lo que lo hace muy útil 
para imágenes con una distribución bimodal (dos picos)."""

# Interseccion
intersection = np.bitwise_and(binary_img, binary_tophat_img)
"""
La función np.bitwise_and() realiza una operación AND a nivel de bits entre dos matrices (en este caso, dos imágenes),
pixel por pixel. 
    resultado es 1 (verdadero) solo cuando ambos píxeles de entrada son 1 (blanco), y 0 (falso) en todos los demás casos.

En el contexto de imágenes binarias:
binary_img y binary_tophat_img son imágenes binarias, es decir, las imágenes contienen solo valores 0 (negro) y 255 (blanco).
    255 indica los píxeles que pertenecen a los objetos o características que queremos resaltar.
    0 indica el fondo o las áreas que no son de interés.
Cuando aplicamos np.bitwise_and(binary_img, binary_tophat_img), obtenemos una nueva imagen en la que un píxel 
será 255 (blanco) solo si ambos píxeles correspondientes en binary_img y binary_tophat_img son 255 (es decir, ambos son 
verdaderos). En cualquier otro caso, el resultado será 0 (negro).
"""
#intersection.shape
#(360, 640)
#np.unique(intersection)
#array([  0, 255], dtype=uint8)  BINARIA

 #VISUALIZACION-------------------------------------------------------------------------------

fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
# Se crearán 4 subgráficas en una figura de 12x8 pulgadas, cada una con gráficos diferentes,
# y las escalas de los ejes X e Y serán compartidas entre todas las subgráficas.
"""
    plt.subplots(2, 2):

Crea una rejilla de subgráficas de 2 filas y 2 columnas, 4 subgráficas en total.
Esto crea una matriz de ejes (axs), con una forma de (2, 2), que es una matriz 2x2 donde cada celda contiene un eje 
individual.

    figsize=(12, 8):

Define el tamaño de la figura ( 12 pulgadas de ancho y 8 pulgadas de alto.)

    sharex=True, sharey=True:

sharex=True: Significa que las escuelas del eje X serán compartidas entre todas las subgráficas. 
sharey=True: Significa que las escuelas del eje Y también serán compartidas entre todas las subgráficas. Esto asegura que todas las subgráficas tengan el mismo rango en el eje Y.

    ¿Qué devuelve plt.subplots()?
        fig: Es la figura principal que contiene todas las subgráficas. Es un objeto de tipo matplotlib.figure.Figure.
        axs: Es una matriz de objetos de ejes (matplotlib.axes.Axes). Cada objeto en axs corresponde a un eje individual 
    donde puedes trazar tus gráficos. En este caso, será una matriz de 2x2 de ejes
"""

axs = axs.ravel()  # Convertir en un arreglo para iterar fácilmente
images = [binary_img, tophat_img, binary_tophat_img, intersection]
titles = [
    "Original Binaria",
    "Top Hat",
    "Top Hat Binaria",
    "Intersección"
]
#for i in range(4):
for i in range(len(images)):
    axs[i].imshow(images[i], cmap='gray')  #axs[0, 0], axs[0, 1], etc.: Accedemos a cada uno de los ejes en la matriz axs 
    axs[i].set_title(titles[i])                                # para crear gráficos en las subgráficas específicas. 
    axs[i].axis('off')                     # axs[0, 0] es el eje de la primera subgráfica en la fila 1, columna 1.
plt.tight_layout()
plt.show(block= False)


# COMPONENTES CONECTADAS (análisis de componentes conectados en una imagen binaria.) -------------------------------------------------------------------
'''
se utiliza para identificar y etiquetar regiones o componentes en una imagen que están conectados entre sí (es decir, 
tienen píxeles vecinos con un valor similar). 
El código filtra estos componentes según su área y la relación de aspecto entre su altura y anchura,
para luego guardar los componentes que cumplen con esos criterios en una nueva imagen.

Filtra de la imagen todos los componentes conectados que cuyas áreas están fuera de los límites establecidos
y que no tengan una relación de aspecto alto/ancho entre 1,5 y 3 (formato de los caracteres).
Este código filtra los componentes conectados de la imagen en función de tres criterios:
    Área: El componente debe tener un área dentro de un rango especificado (mayor que min_area y menor que max_area).
    Relación de aspecto: El componente debe tener una relación de aspecto (altura/ancho) que esté en el rango de 1.5 a 3.
    Solo los componentes que pasan ambos filtros (área y relación de aspecto) serán procesados o considerados como "relevantes" para el análisis posterior.
'''
connectivity = 8 
# tipo de conectividad para la búsqueda de componentes conectados.
# Un valor de 8 significa que se consideran todos los píxeles vecinos en un entorno de 8 conectividades (todos los píxeles vecinos, en diagonal o en línea recta). 
# Si usas connectivity = 4, solo considerarías los vecinos en línea recta (arriba, abajo, izquierda, derecha).
min_area = 16  # es en pixeles
max_area = 100
# Son los límites de área que deben cumplir los componentes detectados, Solo se seleccionarán los componentes cuya área esté entre estos dos valores.

# Encontrar los componentes conectados
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(intersection, connectivity, cv2.CV_32S)
                # componentes conectados en una imagen binaria: regiones de píxeles adyacentes que tienen el mismo valor
    #num_labels: El NÚMEROnúmero total de componentes conectados detectados, incluido el fondo.
#num_labels  # 1971 ESCALAR
    #labels: Una MATRIZ de ETIQUETAS donde cada píxel tiene asignado un valor que corresponde al componente al que pertenece (el fondo tiene valor 0).
#labels  # ETIQUETAS
    #stats: Una matriz que contiene información estadística sobre cada componente. Cada fila corresponde a un componente (incluido el fondo). Las columnas contienen información como el área, la posición y las dimensiones del componente (alto, ancho).
    #centroids: Las coordenadas del centroide de cada componente
# Crear una imagen de salida para los componentes que cumplen con el área mínima y relación de aspecto
#stats
output_img = np.zeros_like(img)
# Iterar sobre los componentes CONECTADOS y procesar solo los que cumplen el área mínima y relación de aspecto
for i in range(1, num_labels):  # Ignorar el componente 0 (fondo), por eso arranca en 1 (num_labels es el número total de componentes conectados encontrados)
    area = stats[i, cv2.CC_STAT_AREA] # stats[i] lista (fila) de indice i. cv2.connectedComponentsWithStats()= la posición del bounding box(x,y), y las dimensiones (w,h) y área.
                # yo quiero de "esta (i)" fila/lista, el dato de la columna "área""
                        # cv2.CC_STAT_AREA --> índice en la matriz stats que indica el área Ej: stats[2, cv2.CC_STAT_AREA]  = 3 área
    if (area > min_area) & (area < max_area):  # filtrando los componentes según su área.
        h, w = stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_WIDTH] # cv2.CC_STAT_HEIGHT y cv2.CC_STAT_WIDTH son los índices en la matriz stats que contienen la altura y ancho del bounding box del componente i, dimensiones del rectángulo que envuelve al componente
        h_w_ratio = h / w       # relación de aspecto del componente, que es la proporción entre la altura (h) y el ancho (w).
        if 1.5 <= h_w_ratio <= 3: # filtro por relación entre la altura y el ancho está en el rango [1.5, 3],
            output_img[labels == i] = 255
plt.imshow(output_img)
plt.show(block = False)

## pruebas para relacion de aspecto -----------------------------------------------------------------------
"""
# Imprimir información sobre cada componente
print("Información sobre cada componente:")
print(f"Total de componentes: {num_labels - 1}")  # Ignorar el componente 0 (fondo)
for i in range(1, 10):  # solo muestro de la fila de indice 1 a indice 10
   x, y, w, h, area = stats[i]
   print(f"Componente {i}:")
   print(f"  - Coordenada superior izquierda: ({x}, {y})")
   print(f"  - Ancho: {w} píxeles")
   print(f"  - Altura: {h} píxeles")
   print(f"  - Área: {area} píxeles")

# Crear una lista de componentes con su índice y área
components = []
for i in range(1, num_labels):  # Comenzamos desde 1 para ignorar el fondo (componente 0)
    x, y, w, h, area = stats[i]
    components.append((i, area, x, y, w, h))  # Guardar información de cada componente

# Ordenar los componentes por área de mayor a menor
components.sort(key=lambda x: x[1], reverse=True)

# Mostrar los componentes ordenados
for component in components[:20]: # Mostrar solo los primeros 20 componentes (sin contemplar el fondo, que ya lo filtre y no lo sume a la lista componentes)
    i, area, x, y, w, h = component
    print(f"Componente {i}:")
    print(f"  - Área: {area} píxeles")
    print(f"  - Coordenada superior izquierda: ({x}, {y})")
    print(f"  - Ancho: {w} píxeles")
    print(f"  - Altura: {h} píxeles")

#Código para filtrar componentes por coordenadas x y y:
# Rango de coordenadas que deseas filtrar
x_min, x_max = 325, 420
y_min, y_max = 130, 190  # Corregido el rango de y (debe ser y_min < y_max)

# Iterar sobre los componentes y filtrar por las coordenadas
for i in range(1, num_labels):  # Ignoramos el componente 0 (fondo)
    x, y, w, h, area = stats[i]
    
    # Verificar si las coordenadas del componente están dentro del rango
    if x_min <= x <= x_max and y_min <= y <= y_max:
        print(f"Componente {i}:")
        print(f"  - Coordenada superior izquierda: ({x}, {y})")
        print(f"  - Ancho: {w} píxeles")
        print(f"  - Altura: {h} píxeles")
        print(f"  - Área: {area} píxeles")
"""   
#--------------------------------------------------------------------------------------------------------------------

'''
Busca dentro de la imagen una secuencia de un grupo de 6 caracteres,
separados en dos grupos de 3 caracteres cada uno.
Patrón buscado XXX XXX
Previamente filtra los componentes cuya coordenada y está alejada de los del resto del grupo.
'''
c_min = 5   # mínima distancia entre caracteres de un grupo
c_max = 16  # máxima distancia entre caracteres de un grupo
e_min = 10  # mínima distancia entre los dos grupos
e_max = 30  # máxima distancia entre los dos grupos
max_distance_y = 15  # Diferencia máxima en pixeles permitida en "y" para formar un grupo
# Aplicar Connected Components para encontrar posibles caracteres
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(intersection, connectivity=8, ltype=cv2.CV_32S)
# Filtrar componentes que parezcan caracteres según su relación de aspecto
possible_characters = []
for i in range(1, num_labels):  # Empezamos en 1 para ignorar el fondo  
    h, w = stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_WIDTH]
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    possible_characters.append((x, y, w, h, centroids[i]))
# Ordenar los posibles caracteres de izquierda a derecha
possible_characters.sort()
# Crear grupos iniciales por proximidad en coordenada "y" (agrupa los caracteres que están cerca en el eje vertical, y cada grupo puede contener varios 
# caracteres si están dentro de la distancia máxima max_distance_y.
groups = []   # lista vacía que almacenará sublistas de componentes 
for character in possible_characters:
    appended = False
    for group in groups:
        # Antes de agregar un caracter al grupo verifica que su coordenada "y" no esté
        # alejada de los miembros del grupo una distancia mayor a max_distance_y (15 pixeles)
        if all(abs(character[1] - member[1]) <= max_distance_y for member in group): 
            # Proximidad en y: Dentro de cada grupo, se comprueba si la diferencia entre la coordenada "y "del carácter actual (character[1]) y 
            # las coordenadas "y" de todos los miembros del grupo es menor o igual a un valor máximo de distancia (max_distance_y, que es un valor de umbral en píxeles, como 15).
            # all() evalúa si la diferencia en "y" entre el character y todos los miembros del grupo actual está dentro de la distancia máxima max_distance_y. 
            group.append(character) # Si es cierto, el carácter se agrega a "group" y la variable appended se establece como True.
            appended = True
            break
    if not appended:
        groups.append([character])
# Buscar el patrón `XXX XXX` en cada grupo formado
# Este código está diseñado para identificar patrones específicos dentro de un conjunto de componentes (probablemente caracteres o letras) agrupados en "groups".
# El patrón que busca es un grupo de 6 caracteres consecutivos con ciertas distancias específicas entre ellos, 
# para lo que usa una ventana deslizante de 6 elementos.
"""
Este código está buscando un patrón de 6 caracteres consecutivos dentro de los grupos formados por componentes conectados. El patrón debe cumplir con:
    *Distancias específicas entre los caracteres en el eje x.
    *Una distancia mínima y máxima entre los dos grupos de caracteres.
    *Los caracteres deben estar alineados verticalmente dentro de un rango tolerable.
Si encuentra un patrón que cumple con estas condiciones, calcula las coordenadas que delimitan el área que cubre ese patrón de caracteres.
"""
for group in groups:
    # Asegurarse de que el grupo tenga "al menos 6" caracteres para buscar el patrón
    if len(group) < 6:
        continue  # continue: Si el grupo tiene menos de 6 caracteres, pasa al siguiente grupo y no hace nada más en este ciclo.
    # Intentar encontrar el patrón dentro del grupo
    # si por ejemplo el grupo tiene 10 componentes, armará los siguientes subgrupos en las 10-5 iteraciones
    # iteracion 0: 012345
    # iteracion 1: 123456
    # iteracion 2: 234567
    # iteracion 3: 345678
    # iteracion 4: 456789
    # en la iteración que encuentra el patrón retorna (x_start, x_end, y_start, y_end, subgroup)
    for i in range(len(group) - 5): # Ventana deslizante sobre el grupo
        # El índice i recorre los caracteres del grupo, pero solo hasta el punto donde aún puede tomar un subgrupo de 6 caracteres consecutivos. 
        # Por ejemplo, si tienes un grupo de 10 caracteres, las iteraciones de i para que puedas tomar los subgrupos de 6 elementos (subgrupo de caracteres [0:6], [1:7], ..., [4:10]).
        subgroup = group[i:i + 6] # Crea un subgrupo de 6 caracteres consecutivos en la lista group.
        distances = [subgroup[j + 1][0] - subgroup[j][0] for j in range(5)]  # Distances entre centros de caracteres
        # Aquí se calcula la distancia entre los caracteres consecutivos en el subgrupo, pero solo entre los primeros 5 pares de caracteres consecutivos 
        # (porque estamos considerando 6 caracteres, entonces hay 5 distancias entre ellos).
        # subgroup[j + 1][0] - subgroup[j][0]: Para cada par de caracteres consecutivos (de j a j+1), se calcula la distancia en el eje x (horizontal). 
        # subgroup[j][0] es la coordenada x del carácter j y subgroup[j + 1][0] es la coordenada x del carácter j+1. 
        # La resta da la distancia horizontal entre ellos.
        alturas = [character[1] for character in subgroup] # LISTA útil para calcular si los caracteres están alineados verticalmente dentro de un rango de tolerancia.
        # Verificar si cumplen con el patrón de la patente
        if all(c_min <= dist <= c_max for dist in distances[:2] + distances[3:]) and \
        e_min <= distances[2] <= e_max and (max(alturas) - min(alturas)) < max_distance_y:
            # Obtener los límites de la patente
            x_start = subgroup[0][0]
            x_end = subgroup[-1][0] + subgroup[-1][2]  # x + ancho del último componente
            # Calcular el punto más alto y más bajo en el conjunto de caracteres
            y_start = min([character[1] for character in subgroup])  # y mínimo
            y_end = max([character[1] + character[3] for character in subgroup])  # y + altura del componente
            #return x_start, x_end, y_start, y_end, subgroup
        # El resultado es un rectángulo que cubre la región de los caracteres que forman el patrón XXX XXX, con las coordenadas x_start, x_end, y_start y y_end que delimitan este patrón.


'''
Muestra la imagen original con la patente recuadrada y
un crop de la zona de la patente con los carateres recuadrados.
'''
# def show_results(img: np.ndarray, patente: dict, margin: int)-> None:
img_with_patente = img.copy()
# Definir las coordenadas de la patente con margen
detected_patente= identify_patente(img_gray) # uso la funcion 
x_start, y_start, x_end, y_end = detected_patente["x_start"], detected_patente["y_start"], detected_patente["x_end"], detected_patente["y_end"]
x_start_margin = max(0, x_start - 2)   # defino manualmente margen de 2 pixels
y_start_margin = max(0, y_start - 2)
x_end_margin = x_end + 2  # margin
y_end_margin = y_end + 2  # margin
# Dibujar el rectángulo alrededor de la patente con el margen
cv2.rectangle(img_with_patente, (x_start_margin, y_start_margin), (x_end_margin, y_end_margin), (0, 0, 255), 2)
# Recortar la zona de la patente con margen
img_cropped = img[y_start_margin:y_end_margin, x_start_margin:x_end_margin].copy()
# Dibujar rectángulos alrededor de los caracteres en la zona recortada
for char in detected_patente["characters"]:
    char_x, char_y, char_w, char_h = char[0] - x_start_margin, char[1] - y_start_margin, char[2], char[3]
    cv2.rectangle(img_cropped, (char_x, char_y), (char_x + char_w, char_y + char_h), (0, 0, 255), 1)
# Crear subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img_with_patente)
axes[0].set_title("Area de la Patente en la Imagen")
axes[0].axis("off")
axes[1].imshow(img_cropped)
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


# Armo grupos de umbral, min max
possible_characters = [(10, 100), (20, 105), (15, 250), (30, 260), (12, 108)]
max_distance_y = 15
groups = []
for character in possible_characters:
   appended = False
   for group in groups:
        # Antes de agregar un caracter al grupo verifica que su coordenada "y" no esté
         # alejada de los miembros del grupo una distancia mayor a max_distance_y (15 pixeles)
        if all(abs(character[1] - member[1]) <= max_distance_y for member in group):
             group.append(character)
             appended = True
             print("2")
             break
     # agrega el primer grupo
        if not appended:
            groups.append([character])
            print("1")
print(groups)

'''
Realiza un proceso iterativo con distintos umbrales de binarización
hasta obtener una imagen donde puede reconocerse el patrón XXX XXX de una patente
'''

#def identify_patente(img):
      
# funcion con Otsu NO FUNCIONA----------------------------------------------------
""" 
POsible causa:
En el caso de usar Otsu's thresholding, es probable que el umbral calculado no sea adecuado para la imagen, 
lo que puede causar que la detección no funcione correctamente y retorne None. Esto es algo normal si el umbral 
seleccionado por Otsu no es lo suficientemente bueno para segmentar la imagen correctamente.
"""
"""
def identify_patente(img_gray):
    # Convertir la imagen a escala de grises
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar el umbral de Otsu automáticamente
    _, img_cleaned = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Continuar con el resto del proceso
    img_filtered = filter_area_aspect(img_cleaned)
    result = detect_patente(img_filtered)

    if result is not None:
        x_start, x_end, y_start, y_end, characters = result
        detected_patente = {
            "x_start": x_start,
            "x_end": x_end,
            "y_start": y_start,
            "y_end": y_end,
            "characters": characters
        }
        return detected_patente
"""

#---------------------------------------------------------------------------------------------
# Programa Principal
#---------------------------------------------------------------------------------------------
'''
Identifica una patente
Lee un archivos .png de la carpeta patentes.
Muestra los resultados.
'''
img = cv2.imread(f"Ejercicio2\\Patentes\\img01.png")
#img = cv2.imread("patentes\img01.png")
img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detected_patente = identify_patente(img_gray)
show_results(img_gray, detected_patente, 5)


'''
Identifica todas las patentes en la carpeta "patentes"
Lee los archivos .png de la carpeta patentes.
Muestra los resultados.
'''
path_patentes = 'Ejercicio2\\Patentes'
files = [os.path.join(path_patentes, f) for f in os.listdir(path_patentes) if f.endswith('.png')]
# Diccionario para almacenar la información de las patentes detectadas
detected_patentes = []
for file in files:
    img = cv2.imread(file)
    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_patente = identify_patente(img_gray)
    show_results(img_color, detected_patente, 5)

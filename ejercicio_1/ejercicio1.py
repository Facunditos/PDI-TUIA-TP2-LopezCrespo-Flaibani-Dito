import cv2
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle 
import warnings
warnings.filterwarnings("ignore")




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

      

# ----------------------- Punto A -----------------------------------------------

img_BGR = cv2.imread('monedas.jpg')
img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
imshow(img_RGB, title="Imagen a color")  

img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)   # Leemos imagen
imshow(img,title='Imagen en escala de grises')

# ----- Suavizado -------

img_blur = cv2.medianBlur(img, 5)

# ----- Canny ------------

parejas_umbrales = [(0.1*255,0.25*255),(0.25*255,0.6*255),(0.6*255,0.9*255)]
plt.figure()
ax = plt.subplot(221)
imshow(img_blur, new_fig=False, title="Imagen suavizada")
for i,pareja_umbrales in enumerate(parejas_umbrales):
    th_1,th_2 = pareja_umbrales
    img_canny = cv2.Canny(img_blur, threshold1=th_1, threshold2=th_2)
    pos_sup = f'22{i+2}'
    plt.subplot(int(pos_sup), sharex=ax, sharey=ax), imshow(img_canny, new_fig=False, title=f"Canny\n(u_inf:{th_1}, u_sup:{th_2}")

plt.show(block=False)

# Se elige opción intermedia
th_1 = 0.25*255
th_2 = 0.60*255
img_canny = cv2.Canny(img_blur, threshold1=th_1, threshold2=th_2)

# ----- Dilatación ---------

tamaño_kernels = [9,29,49]
plt.figure()
ax = plt.subplot(221)
imshow(img_canny, new_fig=False, title=f"Canny\n(u_inf:{th_1}, u_sup:{th_2})")
for i,size in enumerate(tamaño_kernels):
    kernel = np.ones((size,size),dtype='uint8')
    img_dilatada = cv2.dilate(img_canny, kernel, iterations=1)
    pos_sup = f'22{i+2}'
    plt.subplot(int(pos_sup), sharex=ax, sharey=ax), imshow(img_dilatada, new_fig=False, title=f"Dilatación\n(kernel:{size} X {size})")

plt.show(block=False)

# Se elige opción intermedia
size = 29
kernel = np.ones((size,size),dtype='uint8')
img_dilatada = cv2.dilate(img_canny, kernel, iterations=1)

# ----- Relleno de huecos ---------

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


plt.figure()
ax = plt.subplot(121)
imshow(img_dilatada, new_fig=False, title=f"Dilatación\n(kernel:{size} X {size})")
img_fh = imfillhole(img_dilatada)
plt.subplot(122, sharex=ax, sharey=ax), imshow(img_fh, new_fig=False, title=f"Relleno de huecos")

plt.show(block=False)

# ----- Erosión ---------

tamaño_kernels = [3,39,89]
plt.figure()
ax = plt.subplot(221)
imshow(img_fh, new_fig=False, title=f"Relleno de huecos")
for i,size in enumerate(tamaño_kernels):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size) )
    img_erosion = cv2.erode(img_fh, kernel, iterations=1)
    pos_sup = f'22{i+2}'
    plt.subplot(int(pos_sup), sharex=ax, sharey=ax), imshow(img_erosion, new_fig=False, title=f"Erosión\n(kernel:{size} X {size})")

plt.show(block=False)

# Se elige opción intermedia
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (39, 39) )
img_erosion = cv2.erode(img_fh, kernel, iterations=1)

# ----- Componentes 8 conectadas ---------
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_erosion, connectivity, cv2.CV_32S)
monedas = []
dados = []
for i in range(1,num_labels):
    estadistica = stats[i,:]
    x = estadistica[0]
    y = estadistica[1]
    ancho = estadistica[2]
    alto = estadistica[3]
    area_caja = estadistica[4]
    img_obj = img_erosion[y:y+alto,x:x+ancho]
    contours, hierarchy = cv2.findContours(img_obj, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    f_p = round(area / (perimeter**2),4)
    info_obj = {
        'coor': (x,y,ancho,alto) ,
        'area_obj': area,
        'area_caja': area_caja,
        'f_p': f_p,
    }
    # Asumimos que si factor de forma del objeto es menor a 0.657 dista de ser un círculo, por ende, es un dado
    if (f_p<0.0657):
        dados.append(info_obj)       
    else:
        monedas.append(info_obj)  
 

imshow(img_RGB,title='Segmentación de dados y monedas')

for il, dado in enumerate(dados):
    x,y,ancho,alto = dado['coor']
    rect = Rectangle((x,y), ancho, alto, linewidth=1, edgecolor='r', facecolor='none')   
    ax = plt.gca()          
    ax.add_patch(rect)     

for il, moneda in enumerate(monedas):
    x,y,ancho,alto = moneda['coor']
    rect = Rectangle((x,y), ancho, alto, linewidth=1, edgecolor='b', facecolor='none')   
    ax = plt.gca()          
    ax.add_patch(rect)  

plt.show(block=False)

# ----------------------- Punto B -----------------------------------------------

# Construimos una lista de listas. Cada lista reportará el área de la moneda y su índice. Luego se agregará el número de cluster
info_monedas = [[moneda['area_obj'],idx_moneda] for idx_moneda,moneda in enumerate(monedas)]
# Ordenamos las monedas de menor a mayor según su área
info_monedas.sort()
areas_monedas_ord_asc = [moneda[0] for moneda in info_monedas]

# Calculamos el salto de área que hay al pasar de una moneda a la otra
deltas = []
for i,area_moneda in enumerate(areas_monedas_ord_asc):
    if (i==0):
        continue
    area_moneda_anterior = areas_monedas_ord_asc[i-1]
    delta =  area_moneda - area_moneda_anterior
    deltas.append(delta)

# Calculamos un delta atípico, el cual asumimos que será útil para seprar clusters

q1 = np.percentile(deltas, 25)
q3 = np.percentile(deltas, 75)
iqr = q3 - q1
delta_sep_clusters = q3 + iqr *1.5 


plt.figure()
ax = plt.subplot(121)
plt.bar(x=range(1,len(monedas)+1),height=areas_monedas_ord_asc)
plt.title('Area de las monedas ordenadas ascendentemente')
plt.xlabel('número de moneda')
plt.ylabel('área')
etiquetas_y = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:,.0f}'.format(etiqueta_y) for etiqueta_y in etiquetas_y])
plt.xticks(range(1,len(monedas)+1))

plt.subplot(122)
plt.bar(x=range(2,len(deltas)+2),height=deltas)
plt.axhline(delta_sep_clusters,color='blue',label=f'delta atípico',ls='--')
plt.legend()
plt.title('Aumento del área entre monedas')
plt.xlabel('número de moneda')
plt.ylabel('delta')
etiquetas_y = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:,.0f}'.format(etiqueta_y) for etiqueta_y in etiquetas_y])
plt.xticks(range(2,len(deltas)+2))

plt.show(block=False)

# Sabemos que la moneda 1 inexorablemente pertecene al primer cluster
numero_cluster = 1
info_monedas[0].append(numero_cluster)
for i,delta_obs in enumerate(deltas):
    # Si el delta observado no supera el delta diferenciador de cluster, 
    # se asigna a la moneda bajo análisis el cluster contrario. En caso de 
    # estar en presencia de un delta significativo, se asume que corresponde crear un nuevo cluster
    if (delta_obs<delta_sep_clusters):
        info_monedas[i+1].append(numero_cluster)
    else:
        numero_cluster +=1
        info_monedas[i+1].append(numero_cluster)

# Agregamos para cada moneda info sobre el cluster al cual pertence
for info_moneda in info_monedas:
    idx_moneda = info_moneda[1]
    numero_cluster = info_moneda[2]
    monedas[idx_moneda]['numero_cluster'] = numero_cluster

print("-----------------------------")
print("----Conteo de monedas----")
print(f'En función al área de las monedas se pueden distinguir {numero_cluster} clusters o tipos de monedas diferenes')

print("Asociamos el cluster 1 con las monedas de 10 centavos, cluster 2 con las de 1 peso y cluster 3 con las de 50 centavos")

q_moneda_10 = 0
q_moneda_50 = 0
q_moneda_100 = 0

imshow(img,title='Separación de monedas en clusters')

for il, moneda in enumerate(monedas):
    x,y,ancho,alto = moneda['coor']
    num_cluster = moneda['numero_cluster']
    if num_cluster == 1:
        rect = Rectangle((x,y), ancho, alto, linewidth=1, edgecolor='r', facecolor='none')   
        q_moneda_10 +=1
    elif num_cluster == 2:
        rect = Rectangle((x,y), ancho, alto, linewidth=1, edgecolor='b', facecolor='none')   
        q_moneda_100 +=1
    else:
        rect = Rectangle((x,y), ancho, alto, linewidth=1, edgecolor='g', facecolor='none')   
        q_moneda_50 +=1
    ax = plt.gca()          
    ax.add_patch(rect)   

plt.show(block=False)

dinero = q_moneda_10 * 10 + q_moneda_50 * 50 + q_moneda_100 * 100
pesos = dinero // 100
centavos = dinero % 100
print(f"El conteo de monedas arroja que sobre la mesa hay {pesos} pesos con {centavos} centavos producto de identificarse:\n\t-{q_moneda_100} monedas de 1 peso\n\t-{q_moneda_50} monedas de 50 centavos\n\t-{q_moneda_10} monedas de 10 centavos")


# ----------------------- Punto C -----------------------------------------------

# Se crea una lista de seis elementos porque hay seis valores de dados distintos
# el primer elemento informa la cantidad de dados de valor 1 y el úlitmo elemento 
# la cantidad de dados de valor 6
q_dados_segun_valor = [0 for _ in range(6)]

for dado in dados:
    x,y,ancho,alto = dado['coor']
    img_dado = img[y:y+alto,x:x+ancho]
    img_dado_suavizada = cv2.medianBlur(img_dado,3)
    circles = cv2.HoughCircles(img_dado_suavizada,cv2.HOUGH_GRADIENT, 1, 20, param1=65, param2=40, minRadius=0, maxRadius=50)  # Circulos chicos
    # Se asume que la cantaidad de círculos encontrados se corresponde con la cara superior del dado
    valor_dado = len(circles[0])
    q_dados_segun_valor[valor_dado-1] += 1
    circles = np.uint16(np.around(circles))
    cimg = cv2.cvtColor(img_dado, cv2.COLOR_GRAY2BGR)
    for i in circles[0,:]:
        cv2.circle(cimg, (i[0],i[1]), i[2], (0,255,0), 2)   # draw the outer circle
        cv2.circle(cimg, (i[0],i[1]), 2, (0,0,255), 2)      # draw the center of the circle
    imshow(cimg,title=f'Dado cuya cara superior tiene un valor de {valor_dado}')

print("\n----Conteo de dados----")
print(f"El conteo de dados arroja que sobre la mesa {len(dados)} dados de las siguientes denominaciones:")
[print(f'\t-{q_dado} dado de valor {i+1}')for i,q_dado in enumerate(q_dados_segun_valor) if q_dado>0]
print("-----------------------------")

input('\n\nquiere cerrar el programa:\t')


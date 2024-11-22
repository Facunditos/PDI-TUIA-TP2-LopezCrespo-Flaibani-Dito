import cv2
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle 



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

# ----------------------- Ejercicio A -----------------------------------------------

img = cv2.imread('monedas.jpg', cv2.IMREAD_GRAYSCALE)   # Leemos imagen
imshow(img,title='gris')

img_blur = cv2.medianBlur(img, 5)
imshow(img_blur,title='suavizada')

th_1 = 0.25*255
th_2 = 0.60*255
img_canny = cv2.Canny(img_blur, threshold1=th_1, threshold2=th_2)
imshow(img_canny,title='canny')

kernel = np.ones((29,29),dtype='uint8')
img_dilatada = cv2.dilate(img_canny, kernel, iterations=1)
imshow(img_dilatada,title='dilatada')

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

# --- Rellenado de huecos -----------------------------------------------------
def imfillhole(img):
    # img: Imagen binaria de entrada. Valores permitidos: 0 (False), 255 (True).
    mask = np.zeros_like(img)                                                   # Genero mascara para...
    mask = cv2.copyMakeBorder(mask[1:-1,1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255)) # ... seleccionar los bordes.
    marker = cv2.bitwise_not(img, mask=mask)                # El marcador lo defino como el complemento de los bordes.
    img_c = cv2.bitwise_not(img)                            # La mascara la defino como el complemento de la imagen.
    img_r = imreconstruct(marker=marker, mask=img_c)        # Calculo la reconstrucción R_{f^c}(f_m)
    img_fh = cv2.bitwise_not(img_r)                         # La imagen con sus huecos rellenos es igual al complemento de la reconstruccion.
    return img_fh

img_fh = imfillhole(img_dilatada)
imshow(img_fh,title='relleno de huecos')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (39, 39) )
img_erosion = cv2.erode(img_fh, kernel, iterations=1)
imshow(img_erosion,title='erosion')

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
        'img': img_obj,
        'area_obj': area,
        'area_caja': area_caja,
        'f_p': f_p,
    }
    # Asumimos que si factor de forma del objeto es menor a 0.657 dista de ser un círculo, por ende, es un dado
    if (f_p<0.0657):
        dados.append(info_obj)       
    else:
        monedas.append(info_obj)  

for moneda in monedas:
    x,y,ancho,alto = moneda['coor']
    imshow(img[y:y+alto,x:x+ancho])


for dado in dados:
    x,y,ancho,alto = dado['coor']
    imshow(img[y:y+alto,x:x+ancho])    

plt.figure(), plt.imshow(img, cmap='gray')

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

# ----------------------- Ejercicio B -----------------------------------------------

# Construimos una lista de listas. Cada lista reportará el área de la moneda, su índice y su número de cluster
info_monedas = [[moneda['area_obj'],idx_moneda] for idx_moneda,moneda in enumerate(monedas)]
info_monedas.sort()
areas_monedas_ord_asc = [moneda[0] for moneda in info_monedas]


deltas = []
for i,area_moneda in enumerate(areas_monedas_ord_asc):
    if (i==0):
        continue
    area_moneda_anterior = areas_monedas_ord_asc[i-1]
    delta =  area_moneda - area_moneda_anterior
    deltas.append(delta)

q1 = np.percentile(deltas, 25)
q3 = np.percentile(deltas, 75)
iqr = q3 - q1
delta_sep_clusters = q3 + iqr *1.5 


plt.figure()
ax = plt.subplot(221)
plt.bar(x=range(1,len(monedas)+1),height=areas_monedas_ord_asc)
plt.title('area de las monedas ordenadas ascendentemente')
plt.xlabel('número de moneda')
plt.ylabel('área')
plt.xticks(range(1,len(monedas)+1))

plt.subplot(222)
plt.bar(x=range(2,len(deltas)+2),height=deltas)
plt.axhline(delta_sep_clusters,color='blue',label=f'delta atípico',ls='--')
plt.legend()
plt.title('Aumento del área entre monedas')
plt.xlabel('número de moneda')
plt.ylabel('delta')
plt.xticks(range(2,len(deltas)+2))

plt.show(block=False)

# Sabemos que la moneda 1 inexorablemente pertecene al primer cluster
numero_cluster = 1
info_monedas[0].append(numero_cluster)
for i,delta_obs in enumerate(deltas):
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

print(f'En función al área de las monedas se pueden distinguir {numero_cluster} clusters o tipos de monedas diferenes')

# Asociamos el cluster 1 con las monedas de 10 centavos, cluster 2 con las de 1 peso y cluster 3 con las de 50 centavos

plt.figure(), plt.imshow(img, cmap='gray')

for il, moneda in enumerate(monedas):
    x,y,ancho,alto = moneda['coor']
    num_cluster = moneda['numero_cluster']
    if num_cluster == 1:
        rect = Rectangle((x,y), ancho, alto, linewidth=1, edgecolor='r', facecolor='none')   
        conteo +=10
    elif num_cluster == 2:
        conteo +=100
    else:
        conteo +=50
    ax = plt.gca()          
    ax.add_patch(rect)   

plt.show(block=False)

           
conteo = 0
for moneda in monedas:
    num_cluster = moneda['numero_cluster']
    if num_cluster == 1:
        conteo +=10
    elif num_cluster == 2:
        conteo +=100
    else:
        conteo +=50
    print('num_cluster',num_cluster)    
    print('conteo',conteo)    

pesos = conteo // 100
centavos = conteo % 100
print(f'El conteno de monedas arroja que sobre la mesa hay {pesos} pesos con {centavos} centavos')        




monedas_10_c = []
monedas_50_c = []
monedas_100_c = []

i = 0
while i<len():
    cluster = []


persona = ('facundo',35)    

nombre,edad = persona



[print(int(area)) for area in areas_monedas]

for moneda in monedas:
    f_p = moneda['f_p']
    imshow(moneda['img'],title=f'factor de forma: {f_p}')

imshow(dados[0]['img'])


## Suavizamos la imagen utlizando como filtro paso bajo: GaussianBlur
img_blur = cv2.medianBlur(img, 5)

plt.figure()
ax = plt.subplot(221)
imshow(img, new_fig=False, title="Imagen Original")
plt.subplot(222, sharex=ax, sharey=ax), imshow(img_blur, new_fig=False, title="Imagen suavizada")
plt.show(block=False)


## Aplicamos Canny para detectar bordes
## CONSULTA: ¿Canny aplica internamente un filtro Gaussiano?

umbrales = [(0.01,0.05),(0.10,0.20),(0.20,0.35)]
plt.figure()
ax = plt.subplot(221)
imshow(img, new_fig=False, title="Imagen Original")
for i,tup_th in enumerate(umbrales):
    th_1 = tup_th[0] *255
    th_2 = tup_th[1] * 255
    img_canny = cv2.Canny(img_blur, threshold1=th_1, threshold2=th_2)
    pos_sup = f'22{i+2}'
    plt.subplot(int(pos_sup), sharex=ax, sharey=ax), imshow(img_canny, new_fig=False, title=f"Canny - U1={th_1/255*100}% | U2={th_2/255*100}%")

plt.show(block=False)


th_1 = 0.25*255
th_2 = 0.60*255
img_canny = cv2.Canny(img_blur, threshold1=th_1, threshold2=th_2)


imshow(img_canny)



# ---- Dilatación  (Closing) -----------------------
tamaño_kernel = [3,10,19]
plt.figure()
ax = plt.subplot(221)
imshow(img_canny, new_fig=False, title="Imagen Original")
for i,size in enumerate(tamaño_kernel):
    kernel = np.ones((size,size),dtype='uint8')
    size
    img_dilatada = cv2.dilate(img_canny, kernel, iterations=1)
    pos_sup = f'22{i+2}'
    plt.subplot(int(pos_sup), sharex=ax, sharey=ax), imshow(img_dilatada, new_fig=False, title=f"Dilatación: forma{kernel.shape}")

plt.show(block=False)

kernel = np.ones((25,25),dtype='uint8')
img_dilatada = cv2.dilate(img_canny, kernel, iterations=1)

B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
img_dilatada = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, B)
    
imshow(img_dilatada)
B = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
img_apertura = cv2.morphologyEx(img_dilatada, cv2.MORPH_OPEN, B) 

imshow(img_apertura,title='apertura')
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

# --- Rellenado de huecos -----------------------------------------------------
def imfillhole(img):
    # img: Imagen binaria de entrada. Valores permitidos: 0 (False), 255 (True).
    mask = np.zeros_like(img)                                                   # Genero mascara para...
    mask = cv2.copyMakeBorder(mask[1:-1,1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255)) # ... seleccionar los bordes.
    marker = cv2.bitwise_not(img, mask=mask)                # El marcador lo defino como el complemento de los bordes.
    img_c = cv2.bitwise_not(img)                            # La mascara la defino como el complemento de la imagen.
    img_r = imreconstruct(marker=marker, mask=img_c)        # Calculo la reconstrucción R_{f^c}(f_m)
    img_fh = cv2.bitwise_not(img_r)                         # La imagen con sus huecos rellenos es igual al complemento de la reconstruccion.
    return img_fh

img_fh = imfillhole(img_dilatada)


imshow(img_fh,title='relleno')

# --- Apertura (Opening) ------------------------

B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (53, 53))
img_apertura = cv2.morphologyEx(img_fh, cv2.MORPH_OPEN, B) 

imshow(img_apertura,title='apertura')


connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_apertura, connectivity, cv2.CV_32S)

info_objetos = []
for i in range(1,num_labels):
    estadistica = stats[i,:]
    x = estadistica[0]
    y = estadistica[1]
    ancho = estadistica[2]
    alto = estadistica[3]
    img_obj = img_apertura[y:y+alto,x:x+ancho]
    contours, hierarchy = cv2.findContours(img_obj, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    f_p = round(area / (perimeter**2),4)
    info_obj = (img_obj,f_p)
    info_objetos.append(info_obj)
    
dados = []
monedas = []
for img,f_p in info_objetos:
    #imshow(img,title=f'{f_p}')    
    if ((f_p>=0.00565) and f_p<=0.0647):
        dados.append((img,f_p))       
    else:
        monedas.append((img,f_p))  








stats[1,:][0]
# ---- Clausura (Closing) -----------------------
tamaño_elemento_estructural = [1,3,9]
plt.figure()
ax = plt.subplot(221)
imshow(img_canny, new_fig=False, title="Imagen Original")
for i,size in enumerate(tamaño_elemento_estructural):
    B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    img_clausura = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, B)
    pos_sup = f'22{i+2}'
    plt.subplot(int(pos_sup), sharex=ax, sharey=ax), imshow(img_clausura, new_fig=False, title=f"Clausura: {size}")

plt.show(block=False)

k1 = 17#20
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k1, k1))
dilated = cv2.dilate(img_canny, kernel1)
img_dilatada = cv2.dilate(img_canny, kernel, iterations=1)


imshow(img_dilatada)

# --- Erosion (Erode) ---------------------------

tamaño_elipse = [1,3,9] # Probar con 10 - 30 - 70

plt.figure()
ax = plt.subplot(221)
imshow(img_canny, new_fig=False, title="Imagen Original")
for i,size in enumerate(tamaño_elipse):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size) )
    img_erosion = cv2.erode(img_canny, kernel, iterations=1)
    pos_sup = f'22{i+2}'
    plt.subplot(int(pos_sup), sharex=ax, sharey=ax), imshow(img_erosion, new_fig=False, title=f"Erosión: forma{size}")

plt.show(block=False)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3) )
img_erosion = cv2.erode(img_canny, kernel, iterations=1)

imshow(img_erosion,title='erosion')

# Factor de forma
moneda_1 =img_fh[1347:1800,2215:2624].copy()
imshow(moneda_1,title='moneda 1')

contours, hierarchy = cv2.findContours(moneda_1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

cnt = contours[0]
cnt
area = cv2.contourArea(cnt)
perimeter = cv2.arcLength(cnt,True)
f_p = area / (perimeter**2)
f_p


moneda_1 =img[1160:1500,430:810].copy()
imshow(moneda_1)
moneda_1_blur = cv2.medianBlur(moneda_1,3)
imshow(moneda_1_blur)
moneda_1_canny = cv2.Canny(moneda_1_blur,70,90)
imshow(moneda_1_canny)
moneda_1_canny = cv2.copyMakeBorder(moneda_1_canny, 50, 50, 50, 50, cv2.BORDER_CONSTANT, None, 0)
imshow(moneda_1_canny)

kernel = np.ones((3,3),dtype='uint8')
moneda_1_dilatada = cv2.dilate(moneda_1_canny, kernel, iterations=1)
imshow(moneda_1_dilatada)
moneda_1_fh = imfillhole(moneda_1_dilatada)
imshow(moneda_1_fh,title='moneda 1')
B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
moneda_1_clausura = cv2.morphologyEx(moneda_1_fh, cv2.MORPH_CLOSE, B)
imshow(moneda_1_clausura)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9) )
moneda_1_erosion = cv2.erode(moneda_1_fh, kernel, iterations=1)
imshow(moneda_1_erosion)
B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (75,75))
moneda_1_apertura = cv2.morphologyEx(moneda_1_erosion, cv2.MORPH_OPEN, B) 
imshow(moneda_1_apertura) 


contours, hierarchy = cv2.findContours(moneda_1_erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

cnt = contours[0]
cnt
area = cv2.contourArea(cnt)
perimeter = cv2.arcLength(cnt,True)
f_p = area / (perimeter**2)
f_p
import cv2
import numpy as np
import matplotlib.pyplot as plt 


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


img = cv2.imread('monedas.jpg', cv2.IMREAD_GRAYSCALE)   # Leemos imagen
imshow(img,title='gris')

img_blur = cv2.medianBlur(img, 5)
imshow(img_blur,title='suavizada')

th_1 = 0.25*255
th_2 = 0.60*255
img_canny = cv2.Canny(img_blur, threshold1=th_1, threshold2=th_2)
imshow(img_canny,title='canny')

kernel = np.ones((23,23),dtype='uint8')
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

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33) )
img_erosion = cv2.erode(img_fh, kernel, iterations=1)
imshow(img_erosion,title='erosion')

B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (199, 199))
img_apertura = cv2.morphologyEx(img_fh, cv2.MORPH_OPEN, B) 
imshow(img_apertura,title='apertura')


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
    if ((f_p>=0.0565) and f_p<=0.0647):
        dados.append(info_obj)       
    else:
        monedas.append(info_obj)  

for moneda in dados:
    moneda['area_obj']

moneda_i_3 = monedas[3]
moneda_i_3['area_obj']
moneda_i_3['area_caja']
imshow(moneda_i_3['img'])

moneda_i_4 = monedas[4]
moneda_i_4['area_obj']
moneda_i_4['area_caja']
imshow(moneda_i_4['img'])

moneda_i_5 = monedas[5]
moneda_i_5['area_obj']
moneda_i_5['area_caja']
imshow(moneda_i_5['img'])

moneda_i_6 = monedas[6]
x,y,ancho,alto = moneda_i_6['coor']
imshow(img[y:y+alto,x:x+ancho])
moneda_i_6['area_obj']
moneda_i_6['area_caja']
imshow(moneda_i_6['img'])

moneda_i_7 = monedas[7]
moneda_i_7['area_obj']
moneda_i_7['area_caja']
imshow(moneda_i_7['img'])


moneda_i_9 = monedas[9]
x,y,ancho,alto = moneda_i_9['coor']
moneda_i_9['area_obj']
moneda_i_9['area_caja']
imshow(moneda_i_9['img'])

imshow(img[y:y+alto,x:x+ancho])

for dado in dados:
    f_p = dado['f_p']
    imshow(dado['img'],title=f'factor de forma: {f_p}')

areas_monedas = [(moneda['area_caja'],idx_moneda) for idx_moneda,moneda in enumerate(monedas)]
areas_monedas_ord_asc = sorted(areas_monedas)
deltas = []
for i,area_moneda in enumerate(areas_monedas_ord_asc):
    area,idx_moneda = area_moneda
    if (i==0):
        continue
    delta =  area - areas_monedas_ord_asc[i-1][0]
    deltas.append(delta)

[print(f"{delta:,d}") for delta in deltas]

for delta in deltas:
    print(f"{delta:,d}")

delta_sep_cluster = np.mean(deltas)

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
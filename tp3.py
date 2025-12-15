import cv2
import numpy as np
import os

# PASO 1: DETECCIÓN DE FRAMES ESTÁTICOS
def detectar_frames_estaticos(cap, ventana=4):
    """
    Detecta zonas de bajo movimiento promedio
    """
    print("PASO 1: DETECTANDO MOVIMIENTO")
    
    ret, frame_prev = cap.read()
    movimientos = []
    frames = []
    frame_count = 0
    
    # Primera pasada: registrar movimientos
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gray, gray_prev)
        movimiento = np.sum(diff)
        
        movimientos.append(movimiento)
        frames.append(frame.copy())
        frame_prev = frame.copy()
        
        # if frame_count % 10 == 0: #debug
        #     print(f"Frame {frame_count}: Movimiento = {movimiento:.0f}")
    
    # Buscar ventana con menor movimiento promedio
    mejor_inicio = 0
    menor_promedio = float('inf')
    
    for i in range(len(movimientos) - ventana):
        ventana_actual = movimientos[i:i+ventana]
        promedio = np.median(ventana_actual)
        
        if promedio < menor_promedio:
            menor_promedio = promedio
            mejor_inicio = i # + ventana // 2  # Frame central de la ventana
    
    print(f"Zona más estática: frames {mejor_inicio-ventana//2} a {mejor_inicio+ventana//2}")
    # print(f"  Movimiento promedio: {menor_promedio:.0f}")
    
    return frames[mejor_inicio], mejor_inicio + 1


# PASO 2: SEGMENTACIÓN DE DADOS (con parametros hsv de la máscara obtenida)

def segmentar_dados(frame): #, mostrar_debug=True):
    """
    Detecta y segmenta los dados rojos usando los valores optimizados.
    """

    print("PASO 2: SEGMENTACIÓN DE DADOS")
    
    # Convertir a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # VALORES OPTIMIZADOS
    # S_min = 111 -> para ignorar los puntos blancos (que tienen baja saturación)
    # V_min = 100 -> para ignorar el fondo oscuro
    
    # RANGO 1: Rojo del inicio del espectro (0-10)
    lower_red1 = np.array([0, 111, 100])
    upper_red1 = np.array([10, 255, 255])
    
    # RANGO 2: Rojo del final del espectro (170-180)
    # Usamos la misma S y V que en el rango 1 (Regla del espejo)
    lower_red2 = np.array([170, 111, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Crear máscaras
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)
    
    # if mostrar_debug:
    #     cv2.imshow('2.1 - Mascara Cruda (Con huecos)', mask_red)

    # MORFOLOGÍA AGRESIVA (Kernel 21)
    
    # Usamos un kernel de 21x21 como resultado del análisis de la máscara.
    # Para fusionar los fragmentos rojos y tapar los agujeros de los puntos blancos.
    kernel_size = 21
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # CLOSE: Rellena los huecos internos (puntos de los dados)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    
    # OPEN: Limpia el ruido externo
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    
    # if mostrar_debug:
    #     cv2.imshow(f'2.2 - Mascara Solida (Kernel {kernel_size})', mask_red)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Contornos detectados: {len(contours)}")
    
    # Filtrar contornos
    dados = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        # lógica por geometría, sería bueno implementar comparar entre los dados, si se cumple similitud, se aprueban
        # Filtros de tamaño y forma
        if 500 < area < 10000:  # como la máscara es adecuada, cualquier valor está bien como max
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            
            if 0.7 < aspect_ratio < 1.3: # aproximadamente un cuadrado
                dados.append((x, y, w, h))
                # print(f"  Dado {len(dados)}: área={area:.0f}, ratio={aspect_ratio:.2f}") #debug
    
    # if mostrar_debug:
    #     frame_contornos = frame.copy()
    #     for i, (x, y, w, h) in enumerate(dados):
    #         cv2.rectangle(frame_contornos, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #     cv2.imshow('2.3 - Deteccion Final', frame_contornos)
    
    print(f"- Total de dados válidos: {len(dados)}")
    return dados


# PASO 3: IDENTIFICACIÓN DEL VALOR
def identificar_valor(roi, dado_num=0): #, mostrar_debug=True)
    """
    Identifica el valor (1-6) contando puntos blancos dentro del ROI.
    """
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Rango para detectar BLANCO (puntos)
    # Baja saturación, Alto valor
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    
    mask_white = cv2.inRange(roi_hsv, lower_white, upper_white)
    
    # Limpieza suave para los puntos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
    
    # if mostrar_debug:
    #     cv2.imshow(f'3.{dado_num} - Puntos', mask_white)
    
    # Contar componentes
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_white, connectivity=8
    )
    
    puntos = 0
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Filtro de área para un punto individual
        if 30 < area < 400: 
            puntos += 1
            
    # Fallback: si no detecta puntos (ej. un 1 central muy grande), intentar otro método
    # o simplemente devolver 1 si el área blanca total es grande, pero por ahora
    # nos fiamos del conteo.
    if puntos == 0:
        # A veces el 1 es tan grande que se filtra
        # Asumimos 1 si hay una mancha blanca grande en el centro, pero lo dejamos simple.
        pass

    return puntos


# PASO 4: GENERACIÓN DE VIDEO CON ANOTACIONES
def generar_video_anotado(video_path, dados, valores, frame_estatico_num, output_path):
    print("PASO 4: GENERANDO VIDEO ANOTADO")
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        if frame_count >= frame_estatico_num:
            for i, ((x, y, w, h), valor) in enumerate(zip(dados, valores)):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                texto = f"Dado{i+1}: {valor}"
                # Poner texto centrado sobre el dado
                cv2.putText(frame, texto, (x - 60, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        
        out.write(frame)

    cap.release()
    out.release()
    print(f"- Video guardado: {output_path}")

# PRINCIPAL
def procesar_video_completo(video_path):
    output_dir = "resultados"
    os.makedirs(output_dir, exist_ok=True)
    nombre_base = os.path.splitext(os.path.basename(video_path))[0]
    
    print(f"\nPROCESANDO: {video_path}")
    print(f"{"="*24}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error al abrir video.")
        return

    # 1. Detectar quietos
    frame, frame_num = detectar_frames_estaticos(cap)
    if frame is None: return
    
    # 2. Segmentar con nuevos parámetros
    dados = segmentar_dados(frame)
    if not dados: return
    
    # 3. Identificar valores
    valores = []
    resultado_img = frame.copy()
    
    print("PASO 3: LEYENDO VALORES")
    for i, (x, y, w, h) in enumerate(dados):
        roi = frame[y:y+h, x:x+w]
        valor = identificar_valor(roi, dado_num=i+1)
        valores.append(valor)
        
        cv2.rectangle(resultado_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(resultado_img, str(valor), (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"  Dado {i+1}: {valor}")
        
    # cv2.imshow('Resultado Final', resultado_img) # Este podríamos dejarlo para mostrar el resultado final
    cv2.imwrite(f"{output_dir}/{nombre_base}_resultado.jpg", resultado_img)

    
    # 4. Video
    cap.release()
    output_video = f"{output_dir}/{nombre_base}_anotado.mp4"
    generar_video_anotado(video_path, dados, valores, frame_num, output_video)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    videos = ['tirada_1.mp4', 'tirada_2.mp4', 'tirada_3.mp4', 'tirada_4.mp4']
    for video in videos:
        if os.path.exists(video):
            procesar_video_completo(video)
        else:
            print(f"Video no encontrado: {video}")
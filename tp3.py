import cv2
import numpy as np
import os

def mascara_roja(frame):
    """
    Devuelve una máscara binaria con las zonas rojas del frame.
    Utiliza Blur y morfologías de cierre y apertura para mejorar la máscara.
    """
    frame = cv2.GaussianBlur(frame, (7, 7), 0)

    lower_red1 = np.array([0, 120, 100])
    upper_red1 = np.array([5, 255, 255])
    
    lower_red2 = np.array([175, 120, 100])
    upper_red2 = np.array([180, 255, 255])
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2) # Union de ambas máscaras (rojo bajo y alto)

    # Cierre grande para unir zonas rojas cercanas y desaparecer puntitos de los dados
    kernel_close = np.ones((11, 11), np.uint8)
    mask_solida = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel_close)

    # Pequeña apertura para borrar ruido en la mesa
    kernel_open = np.ones((5, 5), np.uint8)
    mask_solida = cv2.morphologyEx(mask_solida, cv2.MORPH_OPEN, kernel_open)

    return mask_solida

def diferencia_frames(frame1, frame2):
    """
    Devuelve la cantidad de píxeles que difieren entre 2 frames binarios.
    """
    delta = cv2.absdiff(frame1, frame2)
    cambios_pixels = cv2.countNonZero(delta)
    return np.sum(cambios_pixels)    

def identificar_valor(dado_roi):
    """
    Dado el ROI de un dado segmentado, cuenta los puntos blancos (valor del dado).
    """
    roi_hsv = cv2.cvtColor(dado_roi, cv2.COLOR_BGR2HSV)
    
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    
    mask_white = cv2.inRange(roi_hsv, lower_white, upper_white)

    # Limpieza suave para los puntos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
    
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_white, connectivity=8
    )
    
    return n_labels - 1  # Restar 1 para no contar el fondo
        

def procesar_frames_estaticos(buffer_frames, frame_estatico_mask, out):
    """
    Procesa una lista de frames estáticos de dados quietos.
    Escribe los frames procesados en el video de salida.
    """
    PADDING = 5
    dados = []

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(frame_estatico_mask, connectivity=8)                    
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        x -= PADDING
        y -= PADDING
        w += 2 * PADDING
        h += 2 * PADDING

        x *= 4; y *= 4; w *= 4; h *= 4 # Volver a resolucíon original hd

        dado_roi = buffer_frames[0][y:y+h, x:x+w]
        valor = identificar_valor(dado_roi)
        print(f'    Dado identificado. Valor: {valor}')
        dados.append((x, y, w, h, valor))

    print(f'  Generando video anotado con {len(dados)} dados identificados.')

    for f in buffer_frames: # Anotar y guardar cada frame de dados quietos
        for (x, y, w, h, valor) in dados:
            cv2.rectangle(f, (x, y), (x+w, y+h), (255, 255, 0), 3)
            cv2.putText(f, str(valor), (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        out.write(f)


def procesar_video_tirada(video):
    print('\n\nProcesando video:', video)
    cap = cv2.VideoCapture(video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 

    if not cap.isOpened():
        print("Error al abrir video.")
        return

    out = cv2.VideoWriter(video.replace('.mp4', '-anotado.mp4'),
                      cv2.VideoWriter_fourcc(*'mp4v'), 
                      fps, (width, height))
    
    ret, frame = cap.read()
    out.write(frame)

    frame_prev = cv2.resize(frame, dsize=(int(width/4), int(height/4)))
    frame_prev_mask = mascara_roja(frame_prev)

    frame_estatico_mask = None # Máscara de 1 frame 'del medio' con dados quietos
    frame_count = 1    
    frames_permitidos = 0
    buffer_frames = []

    while cap.isOpened():
        ret, frame_hd = cap.read()
        if not ret: break
        frame = cv2.resize(frame_hd, dsize=(int(width/4), int(height/4)))
        frame_count += 1
        frame_mask = mascara_roja(frame)

        if not frame_mask.any():
            for f in buffer_frames:
                out.write(f)
            buffer_frames = []
            out.write(frame_hd)
            continue

        if diferencia_frames(frame_prev_mask, frame_mask) <= 50: # Frame similar al anterior
            frame_prev_mask = frame_mask.copy()
            if frames_permitidos < 5:
                frames_permitidos += 1
            buffer_frames.append(frame_hd)

            if len(buffer_frames) == 5: 
                frame_estatico_mask = frame_mask.copy() # Guardar máscara de un frame 'del medio'
        
        elif frames_permitidos and len(buffer_frames) > 5: # Frames permitidos
            # No copia la mascara actual, sigue usando la del ultimo frame similar.
            frames_permitidos -= 1
            buffer_frames.append(frame_hd)

        else: # Frame diferente al anterior y no hay frames permitidos
            frame_prev_mask = frame_mask.copy()
            
            if len(buffer_frames) > 10: # Asume dados quietos. Procesa buffer con anotaciones
                print('  Dados quietos entre frames', frame_count - len(buffer_frames), 'y', frame_count - 1)
                procesar_frames_estaticos(buffer_frames, frame_estatico_mask, out)
            
            else: # Escribe sin procesar.
                for f in buffer_frames:
                    out.write(f)

            buffer_frames = []
            out.write(frame_hd)

    if buffer_frames: # Si el video terminó y hay frames en el buffer
        if len(buffer_frames) > 10:
            print('  Dados quietos entre frames', frame_count - len(buffer_frames), 'y', frame_count)
            procesar_frames_estaticos(buffer_frames, frame_estatico_mask, out)
        else:
            for f in buffer_frames:
                out.write(f)


    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f'  Video guardado: {video.replace(".mp4", "-anotado.mp4")}')

if __name__ == "__main__":
    videos = ['tirada_1.mp4', 'tirada_2.mp4', 'tirada_3.mp4', 'tirada_4.mp4']
    
    for video in videos:
        if os.path.exists(video):
            procesar_video_tirada(video)
        else:
            print(f"Video no encontrado: {video}")
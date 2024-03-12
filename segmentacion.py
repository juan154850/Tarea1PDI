import cv2
import numpy as np
import time

# Cargar el video
video_path = './video.mp4'  # Cambia esto al camino de tu video
cap = cv2.VideoCapture(video_path)
# Capturar fps del video
fps = cap.get(cv2.CAP_PROP_FPS)

# Crear una ventana y configurar la función de callback de mouse
cv2.namedWindow('Video')
cv2.namedWindow('Video Masks')

counter = 0
last_detection_time = time.time()
cooldown_duration = 2  # segundos de espera entre detecciones

# Abrir el video
while True:
    ret, frame = cap.read()

    # Parametrizaciones
    scale_percent = 20  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    frame2 = cv2.resize(cv2.cvtColor(np.copy(frame), cv2.COLOR_BGR2HSV), dim, interpolation=cv2.INTER_AREA)

    lower_pink = np.array([145, 100, 100])
    upper_pink = np.array([165, 255, 255])
    pink_mask = cv2.inRange(frame2, lower_pink, upper_pink)

    # Kernel
    kernel = np.ones((5, 5), np.uint8)

    # Umbrales
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(frame2, lower_blue, upper_blue)

    blue_mask = cv2.erode(blue_mask, kernel, iterations=1)
    blue_mask = cv2.dilate(blue_mask, kernel, iterations=1)

    # Contornos
    contours_pink, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos
    cv2.drawContours(frame2, contours_pink, -1, (0, 255, 0), 1)
    cv2.drawContours(frame2, contours_blue, -1, (0, 0, 255), 1)

    cv2.putText(frame2, f'Counter: {counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    
    if not ret:
        break

    cv2.imshow('Video', frame2)
    cv2.imshow('Video Masks', pink_mask)
    cv2.imshow('Video Ball Masks', blue_mask)

    
    tolerancia = 2
    current_time = time.time()
    # Verificar intersección de contornos
    for contour_pink in contours_pink:
        for contour_blue in contours_blue:
            for point in contour_blue:
                x, y = point[0]
                result = cv2.pointPolygonTest(contour_pink, (int(x), int(y)), True)
                if abs(result) < tolerancia and (current_time - last_detection_time) > cooldown_duration:
                    counter +=1
                    last_detection_time = current_time
                    print(f"counter: {counter}")

    key = cv2.waitKey(int(30)) & 0xFF

    if key == 27:  # Presiona la tecla 'Esc' para salir
        break

cap.release()
cv2.destroyAllWindows()

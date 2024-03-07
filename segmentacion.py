import cv2
import numpy as np

# Cargar el video
video_path = './video.mp4'  # Cambia esto al camino de tu video
cap = cv2.VideoCapture(video_path)
#Capturar fps del video
fps = cap.get(cv2.CAP_PROP_FPS)

# Crear una ventana y configurar la función de callback de mouse
cv2.namedWindow('Video')

#
cv2.namedWindow('Video Masks')

#Abrir el video
while True:
    ret, frame = cap.read()

    #Parametrizaciones
    scale_percent = 40 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    frame2 = cv2.resize(cv2.cvtColor(np.copy(frame), cv2.COLOR_BGR2HSV), dim, interpolation=cv2.INTER_AREA)

    lower_pink = np.array([145, 100, 100])
    upper_pink = np.array([165, 255, 255])
    pink_mask = cv2.inRange(frame2, lower_pink, upper_pink)

    # Kernel
    kernel = np.ones((5,5),np.uint8)

    # Umbrales
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(frame2, lower_blue, upper_blue)

    blue_mask = cv2.erode(blue_mask, kernel, iterations = 1)
    blue_mask = cv2.dilate(blue_mask, kernel, iterations = 1)

    if not ret:
        break
    
    cv2.imshow('Video', frame2)
    cv2.imshow('Video Masks', pink_mask)
    cv2.imshow('Video Ball Masks', blue_mask)

    key = cv2.waitKey(int(30)) & 0xFF
    
    if key == 27:  # Presiona la tecla 'Esc' para salir
        break

cap.release()
cv2.destroyAllWindows()


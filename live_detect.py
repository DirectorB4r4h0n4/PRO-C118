import cv2

# Crea la variable body_classifier para asignar el archivo CascadeClassifier
body_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define un objeto video capture
vid = cv2.VideoCapture(0)

while True:
    # Captura el video cuadro por cuadro
    ret, frame = vid.read()

    # Convierte cada cuadro a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pasa cada cuadro al clasificador
    faces = body_classifier.detectMultiScale(gray, 1.1, 5)

    # Dibuja un rectángulo alrededor de cada cuerpo detectado
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (225, 0, 0), 2)
        
    # Muestra el cuadro de resultado
    cv2.imshow("Web cam", frame)
      
    # Cierra la ventana con la tecla espaciadora
    if cv2.waitKey(25) == 32:
        break
  
# Después del bucle, libera el objeto capturado
vid.release()

# Cierra todas las ventanas
cv2.destroyAllWindows()

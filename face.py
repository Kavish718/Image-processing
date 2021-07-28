import cv2
import numpy as np

cam = cv2.VideoCapture(0)
while True:
    _,img = cam.read()
    cv2.imshow("original",img)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('out',img)
    if(cv2.waitKey(1) & 0xff ==ord('q')):
        cv2.destroyAllWindows()
        break
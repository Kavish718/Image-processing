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
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
        #obtaining the pic face separately to apply blur
        face_pic=img[y:y+h,x:x+w]
        #applying gaussian blur to the face
        face_pic=cv2.GaussianBlur(face_pic,(15,15),25)
        #merging the blurred face onto the original image
        row,col,chennels=face_pic.shape
        img[y:y+row, x:x+col]=face_pic
    cv2.imshow('blurredface',img)
    if(cv2.waitKey(1) & 0xff ==ord('q')):
        cv2.destroyAllWindows()
        break

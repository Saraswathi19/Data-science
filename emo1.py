import cv2
import numpy as np
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier("C:\Users\saras\Downloads\haarcascade_frontalface_alt.xml")
cap = cv2.VideoCapture(0)
scaling_factor = 1
while True:
 ret, frame = cap.read()
 frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
 faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3)
 for (x, y, w, h) in faces:
   face = frame[y:y+h, x:x+w]
   emotions = DeepFace.analyze(face, actions=['emotion'],enforce_detection=False)


   emotion_text = "Emotion: " + emotions[0]['dominant_emotion']

   print(emotion_text)


   cv2.putText(frame, emotion_text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
   cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

   cv2.imshow('frame', frame)

 if cv2.waitKey(1) == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
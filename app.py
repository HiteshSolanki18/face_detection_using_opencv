# import library
import cv2
from sklearn.preprocessing import scale

# import train cascade file for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# object of camera
vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # Reading every frame one by one
    _,frame = vc.read()

    # converting that frame in Gray
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # detecting face objects
    faces = face_cascade.detectMultiScale(gray,1.1,4)

    # Drawing rectange on faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    # Showing Each Frame
    cv2.imshow('frame',frame)

    # waitkey and exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# release videocapture object
vc.release()

# destroy all the window
cv2.destroyAllWindows()
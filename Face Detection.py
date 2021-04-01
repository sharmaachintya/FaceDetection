import cv2
import os

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
alg = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
#alg="haarcascade_frontalface_default.xml"
haar_cascade=cv2.CascadeClassifier(alg)
cam=cv2.VideoCapture(0)

count=1
while True:
    print(count)
    _,img=cam.read()
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=haar_cascade.detectMultiScale(grayImg,1.3,4)
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        count=count+1
    cv2.imshow("FaceDetection",img)
    key=cv2.waitKey(10)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()

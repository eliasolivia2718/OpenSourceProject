# simpleCamTest.py
import numpy as np
import cv2

#cascade classifier 를 등록
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#얼굴과 눈을 찾는 함수
def detect(gray,frame):
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5,minSize=(100,100),flags=cv2.CASCADE_SCALE_IMAGE)
    
    # 얼굴에 사각형 그리고 눈 찾기
    for(x,y,w,h) in faces:
        #이미지프레임의 얼굴 좌상단(x,y)에서 시작해서 (x+w,y+h)까지의 사각형을 만든다(색(255 0 0), 굵기 2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        # 이미지를 얼굴 크기만큼 잘라서 그레이스케일, 컬러이미지로 만듬
        face_gray=gray[y:y+h,x:x+w]
        face_color=frame[y:y+h,x:x+w]
        
        # 얼굴영역에서만 (fcae_gray) 눈을 찾음
        eyes=eyeCascade.detectMultiScale(face_gray,1.1,3)
        
        # 이미지프레임의 누 좌상단(x,y)에서 시작해서 (x+w,y+h)까지의 사각형을 만든다(색(0 255 0), 굵기 2) 
        for(ex, ey,ew,eh) in eyes:
            cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
    return frame

#웹캠에서 이미지 가져오기
video_capture = cv2.VideoCapture(0)

while(True):
    #웹캠 이미지를 프레임으로 자름
    _, frame =video_capture.read()
    # 그리고 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 만들어준 얼굴
    canvas=detect(gray,frame)
    #이미지를 지속적으로 윈도우에 띄움
    cv2.imshow('canvas',canvas)
    #esc누르면 종료
    if cv2.waitKey(30) == 27: #esc
        break
video_capture.release()
cv2.destroyAllWindows()
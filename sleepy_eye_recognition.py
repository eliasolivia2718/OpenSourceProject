import numpy as np
import cv2
import dlib

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 얼굴의 각 구역의 포인트들을 구분해 놓기
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

""" 
    def = dlib를 이용 얼굴과 눈을 찾는 함수
    input = 그레이 스케일 이미지
    output = 얼굴 중요 68개의 포인트 에 그려진 점 + 이미지
"""


def detect(gray, frame):
    # 일단, 등록한 Cascade classifier 를 이용 얼굴을 찾음
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    # 얼굴에서 랜드마크를 찾자
    for (x, y, w, h) in faces:
        # 오픈 CV 이미지를 dlib용 사각형으로 변환하고
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        # 랜드마크 포인트들 지정
        landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dlib_rect).parts()])
        # 원하는 포인트들을 landmarks_list에 넣는다
        landmarks_list = RIGHT_EYE_POINTS + LEFT_EYE_POINTS

        landmarks_display = []
        for index in landmarks_list:
            landmarks_display.append(landmarks[index])

        # 밑에서 얻은 눈의 좌표를 저장하게될 리스트
        pos_list = []

        # 포인트 출력
        for idx, point in enumerate(landmarks_display):
            pos = (point[0, 0], point[0, 1])
            pos_list.append(pos)
            cv2.circle(frame, pos, 2, color=(0, 255, 255), thickness=-1)

        # 눈들의 좌표를 기반으로 가로 세로 길이 구하기
        rightEye_width = pos_list[3][0] - pos_list[0][0]
        rightEye_height = (pos_list[4][1] + pos_list[5][1]) - (pos_list[1][1] + pos_list[2][1])

        leftEye_width = pos_list[9][0] - pos_list[6][0]
        leftEye_height = (pos_list[10][1] + pos_list[11][1]) - (pos_list[7][1] + pos_list[8][1])

        # 화면에 출력
        cv2.putText( frame, text=f'RightEye: ({rightEye_width:0.4f}, {rightEye_height:0.4f}), \n'
                                 f'LeftEye: ({leftEye_width:0.4f}, {leftEye_height:0.4f})'
                     , org=(10,30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                 fontScale=0.5,color=(0,0,255), thickness=2)

    return frame


# 웹캠에서 이미지 가져오기
# 동영상일 경우 아래 경로 지정해서 사용할 수 있음
# 경로지정 or 웹캠 장치 번호
file_path = 1
video_capture = cv2.VideoCapture(file_path)

while True:
    # 웹캠 이미지를 프레임으로 자름
    _, frame = video_capture.read()
    # 그리고 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 만들어준 얼굴 눈 찾기
    canvas = detect(gray, frame)
    # 찾은 이미지 보여주기
    cv2.imshow("Sleepy Eye Recognition", canvas)

    # q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 끝
video_capture.release()
cv2.destroyAllWindows()

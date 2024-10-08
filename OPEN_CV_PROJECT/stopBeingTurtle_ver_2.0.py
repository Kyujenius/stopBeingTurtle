import cv2  # OpenCV: 이미지 및 비디오 처리 라이브러리
import mediapipe as mp  # MediaPipe: 구글의 멀티미디어 처리 라이브러리
import numpy as np  # NumPy: 수치 연산 라이브러리
import time
import telebot

# 봇 설정
token = '-:-'  # 봇 토큰 (BotFather에서 받은 토큰으로 교체)
bot = telebot.TeleBot(token=token)
user_id = 7287129421  # 사용자 ID (알림을 받을 사용자의 Telegram ID)

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils  # 랜드마크 그리기 유틸리티
mp_pose = mp.solutions.pose  # 자세 추정 모델
mp_face_detection = mp.solutions.face_detection  # 얼굴 감지 모델

# 알림 기준 값
ALERT_RIGHT_ANGLE_THRESHOLD = 70  # 오른쪽 어깨-귀 각도 기준 (사용자 설정)
ALERT_LEFT_ANGLE_THRESHOLD = 110  # 왼쪽 어깨-귀 각도 기준 (사용자 설정)
FACE_SCALE_THRESHOLD = 1.5  # 얼굴 크기 증가 비율 기준
CHIN_TO_STERNUM_THRESHOLD = 0.8  # 턱-명치 거리 감소 비율 기준
ALERT_INTERVAL = 5  # 알림 간격 (초)

# 변수 초기화
initial_face_size = None
initial_chin_to_sternum_distance = None
alert_time = None
isFaceSizeVersion = True

# 비디오 캡처 객체 생성 (웹캠 사용)
cap = cv2.VideoCapture(1)  # 0: 기본 웹캠, 다른 카메라 사용 시 번호 변경

# 각도 계산 함수 (두 점 사이의 각도 계산)
def calculate_angle(a, b):
    a = np.array(a)
    b = np.array(b)
    radians = np.arctan2(b[1] - a[1], b[0] - a[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# MediaPipe Pose 및 Face Detection 객체 생성
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # BGR 이미지를 RGB로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 얼굴 감지
        face_results = face_detection.process(image)

        # 자세 추정
        results = pose.process(image)

        # RGB 이미지를 다시 BGR로 변환
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape

        try:
            if(isFaceSizeVersion):
            # 얼굴 크기 측정
                if face_results.detections:
                    for detection in face_results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        face_width = int(bbox.width * image_width)
                        face_height = int(bbox.height * image_height)
                        face_size = (face_width + face_height) / 2

                        if initial_face_size is None:
                            initial_face_size = face_size

                        face_scale = face_size / initial_face_size

                        cv2.putText(image, f"Face Scale: {face_scale:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


                        if face_scale > FACE_SCALE_THRESHOLD:
                            current_time = time.time()
                            if alert_time is None or current_time - alert_time >= ALERT_INTERVAL:
                                print("Alert: 얼굴이 가까워졌습니다. 자세를 뒤로 조정하세요.")
                                bot.send_message(user_id, "얼굴이 가까워졌습니다. 자세를 뒤로 조정하세요.")
                                alert_time = current_time
            else:
                # 턱-명치 거리 측정
                landmarks = results.pose_landmarks.landmark
                # 턱 끝
                nose = landmarks[mp_pose.PoseLandmark.NOSE]
                # 명치 
                sternum = ((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2,
                        (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2)
                chin_to_sternum_distance = np.linalg.norm(np.array([nose.x, nose.y]) - np.array(sternum))

                if initial_chin_to_sternum_distance is None:
                    initial_chin_to_sternum_distance = chin_to_sternum_distance

                chin_to_sternum_ratio = chin_to_sternum_distance / initial_chin_to_sternum_distance

                cv2.putText(image, f"Chin to Sternum Ratio: {chin_to_sternum_ratio:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if chin_to_sternum_ratio < CHIN_TO_STERNUM_THRESHOLD:
                    current_time = time.time()
                    if alert_time is None or current_time - alert_time >= ALERT_INTERVAL:
                        print("Alert: 고개를 숙이고 있습니다. 자세를 바르게 하세요.")
                        bot.send_message(user_id, "고개를 숙이고 있습니다. 자세를 바르게 하세요.")
                        alert_time = current_time

        except Exception as e:
            print(f"Error processing landmarks: {e}")

        # 결과 이미지 표시
        cv2.imshow('MediaPipe Pose', image)  # 창 이름: 'MediaPipe Pose'

        if cv2.waitKey(5) & 0xFF == 27:  # ESC 키를 누르면 종료 (5ms 대기)
            break

# 웹캠 및 OpenCV 창 해제
cap.release()
cv2.destroyAllWindows()

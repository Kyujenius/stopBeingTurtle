import cv2
import mediapipe as mp
import numpy as np
import time
import telebot
import tkinter as tk
from tkinter import ttk
from PIL import Image,ImageTk

# 봇 설정
token = '7366178157:AAG2MOVxAZXB2kB3l9r7uGCom5jP8DnR9dQ'  # 봇 토큰 (BotFather에서 받은 토큰으로 교체)
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
CHIN_TO_STERNUM_THRESHOLD = 0.7  # 턱-명치 거리 감소 비율 기준
ALERT_INTERVAL = 5  # 알림 간격 (초)

# 변수 초기화
initial_face_size = None
initial_chin_to_sternum_distance = None
alert_time = None
is_measuring = False
measurement_type = None
countdown = 0
countdown_duration = 7

# 비디오 캡처 객체 생성 (웹캠 사용)
cap = cv2.VideoCapture(0)  # 0: 기본 웹캠, 다른 카메라 사용 시 번호 변경

# 각도 계산 함수
def calculate_angle(a, b):
    a = np.array(a)
    b = np.array(b)
    radians = np.arctan2(b[1] - a[1], b[0] - a[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Tkinter GUI 생성
root = tk.Tk()
root.title("자세 측정 프로그램")

# 이미지 로드 및 표시
image_path = "./OPEN_CV_PROJECT/turtle_neck.png"  # 이미지 파일 경로 (실제 경로로 변경)
image = Image.open(image_path)
image = image.resize((300, 200))  # 이미지 크기 조정 (선택 사항)
photo = ImageTk.PhotoImage(image)
image_label = ttk.Label(root, image=photo)
image_label.pack(pady=10)

# 버튼 클릭 이벤트 처리 함수
def start_measurement(selected_type):
    global measurement_type, countdown, is_measuring
    measurement_type = selected_type
    countdown = 7
    is_measuring = False
    root.destroy() 

def end_program():
    # 웹캠 및 OpenCV 창 해제
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()  # GUI 창 닫기


# 버튼 생성
ttk.Button(root, text="턱-명치 거리 측정", command=lambda: start_measurement('chin_to_sternum')).pack(pady=5)
ttk.Button(root, text="얼굴 크기 변화 측정", command=lambda: start_measurement('face_size')).pack(pady=5)
ttk.Button(root, text="닫기", command=lambda: end_program()).pack(pady=5)

description = ttk.Label(root, text="측정 방식을 택한 뒤 7초 뒤에 측정이 시작됩니다.\n노트북위치와, 최대한 바른 자세를 초기에 잡아주세요!")
description.pack()

# GUI 실행
root.mainloop()

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
            # 카운트다운 및 초기값 설정
            if countdown > 0:
                cv2.putText(image, str(countdown), (image_width // 2, image_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                time.sleep(1)
                countdown -= 1

                if countdown == 0:
                    is_measuring = True
                    if measurement_type == 'face_size':
                        if face_results.detections:
                            for detection in face_results.detections:
                                bbox = detection.location_data.relative_bounding_box
                                face_width = int(bbox.width * image_width)
                                face_height = int(bbox.height * image_height)
                                initial_face_size = (face_width + face_height) / 2
                    elif measurement_type == 'chin_to_sternum':
                        landmarks = results.pose_landmarks.landmark
                        nose = landmarks[mp_pose.PoseLandmark.NOSE]
                        sternum = ((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2,
                                    (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2)
                        initial_chin_to_sternum_distance = np.linalg.norm(np.array([nose.x, nose.y]) - np.array(sternum))
                    print("초기값 설정 완료!")

            # 측정 및 알림 (측정 중일 때만 실행)
            if is_measuring:
                if measurement_type == 'face_size':
                    # 얼굴 크기 측정 및 시각화
                    if face_results.detections:
                        for detection in face_results.detections:
                            bbox = detection.location_data.relative_bounding_box
                            x, y, w, h = int(bbox.xmin * image_width), int(bbox.ymin * image_height), \
                                         int(bbox.width * image_width), int(bbox.height * image_height)
                            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 얼굴 영역 표시

                            face_size = (w + h) / 2
                            face_scale = face_size / initial_face_size
                            cv2.putText(image, f"Face Scale: {face_scale:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            if face_scale > FACE_SCALE_THRESHOLD:
                                current_time = time.time()
                                if alert_time is None or current_time - alert_time >= ALERT_INTERVAL:
                                    print("Alert: 얼굴이 가까워졌습니다. 자세를 뒤로 조정하세요.")
                                    bot.send_message(user_id, "🐢")
                                    alert_time = current_time
                elif measurement_type == 'chin_to_sternum':
                    # 턱-명치 거리 측정 및 시각화
                    landmarks = results.pose_landmarks.landmark
                    nose = landmarks[mp_pose.PoseLandmark.NOSE]
                    sternum = ((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2,
                               (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2)
                    chin_to_sternum_distance = np.linalg.norm(np.array([nose.x, nose.y]) - np.array(sternum))

                    nose_px = (int(nose.x * image_width), int(nose.y * image_height))
                    sternum_px = (int(sternum[0] * image_width), int(sternum[1] * image_height))
                    cv2.line(image, nose_px, sternum_px, (0, 255, 0), 2)  # 턱-명치 선 표시

                    chin_to_sternum_ratio = chin_to_sternum_distance / initial_chin_to_sternum_distance
                    cv2.putText(image, f"Chin to Sternum Ratio: {chin_to_sternum_ratio:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if chin_to_sternum_ratio < CHIN_TO_STERNUM_THRESHOLD:
                        current_time = time.time()
                        if alert_time is None or current_time - alert_time >= ALERT_INTERVAL:
                            print("Alert: 고개를 숙이고 있습니다. 자세를 바르게 하세요.")
                            bot.send_message(user_id, "🐢")
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

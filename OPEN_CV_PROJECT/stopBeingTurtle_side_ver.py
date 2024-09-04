import cv2  # OpenCV: 이미지 및 비디오 처리 라이브러리
import mediapipe as mp  # MediaPipe: 구글의 멀티미디어 처리 라이브러리
import numpy as np  # NumPy: 수치 연산 라이브러리
import time
import telebot

# 봇 설정
token = '-:-'
bot = telebot.TeleBot(token=token)
user_id = 7287129421  # 예시 사용자 ID (홍규진 telegramId)

updates = bot.get_updates()

# 새로운 내용을 출력하는데, 이때 ID값도 같이 나옴
for u in updates:
    print(u.message)


# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils  # 랜드마크 그리기 유틸리티
mp_pose = mp.solutions.pose  # 자세 추정 모델

# 알림 기준 각도 (예: 20도)
ALERT_RIGHT_ANGLE_THRESHOLD = 70  # 사용자가 설정 가능한 알림 기준 각도

ALERT_LEFT_ANGLE_THRESHOLD = 110  # 사용자가 설정 가능한 알림 기준 각도

# 어깨 너비 기준 값 (픽셀 단위, 사용자 설정)
SHOULDER_WIDTH_THRESHOLD = 200

# 알림 관련 변수
alert_time = None  # 마지막 알림 시간
alert_interval = 5  # 알림 간격 (초)

# 카메라 목록 가져오기
def get_available_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

# GUI 함수 (카메라 선택)
def select_camera():
    available_cameras = get_available_cameras()
    print("사용 가능한 카메라:")
    for i, camera_id in enumerate(available_cameras):
        print(f"{i+1}. 카메라 {camera_id}")

    while True:
        try:
            choice = int(input("카메라 선택 (번호 입력): ")) - 1
            if choice in available_cameras:
                return choice
            else:
                print("잘못된 입력입니다. 다시 선택하세요.")
        except ValueError:
            print("숫자를 입력해야 합니다.")


camera_id = select_camera()

# 비디오 캡처 객체 생성 (웹캠 사용)
cap = cv2.VideoCapture(camera_id)


# 각도 계산 함수 (두 점 사이의 각도 계산)
def calculate_angle(a, b):
    a = np.array(a)  # 점 a를 NumPy 배열로 변환
    b = np.array(b)  # 점 b를 NumPy 배열로 변환
    radians = np.arctan2(b[1] - a[1], b[0] - a[0])  # 라디안 계산
    angle = np.abs(radians * 180.0 / np.pi)  # 라디안을 각도로 변환
    if angle > 180.0:  # 각도가 180도를 넘으면 반대 방향 각도 계산
        angle = 360 - angle
    return angle


# MediaPipe Pose 객체 생성 (자세 추정 모델 초기화)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():  # 웹캠이 열려 있는 동안 반복
        success, image = cap.read()  # 웹캠 프레임 읽기 (success: 성공 여부, image: 프레임 이미지)
        if not success:  # 프레임 읽기에 실패하면 다음 프레임으로 넘어감
            print("Ignoring empty camera frame.")
            continue

        # BGR 이미지를 RGB로 변환 (MediaPipe는 RGB 이미지 사용)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # 성능 향상을 위해 이미지 수정 불가능 설정
        results = pose.process(image)  # 자세 추정 모델 실행

        # RGB 이미지를 다시 BGR로 변환 (OpenCV 표시를 위해)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 랜드마크 감지 및 표시
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) 
        # 감지된 랜드마크를 이미지에 그리기 (랜드마크, 연결선 포함)

        # 귀와 어깨 중심점 좌표 추출
        try:  # 랜드마크 감지 실패 시 예외 처리
            landmarks = results.pose_landmarks.landmark  # 감지된 랜드마크 정보
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]  #왼쪽 귀 랜드마크
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]  #오른쪽 귀 랜드마크
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]  # 왼쪽 어깨 랜드마크
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]  # 오른쪽 어깨 랜드마크

            #귀와 어깨 중심점을 이미지 좌표로 변환 (0~1 범위 값을 픽셀 좌표로 변환)
            image_height, image_width, _ = image.shape
            left_ear_px = (int(left_ear.x * image_width), int(left_ear.y * image_height))
            right_ear_px = (int(right_ear.x * image_width), int(right_ear.y * image_height))
            left_shoulder_px = (int(left_shoulder.x * image_width), int(left_shoulder.y * image_height))
            right_shoulder_px = (int(right_shoulder.x * image_width), int(right_shoulder.y * image_height))
            
            shoulder_width = abs(left_shoulder_px[0] - right_shoulder_px[0])

            # 측면 자세 확인
            is_side_view = shoulder_width < SHOULDER_WIDTH_THRESHOLD


            # 각도 계산 (왼쪽 및 오른쪽 어깨-귀)
            left_angle = calculate_angle(left_ear_px, left_shoulder_px)
            right_angle = calculate_angle(right_ear_px, right_shoulder_px)

             # 귀와 어깨 연결선 그리기
            cv2.line(image, left_ear_px, left_shoulder_px, (0, 255, 0), 2)  # 왼쪽 귀-어깨 연결선 (녹색)
            cv2.line(image, right_ear_px, right_shoulder_px, (0, 255, 0), 2)  # 오른쪽 귀-어깨 연결선 (녹색)

            # 각도 표시
            cv2.putText(image, f"Left Angle: {left_angle:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Right Angle: {right_angle:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


            # 알림 조건 확인 및 알림 출력
            if is_side_view:
                current_time = time.time()
                if (left_angle < ALERT_RIGHT_ANGLE_THRESHOLD and right_angle < ALERT_RIGHT_ANGLE_THRESHOLD) or (left_angle > ALERT_LEFT_ANGLE_THRESHOLD and right_angle > ALERT_LEFT_ANGLE_THRESHOLD) :
                    if alert_time is None or current_time - alert_time >= alert_interval:
                        print("Alert: 거북목이 의심됩니다! 자세를 교정하세요. 🐢")
                        alert_time = current_time
                        bot.send_message(user_id, "🐢")

                elif (alert_time is None or current_time - alert_time >= alert_interval):
                    if alert_time is None or current_time - alert_time >= alert_interval:
                        print("Alert: 잘 하고 있어요~ 🐰")
                        alert_time = current_time
            else:
                current_time = time.time()
                if alert_time is None or current_time - alert_time >= alert_interval:
                    print("Alert: 정확한 측정을 위하여 측면으로 돌아주세요.")
                    alert_time = current_time

        except Exception as e:  # 예외 발생 시 에러 메시지 출력
            print(f"Error processing landmarks: {e}")

        # 결과 이미지 표시
        cv2.imshow('MediaPipe Pose', image)  # 창 이름: 'MediaPipe Pose'

        if cv2.waitKey(5) & 0xFF == 27:  # ESC 키를 누르면 종료 (5ms 대기)
            break

# 웹캠 및 OpenCV 창 해제
cap.release()
cv2.destroyAllWindows()

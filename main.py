# main.py — Распознавание жестов и свайпов + кнопка Close

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
from collections import deque

# Загрузка модели MediaPipe
model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = HandLandmarker.create_from_options(options)

# Отрисовка скелета руки (точки и соединения)
def draw_landmarks_on_image(rgb_image, landmarks):
    h, w, _ = rgb_image.shape
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(rgb_image, (x, y), 2, (0, 255, 0), -1)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]
    for start, end in connections:
        pt1 = (int(landmarks[start].x * w), int(landmarks[start].y * h))
        pt2 = (int(landmarks[end].x * w), int(landmarks[end].y * h))
        cv2.line(rgb_image, pt1, pt2, (0, 255, 0), 1)
    return rgb_image

# Детектор свайпов (активируется жестом V: указательный + средний)
class VSwipeDetector:
    def __init__(self, buffer_size=8, threshold=0.06):
        self.buffer = deque(maxlen=buffer_size)
        self.threshold = threshold
        self.last_swipe_time = 0

    def update(self, landmarks, frame_w, frame_h):
        # Проверка жеста V
        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        ring_down = not (landmarks[16].y < landmarks[14].y)
        pinky_down = not (landmarks[20].y < landmarks[18].y)
        thumb_down = landmarks[4].y > landmarks[3].y

        if not (index_up and middle_up and ring_down and pinky_down and thumb_down):
            self.buffer.clear()
            return None

        self.buffer.append((landmarks[0].x, landmarks[0].y))
        if len(self.buffer) < self.buffer.maxlen:
            return None

        start_x, start_y = self.buffer[0]
        end_x, end_y = self.buffer[-1]
        dx = end_x - start_x
        dy = end_y - start_y

        swipe = None
        if abs(dx) > self.threshold and abs(dx) > abs(dy):
            swipe = 'right' if dx > 0 else 'left'
        elif abs(dy) > self.threshold and abs(dy) > abs(dx):
            swipe = 'down' if dy > 0 else 'up'

        if swipe and time.time() - self.last_swipe_time > 1.0:
            self.last_swipe_time = time.time()
            self.buffer.clear()
            return swipe
        return None

# Классификатор статических жестов (сердечко, окей, палец вверх, палец вниз)
class GestureClassifier:
    def is_finger_up(self, landmarks, tip_idx, pip_idx):
        return landmarks[tip_idx].y < landmarks[pip_idx].y

    def distance(self, a, b):
        return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

    def recognize(self, landmarks):
        if landmarks is None:
            return None

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        wrist = landmarks[0]

        # Сердечко пальцами (finger heart)
        dist_heart = self.distance(thumb_tip, index_tip)
        if dist_heart < 0.08 and thumb_tip.y < wrist.y + 0.1 and index_tip.y < wrist.y + 0.1:
            if not self.is_finger_up(landmarks, 12, 10) and \
               not self.is_finger_up(landmarks, 16, 14) and \
               not self.is_finger_up(landmarks, 20, 18):
                return "finger_heart"

        # Окей (кольцо из большого и указательного)
        dist_ok = self.distance(thumb_tip, index_tip)
        if dist_ok < 0.08:
            if self.is_finger_up(landmarks, 12, 10) and \
               self.is_finger_up(landmarks, 16, 14) and \
               self.is_finger_up(landmarks, 20, 18):
                return "ok"

        # Палец вверх
        thumb_up = self.is_finger_up(landmarks, 4, 3)
        index_down = not self.is_finger_up(landmarks, 8, 6)
        middle_down = not self.is_finger_up(landmarks, 12, 10)
        ring_down = not self.is_finger_up(landmarks, 16, 14)
        pinky_down = not self.is_finger_up(landmarks, 20, 18)

        if thumb_up and index_down and middle_down and ring_down and pinky_down:
            return "thumbs_up"

        # Палец вниз
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_down = thumb_tip.y > thumb_ip.y

        if thumb_down and index_down and middle_down and ring_down and pinky_down:
            return "thumbs_down"

        return None

# Инициализация детекторов
swipe_detector = VSwipeDetector()
gesture_classifier = GestureClassifier()

# Настройка камеры
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Кнопка CLOSE для выхода мышкой
close_button_rect = None
exit_flag = False

def mouse_callback(event, x, y, flags, param):
    global exit_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        if close_button_rect is not None:
            x1, y1, x2, y2 = close_button_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                exit_flag = True

cv2.namedWindow('Gesture Control')
cv2.setMouseCallback('Gesture Control', mouse_callback)

# Основной цикл обработки видео
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]
        frame = draw_landmarks_on_image(frame, hand_landmarks)
        lm_list = hand_landmarks

        # Распознавание свайпов
        swipe = swipe_detector.update(lm_list, w, h)
        if swipe:
            print(f"Swipe: {swipe}")

        # Распознавание статических жестов
        gesture = gesture_classifier.recognize(lm_list)
        if gesture:
            print(f"Gesture: {gesture}")

    # Отрисовка кнопки CLOSE
    btn_x1, btn_y1 = w - 110, 10
    btn_x2, btn_y2 = w - 10, 50
    close_button_rect = (btn_x1, btn_y1, btn_x2, btn_y2)
    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 0, 255), -1)
    cv2.putText(frame, "CLOSE", (btn_x1 + 5, btn_y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Gesture Control', frame)
    if exit_flag or (cv2.waitKey(5) & 0xFF == 27):
        break

cap.release()
cv2.destroyAllWindows()
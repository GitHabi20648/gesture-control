from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np


MODEL_PATH = Path(__file__).resolve().with_name("hand_landmarker.task")
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


class HandLandmarker:
    def __init__(self) -> None:
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._landmarker = (
            mp.tasks.vision.HandLandmarker.create_from_options(options)
        )
        self._last_timestamp_ms = -1

    def detect(
        self, rgb_frame: np.ndarray
    ) -> mp.tasks.vision.HandLandmarkerResult:
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )

        # VIDEO mode requires strictly increasing timestamps.
        current_timestamp_ms = time.monotonic_ns() // 1_000_000
        timestamp_ms = max(current_timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self) -> None:
        self._landmarker.close()


def draw_landmarks_on_image(
    rgb_image: np.ndarray,
    detection_result: mp.tasks.vision.HandLandmarkerResult,
) -> np.ndarray:
    annotated_image = np.copy(rgb_image)
    height, width = annotated_image.shape[:2]

    for hand_landmarks in detection_result.hand_landmarks:
        points = [
            (
                round(landmark.x * (width - 1)),
                round(landmark.y * (height - 1)),
            )
            for landmark in hand_landmarks
        ]

        for start_index, end_index in HAND_CONNECTIONS:
            cv2.line(
                annotated_image,
                points[start_index],
                points[end_index],
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        for point in points:
            cv2.circle(
                annotated_image,
                point,
                4,
                (255, 64, 64),
                -1,
                cv2.LINE_AA,
            )

    return annotated_image


def main() -> None:
    cap = cv2.VideoCapture(0)
    hand_landmarker = None

    try:
        if not cap.isOpened():
            raise RuntimeError(
                "Не удалось открыть камеру. Проверьте подключение и разрешения."
            )

        hand_landmarker = HandLandmarker()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError("Не удалось получить кадр с камеры.")

            # Mirror before detection so landmarks match the displayed frame.
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detection_result = hand_landmarker.detect(rgb_frame)
            annotated_rgb = draw_landmarks_on_image(
                rgb_frame,
                detection_result,
            )
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("Hand landmarks", annotated_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        try:
            if hand_landmarker is not None:
                hand_landmarker.close()
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
Detect hand position from camera source 2 using OpenCV for capture/display.

This uses MediaPipe for hand landmark detection and OpenCV for:
- camera capture
- frame drawing
- live display

Press `q` to quit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from iphone_connection import connect_camera, read_frame


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "hand_landmarker.task"
DEFAULT_SOURCE = 2
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect hand position from the camera feed."
    )
    parser.add_argument("--source", type=int, default=DEFAULT_SOURCE)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--num-hands", type=int, default=1)
    return parser.parse_args()


def create_landmarker(num_hands: int) -> vision.HandLandmarker:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=num_hands,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hands(landmarker: vision.HandLandmarker, frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return landmarker.detect(mp_image)


def draw_hand_skeleton(frame, result) -> None:
    if not result.hand_landmarks:
        cv2.putText(
            frame,
            "No hand detected",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return

    for index, hand_landmarks in enumerate(result.hand_landmarks):
        frame_h, frame_w = frame.shape[:2]
        points = [
            (int(lm.x * frame_w), int(lm.y * frame_h))
            for lm in hand_landmarks
        ]

        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(
                frame,
                points[start_idx],
                points[end_idx],
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        for point in points:
            cv2.circle(frame, point, 4, (0, 255, 255), -1, cv2.LINE_AA)

        handedness = "Hand"
        if result.handedness and index < len(result.handedness) and result.handedness[index]:
            handedness = result.handedness[index][0].category_name

        wrist = hand_landmarks[0]
        text_x = int(wrist.x * frame_w)
        text_y = max(int(wrist.y * frame_h) - 15, 20)
        cv2.putText(
            frame,
            handedness,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        f"Hands detected: {len(result.hand_landmarks)}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    landmarker = create_landmarker(args.num_hands)
    capture = connect_camera(
        source=args.source,
        width=args.width,
        height=args.height,
    )

    try:
        while True:
            frame = read_frame(capture, mirror=False)
            result = detect_hands(landmarker, frame)

            display_frame = frame.copy()
            draw_hand_skeleton(display_frame, result)

            cv2.imshow("Hand Place Detection", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

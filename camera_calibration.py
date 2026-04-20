from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from iphone_connection import connect_camera, read_frame


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_FILE = PROJECT_ROOT / "camera_calibration.npz"
SOURCE = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CHARUCO_SQUARES_X = 5
CHARUCO_SQUARES_Y = 5
SQUARE_LENGTH_MM = 39.0
MARKER_LENGTH_MM = 29.0
MIN_SAMPLES = 12
MIN_CORNERS = 8
DICTIONARY_NAME = "DICT_4X4_1000"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate a camera from a ChArUco board.")
    parser.add_argument("--source", type=int, default=SOURCE, help="Camera index to open.")
    parser.add_argument("--width", type=int, default=FRAME_WIDTH, help="Capture width.")
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT, help="Capture height.")
    parser.add_argument("--squares-x", type=int, default=CHARUCO_SQUARES_X, help="Board squares across.")
    parser.add_argument("--squares-y", type=int, default=CHARUCO_SQUARES_Y, help="Board squares down.")
    parser.add_argument(
        "--square-mm",
        type=float,
        default=SQUARE_LENGTH_MM,
        help="Checker square side length in millimeters.",
    )
    parser.add_argument(
        "--marker-mm",
        type=float,
        default=MARKER_LENGTH_MM,
        help="ArUco marker side length in millimeters.",
    )
    parser.add_argument(
        "--dictionary",
        default=DICTIONARY_NAME,
        help="OpenCV ArUco dictionary name, for example DICT_4X4_50.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=MIN_SAMPLES,
        help="Minimum number of captured views before calibration.",
    )
    parser.add_argument(
        "--min-corners",
        type=int,
        default=MIN_CORNERS,
        help="Minimum detected ChArUco corners required to accept a frame.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="Output .npz file for camera calibration.",
    )
    return parser


def get_dictionary(dictionary_name: str):
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unsupported ArUco dictionary: {dictionary_name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))


def create_charuco_board(
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    marker_length_mm: float,
    dictionary_name: str,
):
    dictionary = get_dictionary(dictionary_name)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length_mm,
        marker_length_mm,
        dictionary,
    )
    detector = cv2.aruco.CharucoDetector(board)
    return dictionary, board, detector


def main() -> None:
    args = build_parser().parse_args()
    _dictionary, board, detector = create_charuco_board(
        args.squares_x,
        args.squares_y,
        args.square_mm,
        args.marker_mm,
        args.dictionary,
    )

    capture = connect_camera(source=args.source, width=args.width, height=args.height)
    all_charuco_corners: list[np.ndarray] = []
    all_charuco_ids: list[np.ndarray] = []
    frame_size: tuple[int, int] | None = None

    print("Calibration controls:")
    print("  c = capture current ChArUco view")
    print("  q = finish and calibrate")
    print(
        f"Need at least {args.min_samples} good views with at least {args.min_corners} detected ChArUco corners."
    )
    print(
        f"Board config: {args.squares_x}x{args.squares_y} squares, "
        f"square {args.square_mm} mm, marker {args.marker_mm} mm, {args.dictionary}."
    )

    try:
        while True:
            frame = read_frame(capture, mirror=False)
            frame_size = (frame.shape[1], frame.shape[0])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display = frame.copy()

            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
            if charuco_ids is not None and len(charuco_ids) > 0:
                cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)

            detected_corner_count = 0 if charuco_ids is None else len(charuco_ids)
            cv2.putText(
                display,
                f"Samples: {len(all_charuco_corners)} | Corners: {detected_corner_count} | c=capture | q=calibrate",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("ChArUco Calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                if charuco_ids is None or charuco_corners is None or len(charuco_ids) < args.min_corners:
                    print(
                        f"Not enough ChArUco corners in current frame. "
                        f"Detected {detected_corner_count}, need at least {args.min_corners}."
                    )
                    continue

                all_charuco_corners.append(charuco_corners.copy())
                all_charuco_ids.append(charuco_ids.copy())
                print(
                    f"Captured sample {len(all_charuco_corners)} "
                    f"with {detected_corner_count} ChArUco corners."
                )
            elif key == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if frame_size is None:
        raise RuntimeError("No frames were captured.")

    if len(all_charuco_corners) < args.min_samples:
        raise RuntimeError(
            f"Not enough samples for calibration: got {len(all_charuco_corners)}, "
            f"need at least {args.min_samples}."
        )

    calibration_result = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners,
        all_charuco_ids,
        board,
        frame_size,
        None,
        None,
    )
    rms_error, camera_matrix, dist_coeffs, _rvecs, _tvecs = calibration_result

    np.savez(
        args.output,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms_error=np.array([rms_error], dtype=np.float32),
        frame_width=np.array([frame_size[0]], dtype=np.int32),
        frame_height=np.array([frame_size[1]], dtype=np.int32),
        squares_x=np.array([args.squares_x], dtype=np.int32),
        squares_y=np.array([args.squares_y], dtype=np.int32),
        square_length_mm=np.array([args.square_mm], dtype=np.float32),
        marker_length_mm=np.array([args.marker_mm], dtype=np.float32),
        sample_count=np.array([len(all_charuco_corners)], dtype=np.int32),
    )

    print(f"Saved calibration to {args.output}")
    print(f"RMS reprojection error: {rms_error:.4f}")
    print("Camera matrix:")
    print(camera_matrix)
    print("Distortion coefficients:")
    print(dist_coeffs.ravel())


if __name__ == "__main__":
    main()

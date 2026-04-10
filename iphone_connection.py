"""
OpenCV helpers for connecting to an Android phone camera over USB-C.

On Windows, the practical setup is:
1. Connect the Android phone with a USB-C cable.
2. Put the phone into webcam / USB camera mode.
3. Open the exposed webcam device from OpenCV.

Examples:
    python iphone_connection.py
    python iphone_connection.py --list
    python iphone_connection.py --source 1
"""

from __future__ import annotations

import argparse
from typing import Union

import cv2


CameraSource = Union[int, str]
DEFAULT_ANDROID_CAMERA_INDEX = 2
CAMERA_BACKENDS = (
    ("CAP_ANY", cv2.CAP_ANY),
    ("CAP_MSMF", cv2.CAP_MSMF),
    ("CAP_DSHOW", cv2.CAP_DSHOW),
)


def normalize_camera_source(source: str) -> CameraSource:
    """Convert numeric strings like '0' into camera indexes."""
    source = source.strip()
    return int(source) if source.isdigit() else source


def expand_stream_source(source: str) -> str:
    """Normalize common Android camera stream inputs into an OpenCV-friendly URL."""
    if source.startswith(("http://", "https://", "rtsp://")):
        lowered = source.lower()
        if lowered.endswith(("/video", "/mjpegfeed", "/videofeed")):
            return source
        return source.rstrip("/") + "/video"

    return source


def list_available_cameras(max_index: int = 10) -> list[int]:
    """Probe camera indexes and return the ones that can deliver a frame."""
    available: list[int] = []

    for index in range(max_index):
        for _backend_name, backend in CAMERA_BACKENDS:
            capture = cv2.VideoCapture(index, backend)
            try:
                if not capture.isOpened():
                    continue

                success, _frame = capture.read()
                if success:
                    available.append(index)
                    break
            finally:
                capture.release()

    return available


def connect_camera(
    source: CameraSource = 0,
    width: int | None = None,
    height: int | None = None,
) -> cv2.VideoCapture:
    """Open a camera source and return an initialized VideoCapture."""
    if isinstance(source, int):
        attempts: list[str] = []

        for backend_name, backend in CAMERA_BACKENDS:
            capture = cv2.VideoCapture(source, backend)
            attempts.append(backend_name)

            if not capture.isOpened():
                capture.release()
                continue

            if width is not None:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height is not None:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            success, _frame = capture.read()
            if success:
                return capture

            capture.release()

        raise RuntimeError(
            f"Could not open camera index {source}. Tried backends: {', '.join(attempts)}"
        )
    else:
        stream_source = expand_stream_source(source)
        capture = cv2.VideoCapture(stream_source)

        if width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera source: {stream_source}")

        return capture


def connect_android_usb_camera(
    preferred_index: int | None = None,
    width: int | None = None,
    height: int | None = None,
    max_index: int = 10,
) -> cv2.VideoCapture:
    """
    Connect to an Android phone exposed over USB-C as a webcam device.

    If no index is provided, the first working camera index is used.
    """
    if preferred_index is not None:
        return connect_camera(preferred_index, width=width, height=height)

    available = list_available_cameras(max_index=max_index)
    if not available:
        raise RuntimeError(
            "No camera devices were found. Make sure the Android phone is connected "
            "via USB-C and set to webcam / USB camera mode."
        )

    return connect_camera(available[0], width=width, height=height)


def read_frame(capture: cv2.VideoCapture, mirror: bool = True):
    """Read a frame from an open camera capture."""
    success, frame = capture.read()
    if not success or frame is None:
        raise RuntimeError("Failed to read a frame from the camera.")

    if mirror:
        frame = cv2.flip(frame, 1)

    return frame


def show_camera_feed(
    source: CameraSource = 0,
    window_name: str = "Android USB Camera",
    mirror: bool = True,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """
    Connect to a camera and display live video until the user presses q.

    `source` can be:
    - an integer camera index, such as 0
    - a stream URL, if you want to override the default USB webcam path
    """
    capture = connect_camera(source=source, width=width, height=height)

    try:
        while True:
            frame = read_frame(capture, mirror=mirror)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def show_android_usb_camera(
    preferred_index: int | None = DEFAULT_ANDROID_CAMERA_INDEX,
    mirror: bool = True,
    width: int | None = None,
    height: int | None = None,
    max_index: int = 10,
) -> None:
    """Find the Android USB webcam device and display its live feed."""
    capture = connect_android_usb_camera(
        preferred_index=preferred_index,
        width=width,
        height=height,
        max_index=max_index,
    )

    try:
        while True:
            frame = read_frame(capture, mirror=mirror)
            cv2.imshow("Android USB Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Connect to a camera with OpenCV and display its live feed."
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional camera index or stream URL. If omitted, Android USB camera auto-detection is used.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional capture width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional capture height.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirroring.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List working camera indexes and exit.",
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=10,
        help="Highest camera index range to probe during auto-detection.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        cameras = list_available_cameras(max_index=args.max_index)
        if cameras:
            print("Available camera indexes:", ", ".join(str(index) for index in cameras))
        else:
            print(
                "No working camera indexes found. Check the USB-C cable and enable "
                "webcam / USB camera mode on the Android phone."
            )
        return

    if args.source is not None:
        source = normalize_camera_source(args.source)
        show_camera_feed(
            source=source,
            mirror=not args.no_mirror,
            width=args.width,
            height=args.height,
        )
        return

    show_android_usb_camera(
        preferred_index=DEFAULT_ANDROID_CAMERA_INDEX,
        mirror=not args.no_mirror,
        width=args.width,
        height=args.height,
        max_index=args.max_index,
    )


if __name__ == "__main__":
    main()

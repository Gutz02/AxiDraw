from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from UART_reading_esp32c3 import KalmanAngle, connection, read_values
from aruco_marker_detection import (
    CALIBRATION_FILE,
    DISTANCE_SCALE_CORRECTION,
    MARKERS_DIR,
    SOURCE,
    build_camera_matrix,
    estimate_marker_pose,
    family_to_opencv_constant,
    load_camera_calibration,
    load_expected_markers,
    plane_distance_mm,
)
from iphone_connection import connect_camera, read_frame


@dataclass
class ImuState:
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    accel_roll_deg: float = 0.0
    accel_pitch_deg: float = 0.0
    last_update_s: float = 0.0
    sample_count: int = 0
    ready: bool = False

PEN_LENGTH_MM = 145.0
TIP_ROLL_SIGN = 1.0
TIP_PITCH_SIGN = -1.0


def imu_reader_worker(
    stop_event: threading.Event,
    state: ImuState,
    state_lock: threading.Lock,
) -> None:
    ser = connection()
    kalman_roll = KalmanAngle()
    kalman_pitch = KalmanAngle()
    kalman_roll_angle = 0.0
    kalman_pitch_angle = 0.0
    filters_initialized = False
    prev_time = None
    sample_count = 0

    try:
        while not stop_event.is_set():
            line = read_values(ser)
            if not line:
                continue

            values = line.split()
            if len(values) < 5:
                continue

            try:
                ax = float(values[0])
                ay = float(values[1])
                az = float(values[2])
                gx = float(values[3])
                gy = float(values[4])
            except ValueError:
                continue

            now = time.perf_counter()
            dt = 0.0 if prev_time is None else now - prev_time
            prev_time = now

            accel_roll = math.atan2(ay, az)
            accel_pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))

            if not filters_initialized:
                kalman_roll.set_angle(accel_roll)
                kalman_pitch.set_angle(accel_pitch)
                kalman_roll_angle = accel_roll
                kalman_pitch_angle = accel_pitch
                filters_initialized = True
            elif dt > 0.0:
                kalman_roll_angle = kalman_roll.update(accel_roll, gx, dt)
                kalman_pitch_angle = kalman_pitch.update(accel_pitch, gy, dt)

            sample_count += 1
            with state_lock:
                state.ax = ax
                state.ay = ay
                state.az = az
                state.roll_deg = math.degrees(kalman_roll_angle)
                state.pitch_deg = math.degrees(kalman_pitch_angle)
                state.accel_roll_deg = math.degrees(accel_roll)
                state.accel_pitch_deg = math.degrees(accel_pitch)
                state.last_update_s = now
                state.sample_count = sample_count
                state.ready = True
    finally:
        ser.close()


def build_detector_parameters() -> cv2.aruco.DetectorParameters:
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 50
    parameters.cornerRefinementMinAccuracy = 0.01
    return parameters


def refine_marker_corners(gray: np.ndarray, marker_corners: np.ndarray) -> np.ndarray:
    refined = marker_corners.reshape((4, 1, 2)).astype(np.float32).copy()
    cv2.cornerSubPix(
        gray,
        refined,
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=(
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            50,
            0.001,
        ),
    )
    return refined.reshape((1, 4, 2))


def pen_tip(mx: float, my: float, roll_deg: float, pitch_deg: float, l: float) -> tuple[float, float]:
    """
    Project pen tip position in image coordinates.
    
    Assumes pen always points downward (+Y) in the image.
    - roll (deg): Rotation around X-axis → horizontal displacement in image
    - pitch (deg): Rotation around Y-axis → vertical displacement in image
    - l (px): Pen length in image pixels
    
    For top-down camera looking at tilted pen:
      - Roll > 0 tilts pen right → tip moves right (+X)
      - Pitch > 0 tilts pen forward → tip moves down (+Y)
    """
    roll_rad = np.radians(roll_deg*TIP_ROLL_SIGN)
    pitch_rad = np.radians(pitch_deg*TIP_PITCH_SIGN)

    # Small-angle / 2D projection for pen lying on horizontal surface
    # Tip displacement from center is L * sin(angle) in direction perpendicular to pen
    delta_x = l * np.sin(roll_rad)
    delta_y = l * np.sin(pitch_rad)

    tx = mx + delta_x
    ty = my + delta_y
    return float(tx), float(ty)


def marker_center_and_scale(marker_corners: np.ndarray, marker_size_mm: float) -> tuple[float, float, float]:
    pts = marker_corners.reshape((4, 2)).astype(np.float32)
    mx = float(np.mean(pts[:, 0]))
    my = float(np.mean(pts[:, 1]))
    side_lengths_px = [
        float(np.linalg.norm(pts[(i + 1) % 4] - pts[i]))
        for i in range(4)
    ]
    avg_side_px = sum(side_lengths_px) / len(side_lengths_px)
    pixels_per_mm = 0.0 if marker_size_mm <= 0.0 else avg_side_px / marker_size_mm
    return mx, my, pixels_per_mm

def main() -> None:
    expected_markers, marker_sizes_mm, ambiguous_markers = load_expected_markers(MARKERS_DIR)
    detectors = {
        family: cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(family_to_opencv_constant(family)),
            build_detector_parameters(),
        )
        for family in expected_markers
    }

    calibration = load_camera_calibration()
    if calibration is None:
        camera_matrix = None
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        distance_scale = DISTANCE_SCALE_CORRECTION
        print(
            f"Calibration file not found at {CALIBRATION_FILE}. "
            f"Using approximate intrinsics with scale correction {DISTANCE_SCALE_CORRECTION:.3f}."
        )
    else:
        camera_matrix, dist_coeffs = calibration
        distance_scale = 1.0
        print(f"Loaded camera calibration from {CALIBRATION_FILE}.")

    capture = connect_camera(source=SOURCE, width=960, height=540)
    imu_state = ImuState()
    imu_lock = threading.Lock()
    stop_event = threading.Event()
    imu_roll_offset_deg = 0.0
    imu_pitch_offset_deg = 0.0
    imu_az_zero = 0.0
    imu_thread = threading.Thread(
        target=imu_reader_worker,
        args=(stop_event, imu_state, imu_lock),
        name="imu-reader",
        daemon=True,
    )
    imu_thread.start()
    drawing_canvas: np.ndarray | None = None

    try:
        while True:
            frame = read_frame(capture, mirror=False)
            if camera_matrix is None:
                camera_matrix = build_camera_matrix(frame.shape[1], frame.shape[0])
            if drawing_canvas is None:
                drawing_canvas = np.zeros_like(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            total = 0
            detected_markers: list[dict[str, object]] = []

            for family, detector in detectors.items():
                corners, ids, _rejected = detector.detectMarkers(gray)
                if ids is None:
                    continue

                for marker_corners, marker_id_array in zip(corners, ids):
                    marker_id = int(marker_id_array[0])
                    if marker_id not in expected_markers[family]:
                        continue

                    total += 1
                    refined_corners = refine_marker_corners(gray, marker_corners)
                    marker_key = (family, marker_id)
                    marker_size_mm = marker_sizes_mm.get(marker_key)
                    pose = None
                    if marker_size_mm is not None:
                        pose = estimate_marker_pose(refined_corners, marker_size_mm, camera_matrix, dist_coeffs)

                    rvec, tvec = pose if pose is not None else (None, None)
                    detected_markers.append(
                        {
                            "family": family,
                            "marker_id": marker_id,
                            "corners": refined_corners,
                            "rvec": rvec,
                            "tvec": tvec,
                            "marker_size_mm": marker_size_mm,
                        }
                    )

                    pts = marker_corners.reshape((4, 2)).astype(int)
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                    for i in range(4):
                        cv2.circle(frame, tuple(pts[i]), 2, (255, 0, 0), -1)

                    marker_label = f"{family}:{marker_id}"
                    if tvec is not None:
                        cv2.drawFrameAxes(
                            frame,
                            camera_matrix,
                            dist_coeffs,
                            rvec,
                            tvec,
                            marker_size_mm * 0.5,
                            2,
                        )
                    elif marker_key in ambiguous_markers:
                        marker_label += " size?"

                    cv2.putText(
                        frame,
                        marker_label,
                        (pts[0][0], max(pts[0][1] - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            with imu_lock:
                imu_snapshot = ImuState(
                    ax=imu_state.ax,
                    ay=imu_state.ay,
                    az=imu_state.az,
                    roll_deg=imu_state.roll_deg,
                    pitch_deg=imu_state.pitch_deg,
                    accel_roll_deg=imu_state.accel_roll_deg,
                    accel_pitch_deg=imu_state.accel_pitch_deg,
                    last_update_s=imu_state.last_update_s,
                    sample_count=imu_state.sample_count,
                    ready=imu_state.ready,
                )

            cv2.putText(
                frame,
                (
                    f"Detected: {total} | "
                    "Press 'c' to zero IMU | Press 'space' to clear drawing | Press 'd' for snapshot | Press 'q' to quit"
                ),
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if imu_snapshot.ready:
                imu_age_s = time.perf_counter() - imu_snapshot.last_update_s
                imu_roll_zeroed_deg = imu_snapshot.roll_deg - imu_roll_offset_deg
                imu_pitch_zeroed_deg = imu_snapshot.pitch_deg - imu_pitch_offset_deg
                pen_height_cm = (
                    (PEN_LENGTH_MM / 10.0)
                    * math.cos(math.radians(imu_roll_zeroed_deg))
                    * math.cos(math.radians(imu_pitch_zeroed_deg))
                )
                for entry in detected_markers:
                    marker_size_mm = entry["marker_size_mm"]
                    if marker_size_mm is None:
                        continue

                    mx, my, pixels_per_mm = marker_center_and_scale(entry["corners"], float(marker_size_mm))
                    if pixels_per_mm <= 0.0:
                        continue

                    pen_length_px = PEN_LENGTH_MM * pixels_per_mm
                    tip_x, tip_y = pen_tip(
                        mx,
                        my,
                        imu_roll_zeroed_deg,
                        imu_pitch_zeroed_deg,
                        pen_length_px,
                    )

                    cv2.circle(frame, (int(round(tip_x)), int(round(tip_y))), 6, (0, 0, 255), -1)
                    cv2.line(
                        frame,
                        (int(round(mx)), int(round(my))),
                        (int(round(tip_x)), int(round(tip_y))),
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        "tip",
                        (int(round(tip_x)) + 8, int(round(tip_y)) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    if drawing_canvas is not None:
                        cv2.circle(
                            drawing_canvas,
                            (int(round(tip_x)), int(round(tip_y))),
                            2,
                            (255, 255, 255),
                            -1,
                        )
                cv2.putText(
                    frame,
                    (
                        f"IMU roll/pitch: {imu_roll_zeroed_deg:.1f} / {imu_pitch_zeroed_deg:.1f} deg "
                        f"| az: {imu_snapshot.az:.3f} | age: {imu_age_s * 1000.0:.0f} ms "
                        f"| samples: {imu_snapshot.sample_count}"
                    ),
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Pen height h=L*cos(r)*cos(p): {pen_height_cm:.2f} cm",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame,
                    "IMU: waiting for UART samples...",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("ArUco + UART Integration", frame)
            if drawing_canvas is not None:
                cv2.imshow("Predicted Tip Drawing", drawing_canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                if imu_snapshot.ready:
                    imu_roll_offset_deg = imu_snapshot.roll_deg
                    imu_pitch_offset_deg = imu_snapshot.pitch_deg
                    imu_az_zero = imu_snapshot.az
                    print(
                        f"IMU zeroed at roll {imu_roll_offset_deg:.2f} deg, "
                        f"pitch {imu_pitch_offset_deg:.2f} deg, az {imu_az_zero:.3f}"
                    )
                else:
                    print("IMU zero requested, but no UART samples are available yet.")
            elif key == ord(" "):
                if drawing_canvas is not None:
                    drawing_canvas.fill(0)
                    print("Predicted tip drawing cleared.")
            elif key == ord("d"):
                print("Snapshot:")
                if imu_snapshot.ready:
                    pen_height_cm = (
                        (PEN_LENGTH_MM / 10.0)
                        * math.cos(math.radians(imu_snapshot.roll_deg - imu_roll_offset_deg))
                        * math.cos(math.radians(imu_snapshot.pitch_deg - imu_pitch_offset_deg))
                    )
                    print(
                        f"  IMU -> roll {imu_snapshot.roll_deg - imu_roll_offset_deg:.2f} deg, "
                        f"pitch {imu_snapshot.pitch_deg - imu_pitch_offset_deg:.2f} deg "
                        f"(raw {imu_snapshot.roll_deg:.2f}, {imu_snapshot.pitch_deg:.2f}), "
                        f"accel xyz ({imu_snapshot.ax:.3f}, {imu_snapshot.ay:.3f}, {imu_snapshot.az:.3f})"
                    )
                    print(f"  Pen height from IMU -> {pen_height_cm:.2f} cm")
                else:
                    print("  IMU -> no samples yet")

                if not detected_markers:
                    print("  Vision -> no expected ArUco marker detected")
                    continue

                for entry in detected_markers:
                    family = entry["family"]
                    marker_id = entry["marker_id"]
                    marker_key = (family, marker_id)
                    marker_size_mm = entry["marker_size_mm"]
                    rvec = entry["rvec"]
                    tvec = entry["tvec"]
                    if marker_size_mm is None:
                        print(
                            f"  {family}:{marker_id} -> pose unavailable; set size in MARKER_SIZE_MM_BY_ID "
                            f"or DEFAULT_MARKER_SIZE_MM"
                        )
                        continue
                    if rvec is None or tvec is None:
                        print(f"  {family}:{marker_id} -> pose solve failed")
                        continue

                    raw_distance_mm = plane_distance_mm(rvec, tvec)
                    distance_mm = raw_distance_mm * distance_scale
                    z_mm = float(tvec[2][0]) * distance_scale
                    print(
                        f"  {family}:{marker_id} ({marker_size_mm:.1f} mm) -> "
                        f"plane distance {distance_mm:.1f} mm, raw {raw_distance_mm:.1f} mm, "
                        f"z {z_mm:.1f} mm"
                    )
                    mx, my, pixels_per_mm = marker_center_and_scale(entry["corners"], float(marker_size_mm))
                    if pixels_per_mm > 0.0:
                        tip_x, tip_y = pen_tip(
                            mx,
                            my,
                            imu_snapshot.roll_deg - imu_roll_offset_deg,
                            imu_snapshot.pitch_deg - imu_pitch_offset_deg,
                            PEN_LENGTH_MM * pixels_per_mm,
                        )
                        print(f"    tip position -> ({tip_x:.1f}, {tip_y:.1f}) px")
            elif key == ord("q"):
                break
    finally:
        stop_event.set()
        imu_thread.join(timeout=1.0)
        capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
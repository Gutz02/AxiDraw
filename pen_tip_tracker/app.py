from __future__ import annotations

import math
import threading

import cv2
import numpy as np

from aruco_marker_detection import (
    CALIBRATION_FILE,
    DISTANCE_SCALE_CORRECTION,
    REFERENCE_MARKER,
    SOURCE,
    build_camera_matrix,
    load_camera_calibration,
)
from iphone_connection import connect_camera, read_frame

from .controls import handle_runtime_key
from .geometry import marker_center_and_scale, project_tip, undistort_marker_corners, wrap_angle_rad
from .imu import snapshot_imu_state, snapshot_kalman_tuning, start_imu_thread
from .models import ImuState, KalmanTuning, LastPenMarkerState, TrackerRuntimeState
from .overlay import draw_header, draw_runtime_overlay, draw_waiting_overlay
from .settings import WINDOW_TITLE
from .vision import compute_pose_alignment, detect_markers, load_detector_context, print_vision_snapshot


def load_camera_setup() -> tuple[np.ndarray | None, np.ndarray, float, tuple[int, int] | None]:
    calibration = load_camera_calibration()
    if calibration is None:
        print(
            f"Calibration file not found at {CALIBRATION_FILE}. "
            f"Using approximate intrinsics with scale correction {DISTANCE_SCALE_CORRECTION:.3f}."
        )
        return None, np.zeros((5, 1), dtype=np.float32), DISTANCE_SCALE_CORRECTION, None

    camera_matrix, dist_coeffs = calibration
    calibration_frame_size: tuple[int, int] | None = None
    calibration_file = np.load(CALIBRATION_FILE)
    frame_width = calibration_file.get("frame_width")
    frame_height = calibration_file.get("frame_height")
    if frame_width is not None and frame_height is not None:
        calibration_frame_size = (int(frame_width[0]), int(frame_height[0]))

    print(f"Loaded camera calibration from {CALIBRATION_FILE}.")
    return camera_matrix, dist_coeffs, 1.0, calibration_frame_size


def scale_camera_matrix(
    camera_matrix: np.ndarray,
    calibration_frame_size: tuple[int, int],
    frame_size: tuple[int, int],
) -> np.ndarray:
    calibration_width, calibration_height = calibration_frame_size
    frame_width, frame_height = frame_size
    scale_x = frame_width / calibration_width
    scale_y = frame_height / calibration_height
    scaled = camera_matrix.astype(np.float32).copy()
    scaled[0, 0] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[0, 2] *= scale_x
    scaled[1, 2] *= scale_y
    return scaled


def draw_angle_indicator(
    frame: np.ndarray,
    center: tuple[int, int],
    radius_px: float,
    yaw_aligned_rad: float,
) -> None:
    radius = max(12, int(round(radius_px)))
    center_x, center_y = center
    zero_angle_rad = -math.pi / 2.0
    current_angle_rad = zero_angle_rad + yaw_aligned_rad
    yaw_aligned_deg = math.degrees(yaw_aligned_rad)

    sector_points = [center]
    step_count = max(2, int(abs(yaw_aligned_deg) / 6.0) + 1)
    for step in range(step_count + 1):
        t = step / step_count
        angle_rad = zero_angle_rad + yaw_aligned_rad * t
        px = int(round(center_x + radius * math.cos(angle_rad)))
        py = int(round(center_y + radius * math.sin(angle_rad)))
        sector_points.append((px, py))
    cv2.fillPoly(frame, [np.array(sector_points, dtype=np.int32)], (180, 120, 255))

    zero_x = int(round(center_x + radius * math.cos(zero_angle_rad)))
    zero_y = int(round(center_y + radius * math.sin(zero_angle_rad)))
    current_x = int(round(center_x + radius * math.cos(current_angle_rad)))
    current_y = int(round(center_y + radius * math.sin(current_angle_rad)))

    cv2.circle(frame, center, radius, (255, 0, 255), 1)
    cv2.line(frame, center, (zero_x, zero_y), (150, 150, 150), 1)
    cv2.line(frame, center, (current_x, current_y), (255, 255, 255), 2)
    cv2.putText(
        frame,
        f"{yaw_aligned_deg:+.1f} deg",
        (center_x + 8, center_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 255),
        1,
        cv2.LINE_AA,
    )


def update_pen_tip_overlay(
    frame: np.ndarray,
    detected_markers,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    runtime_state: TrackerRuntimeState,
    frame_center_x: float,
    frame_center_y: float,
    imu_snapshot: ImuState,
    imu_roll_avg_deg: float,
    imu_pitch_avg_deg: float,
    pen_marker_relative_yaw: float | None,
) -> tuple[float | None, tuple[float, float, float, float] | None]:
    tip_debug_values: tuple[float, float, float, float] | None = None
    pen_height_cm: float | None = None
    marker_seen = False
    phi = math.radians(imu_roll_avg_deg)
    theta = math.radians(imu_pitch_avg_deg)

    for entry in detected_markers:
        if (entry.family, entry.marker_id) != REFERENCE_MARKER or entry.marker_size_mm is None:
            continue

        corrected_pts = undistort_marker_corners(entry.corners, camera_matrix, dist_coeffs)
        mx, my, pixels_per_mm = marker_center_and_scale(corrected_pts, float(entry.marker_size_mm))
        if pixels_per_mm <= 0.0:
            continue

        current_marker_yaw_aligned: float | None = None
        if pen_marker_relative_yaw is not None:
            if not runtime_state.marker_offsets_initialized:
                runtime_state.marker_yaw_offset = pen_marker_relative_yaw
                runtime_state.imu_roll_offset_deg = imu_snapshot.roll_deg
                runtime_state.imu_pitch_offset_deg = imu_snapshot.pitch_deg
                runtime_state.marker_offsets_initialized = True
            current_marker_yaw_aligned = wrap_angle_rad(
                pen_marker_relative_yaw - runtime_state.marker_yaw_offset
            )
        elif runtime_state.last_pen_marker_state is not None:
            current_marker_yaw_aligned = runtime_state.last_pen_marker_state.marker_yaw_aligned

        if current_marker_yaw_aligned is None:
            continue

        projection = project_tip(
            mx=mx,
            my=my,
            pixels_per_mm=pixels_per_mm,
            frame_center_x=frame_center_x,
            frame_center_y=frame_center_y,
            pen_length_mm=runtime_state.pen_length_mm,
            sensor_offset_x_mm=runtime_state.sensor_offset_x_mm,
            sensor_offset_y_mm=runtime_state.sensor_offset_y_mm,
            phi=phi,
            theta=theta,
            yaw_aligned=current_marker_yaw_aligned,
        )
        runtime_state.last_pen_marker_state = LastPenMarkerState(
            mx=mx,
            my=my,
            pixels_per_mm=pixels_per_mm,
            marker_yaw_aligned=current_marker_yaw_aligned,
        )
        marker_seen = True
        pen_height_cm = projection.pen_height_cm
        tip_debug_values = (
            projection.mx_corrected,
            projection.my_corrected,
            projection.tip_dx,
            projection.tip_dy,
        )

        cv2.circle(
            frame,
            (int(round(mx)), int(round(my))),
            int(round(projection.radius_px)),
            color=(255, 0, 255),
            thickness=1,
        )
        draw_angle_indicator(
            frame,
            (int(round(mx)), int(round(my))),
            projection.radius_px,
            current_marker_yaw_aligned,
        )
        cv2.circle(
            frame,
            (int(round(projection.tip_x)), int(round(projection.tip_y))),
            2,
            (0, 0, 255),
            -1,
        )
        cv2.line(
            frame,
            (int(round(mx)), int(round(my))),
            (int(round(projection.tip_x)), int(round(projection.tip_y))),
            (0, 0, 255),
            2,
        )
        break

    if not marker_seen and runtime_state.last_pen_marker_state is not None:
        cached = runtime_state.last_pen_marker_state
        projection = project_tip(
            mx=cached.mx,
            my=cached.my,
            pixels_per_mm=cached.pixels_per_mm,
            frame_center_x=frame_center_x,
            frame_center_y=frame_center_y,
            pen_length_mm=runtime_state.pen_length_mm,
            sensor_offset_x_mm=runtime_state.sensor_offset_x_mm,
            sensor_offset_y_mm=runtime_state.sensor_offset_y_mm,
            phi=phi,
            theta=theta,
            yaw_aligned=cached.marker_yaw_aligned,
        )
        pen_height_cm = projection.pen_height_cm
        tip_debug_values = (
            projection.mx_corrected,
            projection.my_corrected,
            projection.tip_dx,
            projection.tip_dy,
        )

        cv2.circle(
            frame,
            (int(round(cached.mx)), int(round(cached.my))),
            int(round(projection.radius_px)),
            color=(120, 0, 255),
            thickness=1,
        )
        draw_angle_indicator(
            frame,
            (int(round(cached.mx)), int(round(cached.my))),
            projection.radius_px,
            cached.marker_yaw_aligned,
        )
        cv2.circle(
            frame,
            (int(round(projection.tip_x)), int(round(projection.tip_y))),
            2,
            (0, 0, 255),
            -1,
        )
        cv2.line(
            frame,
            (int(round(cached.mx)), int(round(cached.my))),
            (int(round(projection.tip_x)), int(round(projection.tip_y))),
            (0, 0, 255),
            2,
        )

    return pen_height_cm, tip_debug_values


def print_snapshot(
    imu_snapshot: ImuState,
    runtime_state: TrackerRuntimeState,
    kalman_snapshot: KalmanTuning,
    imu_roll_avg_deg: float,
    imu_pitch_avg_deg: float,
    detected_markers,
    distance_scale: float,
    relative_xy_mm: tuple[float, float] | None,
) -> None:
    print("Snapshot:")
    if imu_snapshot.ready:
        pen_height_cm = (
            (runtime_state.pen_length_mm / 10.0)
            * math.cos(math.radians(imu_roll_avg_deg))
            * math.cos(math.radians(imu_pitch_avg_deg))
        )
        print(
            f"  IMU -> roll avg {imu_roll_avg_deg:.2f} deg, "
            f"pitch avg {imu_pitch_avg_deg:.2f} deg "
            f"(zeroed raw {imu_snapshot.roll_deg - runtime_state.imu_roll_offset_deg:.2f}, "
            f"{imu_snapshot.pitch_deg - runtime_state.imu_pitch_offset_deg:.2f}), "
            f"(raw {imu_snapshot.roll_deg:.2f}, {imu_snapshot.pitch_deg:.2f}), "
            f"accel xyz ({imu_snapshot.ax:.3f}, {imu_snapshot.ay:.3f}, {imu_snapshot.az:.3f})"
        )
        print(f"  Pen height from IMU -> {pen_height_cm:.2f} cm")
        print(
            f"  Sensor offset -> x {runtime_state.sensor_offset_x_mm:.2f} mm, "
            f"y {runtime_state.sensor_offset_y_mm:.2f} mm"
        )
        print(
            f"  Kalman roll -> q_angle {kalman_snapshot.roll.q_angle:.4f}, "
            f"q_bias {kalman_snapshot.roll.q_bias:.4f}, "
            f"r_measure {kalman_snapshot.roll.r_measure:.3f}"
        )
        print(
            f"  Kalman pitch -> q_angle {kalman_snapshot.pitch.q_angle:.4f}, "
            f"q_bias {kalman_snapshot.pitch.q_bias:.4f}, "
            f"r_measure {kalman_snapshot.pitch.r_measure:.3f}"
        )
    else:
        print("  IMU -> no samples yet")

    print_vision_snapshot(detected_markers, distance_scale, relative_xy_mm)


def main() -> None:
    detector_context = load_detector_context()
    camera_matrix, dist_coeffs, distance_scale, calibration_frame_size = load_camera_setup()
    print(dist_coeffs)
    capture = connect_camera(source=SOURCE, width=1280, height=720)
    camera_matrix_scaled = False

    imu_state = ImuState()
    imu_lock = threading.Lock()
    kalman_tuning = KalmanTuning()
    kalman_tuning_lock = threading.Lock()
    runtime_state = TrackerRuntimeState()
    stop_event = threading.Event()
    imu_thread = start_imu_thread(
        stop_event,
        imu_state,
        imu_lock,
        kalman_tuning,
        kalman_tuning_lock,
    )

    try:
        while True:
            frame = read_frame(capture, mirror=False)
            frame_size = (frame.shape[1], frame.shape[0])
            frame_center_x = frame.shape[1] / 2.0
            frame_center_y = frame.shape[0] / 2.0
            if camera_matrix is None:
                print("Using approximate camera intrinsics based on frame size.")
                camera_matrix = build_camera_matrix(frame.shape[1], frame.shape[0])
            elif calibration_frame_size is not None and not camera_matrix_scaled:
                print(
                    "Camera frame size differs from calibration size. "
                    f"Calibration size: {calibration_frame_size[0]}x{calibration_frame_size[1]}, "
                    f"frame size: {frame_size[0]}x{frame_size[1]}."
                )
                if calibration_frame_size != frame_size:
                    camera_matrix = scale_camera_matrix(
                        camera_matrix,
                        calibration_frame_size,
                        frame_size,
                    )
                    print(
                        "Scaled camera intrinsics from "
                        f"{calibration_frame_size[0]}x{calibration_frame_size[1]} "
                        f"to {frame_size[0]}x{frame_size[1]}."
                    )
                camera_matrix_scaled = True

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_markers, total = detect_markers(
                frame,
                gray,
                detector_context,
                camera_matrix,
                dist_coeffs,
            )
            imu_snapshot = snapshot_imu_state(imu_state, imu_lock)
            kalman_snapshot = snapshot_kalman_tuning(kalman_tuning, kalman_tuning_lock)
            relative_xy_mm, pen_marker_relative_yaw = compute_pose_alignment(
                detected_markers,
                distance_scale,
            )

            draw_header(frame, total)

            imu_roll_avg_deg = 0.0
            imu_pitch_avg_deg = 0.0
            pen_height_cm: float | None = None
            tip_debug_values: tuple[float, float, float, float] | None = None

            if imu_snapshot.ready:
                imu_roll_zeroed_deg = imu_snapshot.roll_deg - runtime_state.imu_roll_offset_deg
                imu_pitch_zeroed_deg = imu_snapshot.pitch_deg - runtime_state.imu_pitch_offset_deg
                runtime_state.roll_history_deg.append(imu_roll_zeroed_deg)
                runtime_state.pitch_history_deg.append(imu_pitch_zeroed_deg)
                imu_roll_avg_deg = sum(runtime_state.roll_history_deg) / len(runtime_state.roll_history_deg)
                imu_pitch_avg_deg = sum(runtime_state.pitch_history_deg) / len(runtime_state.pitch_history_deg)

                pen_height_cm, tip_debug_values = update_pen_tip_overlay(
                    frame=frame,
                    detected_markers=detected_markers,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    runtime_state=runtime_state,
                    frame_center_x=frame_center_x,
                    frame_center_y=frame_center_y,
                    imu_snapshot=imu_snapshot,
                    imu_roll_avg_deg=imu_roll_avg_deg,
                    imu_pitch_avg_deg=imu_pitch_avg_deg,
                    pen_marker_relative_yaw=pen_marker_relative_yaw,
                )
                draw_runtime_overlay(
                    frame,
                    runtime_state,
                    imu_roll_avg_deg,
                    imu_pitch_avg_deg,
                    kalman_snapshot,
                    relative_xy_mm,
                    pen_marker_relative_yaw,
                    pen_height_cm,
                    tip_debug_values,
                )
            else:
                draw_waiting_overlay(frame)

            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("d"):
                print_snapshot(
                    imu_snapshot,
                    runtime_state,
                    kalman_snapshot,
                    imu_roll_avg_deg,
                    imu_pitch_avg_deg,
                    detected_markers,
                    distance_scale,
                    relative_xy_mm,
                )
                continue

            if handle_runtime_key(
                key,
                imu_snapshot,
                runtime_state,
                kalman_tuning,
                kalman_tuning_lock,
                pen_marker_relative_yaw,
            ):
                break
    finally:
        stop_event.set()
        imu_thread.join(timeout=1.0)
        capture.release()
        cv2.destroyAllWindows()

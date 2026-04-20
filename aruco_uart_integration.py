from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from UART_reading_esp32c3 import KalmanAngle, connection, read_values
from aruco_marker_detection import (
    CALIBRATION_FILE,
    DISTANCE_SCALE_CORRECTION,
    MARKERS_DIR,
    REFERENCE_MARKER,
    SOURCE,
    TARGET_MARKER,
    build_camera_matrix,
    estimate_marker_pose,
    family_to_opencv_constant,
    load_camera_calibration,
    load_expected_markers,
    plane_distance_mm,
    relative_transform,
    transform_to_pose,
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


@dataclass
class KalmanAxisTuning:
    q_angle: float = 0.0035
    q_bias: float = 6.0035
    r_measure: float = 11.010


@dataclass
class KalmanTuning:
    roll: KalmanAxisTuning = field(default_factory=KalmanAxisTuning)
    pitch: KalmanAxisTuning = field(
        default_factory=lambda: KalmanAxisTuning(
            q_angle=0.0035,
            q_bias=5.0020,
            r_measure=10.910,
        )
    )
    active_axis: str = "roll"


MOVING_AVERAGE_SAMPLES = 1
PEN_LENGTH_MM = 180
PEN_LENGTH_STEP_MM = 0.5
CAMERA_HEIGHT_CM = 75.4
TIP_ROLL_SIGN = 1.0
TIP_PITCH_SIGN = -1.0
XY_TARGET_MARKER = ("4x4_1000", 2)
Q_ANGLE_STEP = 0.1
Q_BIAS_STEP = 0.5
R_MEASURE_STEP = 0.1
SENSOR_OFFSET_STEP_MM = 0.01


def imu_reader_worker(
    stop_event: threading.Event,
    state: ImuState,
    state_lock: threading.Lock,
    tuning: KalmanTuning,
    tuning_lock: threading.Lock,
) -> None:
    ser = connection()
    with tuning_lock:
        initial_roll = KalmanAxisTuning(
            q_angle=tuning.roll.q_angle,
            q_bias=tuning.roll.q_bias,
            r_measure=tuning.roll.r_measure,
        )
        initial_pitch = KalmanAxisTuning(
            q_angle=tuning.pitch.q_angle,
            q_bias=tuning.pitch.q_bias,
            r_measure=tuning.pitch.r_measure,
        )
    kalman_roll = KalmanAngle(
        q_angle=initial_roll.q_angle,
        q_bias=initial_roll.q_bias,
        r_measure=initial_roll.r_measure,
    )
    kalman_pitch = KalmanAngle(
        q_angle=initial_pitch.q_angle,
        q_bias=initial_pitch.q_bias,
        r_measure=initial_pitch.r_measure,
    )
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

            with tuning_lock:
                kalman_roll.q_angle = tuning.roll.q_angle
                kalman_roll.q_bias = tuning.roll.q_bias
                kalman_roll.r_measure = tuning.roll.r_measure
                kalman_pitch.q_angle = tuning.pitch.q_angle
                kalman_pitch.q_bias = tuning.pitch.q_bias
                kalman_pitch.r_measure = tuning.pitch.r_measure

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


def pen_tip(
    mx: float,
    my: float,
    phi: float,
    theta: float,
    yaw: float,
    r: float,
    sensor_offset_x: float,
    sensor_offset_y: float,
) -> tuple[float, float, float, float]:
    a = math.atan2(-math.sin(phi) * math.cos(theta), math.sin(theta))
    dx = r * math.sin(a) + sensor_offset_x
    dy = r * math.cos(a) + sensor_offset_y
    
    dx_global = dx *math.cos(yaw) - dy * math.sin(yaw)
    dy_global = dx * math.sin(yaw) + dy * math.cos(yaw)
    tx = mx - dx_global
    ty = my - dy_global
    return float(tx), float(ty), float(dx_global), float(dy_global)


def wrap_angle_rad(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def planar_yaw_from_transform(transform: np.ndarray) -> float:
    rotation_matrix = transform[:3, :3]
    marker_x_axis_in_reference = rotation_matrix[:, 0]
    return math.atan2(
        float(marker_x_axis_in_reference[1]),
        float(marker_x_axis_in_reference[0]),
    )


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


def undistort_marker_corners(
    marker_corners: np.ndarray,
    camera_matrix: np.ndarray | None,
    dist_coeffs: np.ndarray | None,
) -> np.ndarray:
    if camera_matrix is None or dist_coeffs is None:
        return marker_corners.reshape((4, 2)).astype(np.float32)

    pts = marker_corners.reshape((4, 1, 2)).astype(np.float32)
    undistorted = cv2.undistortPoints(
        pts,
        camera_matrix,
        dist_coeffs,
        P=camera_matrix,
    )
    return undistorted.reshape((4, 2)).astype(np.float32)

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
    kalman_tuning = KalmanTuning()
    kalman_tuning_lock = threading.Lock()
    stop_event = threading.Event()
    imu_roll_offset_deg = 0.0
    imu_pitch_offset_deg = 0.0
    imu_az_zero = 0.0
    marker_yaw_offset = 0.0
    marker_offsets_initialized = False
    pen_length_mm = PEN_LENGTH_MM
    sensor_offset_x_mm = 0.0
    sensor_offset_y_mm = 0.0
    last_pen_marker_state: dict[str, float] | None = None
    roll_history_deg: deque[float] = deque(maxlen=MOVING_AVERAGE_SAMPLES)
    pitch_history_deg: deque[float] = deque(maxlen=MOVING_AVERAGE_SAMPLES)
    imu_thread = threading.Thread(
        target=imu_reader_worker,
        args=(stop_event, imu_state, imu_lock, kalman_tuning, kalman_tuning_lock),
        name="imu-reader",
        daemon=True,
    )
    imu_thread.start()
    # drawing_canvas: np.ndarray | None = None

    try:
        while True:
            frame = read_frame(capture, mirror=False)
            frame_center_x = frame.shape[1] / 2.0
            frame_center_y = frame.shape[0] / 2.0
            if camera_matrix is None:
                camera_matrix = build_camera_matrix(frame.shape[1], frame.shape[0])
            # if drawing_canvas is None:
            #     drawing_canvas = np.zeros_like(frame)

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
            with kalman_tuning_lock:
                kalman_snapshot = KalmanTuning(
                    roll=KalmanAxisTuning(
                        q_angle=kalman_tuning.roll.q_angle,
                        q_bias=kalman_tuning.roll.q_bias,
                        r_measure=kalman_tuning.roll.r_measure,
                    ),
                    pitch=KalmanAxisTuning(
                        q_angle=kalman_tuning.pitch.q_angle,
                        q_bias=kalman_tuning.pitch.q_bias,
                        r_measure=kalman_tuning.pitch.r_measure,
                    ),
                    active_axis=kalman_tuning.active_axis,
                )

            relative_xy_mm: tuple[float, float] | None = None
            pen_marker_entry = next(
                (
                    entry
                    for entry in detected_markers
                    if (entry["family"], entry["marker_id"]) == REFERENCE_MARKER
                    and entry["rvec"] is not None
                    and entry["tvec"] is not None
                ),
                None,
            )
            yaw_reference_entry = next(
                (
                    entry
                    for entry in detected_markers
                    if (entry["family"], entry["marker_id"]) == XY_TARGET_MARKER
                    and entry["rvec"] is not None
                    and entry["tvec"] is not None
                ),
                None,
            )
            pen_marker_relative_yaw: float | None = None
            if pen_marker_entry is not None and yaw_reference_entry is not None:
                relative_pose = relative_transform(
                    pen_marker_entry["rvec"],
                    pen_marker_entry["tvec"],
                    yaw_reference_entry["rvec"],
                    yaw_reference_entry["tvec"],
                )
                _relative_rvec, relative_tvec = transform_to_pose(relative_pose)
                relative_xy_mm = (
                    float(relative_tvec[0][0]) * distance_scale,
                    float(relative_tvec[1][0]) * distance_scale,
                )
                pen_marker_in_reference_pose = relative_transform(
                    yaw_reference_entry["rvec"],
                    yaw_reference_entry["tvec"],
                    pen_marker_entry["rvec"],
                    pen_marker_entry["tvec"],
                )
                pen_marker_relative_yaw = planar_yaw_from_transform(pen_marker_in_reference_pose)

            cv2.putText(
                frame,
                (
                    f"Detected: {total} | "
                    "Press '8'/'7' to tune L | 4/1=q_angle | 5/2=q_bias | 6/3=r_measure | i/k=offset_x | o/l=offset_y | Press 'c' to zero IMU | Press 'space' to clear drawing | Press 'd' for snapshot | Press 'q' to quit"
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
                roll_history_deg.append(imu_roll_zeroed_deg)
                pitch_history_deg.append(imu_pitch_zeroed_deg)
                imu_roll_avg_deg = sum(roll_history_deg) / len(roll_history_deg)
                imu_pitch_avg_deg = sum(pitch_history_deg) / len(pitch_history_deg)
                tip_debug_values: tuple[float, float, float, float] | None = None
                marker_2_seen = False
                for entry in detected_markers:
                    marker_size_mm = entry["marker_size_mm"]
                    if marker_size_mm is None:
                        continue

                    corrected_pts = undistort_marker_corners(
                        entry["corners"],
                        camera_matrix,
                        dist_coeffs,
                    )
                    mx, my, pixels_per_mm = marker_center_and_scale(corrected_pts, float(marker_size_mm))
                    if pixels_per_mm <= 0.0:
                        continue


                    if (entry["family"], entry["marker_id"]) == REFERENCE_MARKER:
                        marker_2_seen = True
                        phi = math.radians(imu_roll_avg_deg)
                        theta = math.radians(imu_pitch_avg_deg)

                        current_marker_yaw_aligned: float | None = None
                        if pen_marker_relative_yaw is not None:
                            if not marker_offsets_initialized:
                                marker_yaw_offset = pen_marker_relative_yaw
                                imu_roll_offset_deg = imu_snapshot.roll_deg
                                imu_pitch_offset_deg = imu_snapshot.pitch_deg
                                marker_offsets_initialized = True
                            current_marker_yaw_aligned = wrap_angle_rad(
                                pen_marker_relative_yaw - marker_yaw_offset
                            )
                        elif last_pen_marker_state is not None:
                            current_marker_yaw_aligned = last_pen_marker_state["marker_yaw_aligned"]

                        if current_marker_yaw_aligned is None:
                            continue

                        pen_height_cm = (
                            (pen_length_mm / 10.0)
                            * math.cos(phi)
                            * math.cos(theta)
                        )
                        marker_height_ratio = pen_height_cm / CAMERA_HEIGHT_CM
                        mx_corrected = mx - (mx - frame_center_x) * marker_height_ratio
                        my_corrected = my - (my - frame_center_y) * marker_height_ratio
                        sensor_offset_x_px = sensor_offset_x_mm * pixels_per_mm
                        sensor_offset_y_px = sensor_offset_y_mm * pixels_per_mm
                        
                        radius_cm = math.sqrt(max(0.0, (pen_length_mm / 10.0) ** 2 - pen_height_cm ** 2))
                        
                        radius_px = radius_cm * pixels_per_mm * 10.0
                        last_pen_marker_state = {
                            "mx": mx,
                            "my": my,
                            "pixels_per_mm": pixels_per_mm,
                            "marker_yaw_aligned": current_marker_yaw_aligned,
                        }
                        tip_x, tip_y, tip_dx, tip_dy = pen_tip(
                            mx_corrected,
                            my_corrected,
                            phi,
                            theta,
                            current_marker_yaw_aligned,
                            radius_px,
                            sensor_offset_x_px,
                            sensor_offset_y_px,
                        )
                        cv2.circle(
                            frame,
                            (int(round(mx)), int(round(my))),
                            int(round(radius_px)),
                            color=(255, 0, 255),
                            thickness=1,
                        )
                        
                        tip_debug_values = (mx_corrected, my_corrected, tip_dx, tip_dy)
                        cv2.circle(frame, (int(round(tip_x)), int(round(tip_y))),2, (0, 0, 255), -1)
                        cv2.line(
                            frame,
                            (int(round(mx)), int(round(my))),
                            (int(round(tip_x)), int(round(tip_y))),
                            (0, 0, 255),
                            2,
                        )
                if not marker_2_seen and last_pen_marker_state is not None:
                    phi = math.radians(imu_roll_avg_deg)
                    theta = math.radians(imu_pitch_avg_deg)
                    pen_height_cm = (
                        (pen_length_mm / 10.0)
                        * math.cos(phi)
                        * math.cos(theta)
                    )
                    marker_height_ratio = pen_height_cm / CAMERA_HEIGHT_CM
                    mx = last_pen_marker_state["mx"]
                    my = last_pen_marker_state["my"]
                    pixels_per_mm = last_pen_marker_state["pixels_per_mm"]
                    mx_corrected = mx - (mx - frame_center_x) * marker_height_ratio
                    my_corrected = my - (my - frame_center_y) * marker_height_ratio
                    sensor_offset_x_px = sensor_offset_x_mm * pixels_per_mm
                    sensor_offset_y_px = sensor_offset_y_mm * pixels_per_mm
                    radius_cm = math.sqrt(max(0.0, (pen_length_mm / 10.0) ** 2 - pen_height_cm ** 2))
                    radius_px = radius_cm * pixels_per_mm * 10.0
                    tip_x, tip_y, tip_dx, tip_dy = pen_tip(
                        mx_corrected,
                        my_corrected,
                        phi,
                        theta,
                        last_pen_marker_state["marker_yaw_aligned"],
                        radius_px,
                        sensor_offset_x_px,
                        sensor_offset_y_px,
                    )
                    cv2.circle(
                        frame,
                        (int(round(mx)), int(round(my))),
                        int(round(radius_px)),
                        color=(120, 0, 255),
                        thickness=1,
                    )
                    tip_debug_values = (mx_corrected, my_corrected, tip_dx, tip_dy)
                    cv2.circle(frame, (int(round(tip_x)), int(round(tip_y))), 2, (0, 0, 255), -1)
                    cv2.line(
                        frame,
                        (int(round(mx)), int(round(my))),
                        (int(round(tip_x)), int(round(tip_y))),
                        (0, 0, 255),
                        2,
                    )
                cv2.putText(
                    frame,
                    (
                        f"IMU roll/pitch avg: {imu_roll_avg_deg:.1f} / {imu_pitch_avg_deg:.1f} deg | "
                        f"L = {pen_length_mm:.1f} mm | "
                        f"offset x/y = {sensor_offset_x_mm:.2f} / {sensor_offset_y_mm:.2f} mm"
                    ),
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                if pen_marker_relative_yaw is not None:
                    yaw_status = wrap_angle_rad(pen_marker_relative_yaw - marker_yaw_offset)
                    yaw_status_deg = math.degrees(yaw_status)
                    yaw_source_text = f"marker yaw rel to {XY_TARGET_MARKER[1]}: {yaw_status_deg:.1f} deg"
                elif marker_offsets_initialized:
                    yaw_source_text = f"marker yaw rel to {XY_TARGET_MARKER[1]}: cached"
                else:
                    yaw_source_text = f"marker yaw rel to {XY_TARGET_MARKER[1]}: waiting for calibration"
                cv2.putText(
                    frame,
                    yaw_source_text,
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (180, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    (
                        f"Kalman active={kalman_snapshot.active_axis.upper()} | "
                        f"roll q={kalman_snapshot.roll.q_angle:.4f} "
                        f"b={kalman_snapshot.roll.q_bias:.4f} "
                        f"r={kalman_snapshot.roll.r_measure:.3f} | "
                        f"pitch q={kalman_snapshot.pitch.q_angle:.4f} "
                        f"b={kalman_snapshot.pitch.q_bias:.4f} "
                        f"r={kalman_snapshot.pitch.r_measure:.3f}"
                    ),
                    (20, 120 if relative_xy_mm is not None else 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 255, 200),
                    2,
                    cv2.LINE_AA,
                )
                if relative_xy_mm is not None:
                    cv2.putText(
                        frame,
                        f"Marker 2 rel to 4: x={relative_xy_mm[0]:.1f} mm y={relative_xy_mm[1]:.1f} mm",
                        (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 200, 0),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    frame,
                    (
                        # f"Pen height h=L*cos(r)*cos(p): {pen_height_cm:.2f} cm | "
                        f"a = cos({pen_height_cm:.2f}/{pen_length_mm/10.0:.2f})="
                        f"{math.degrees(math.acos(pen_height_cm / (pen_length_mm / 10.0))):.2f} deg | "
                        f"Radius : {math.sqrt((pen_length_mm/10.0)**2 - pen_height_cm**2):.1f}cm"
                    ),
                    (20, 180 if relative_xy_mm is not None else 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
                if tip_debug_values is not None:
                    cv2.putText(
                        frame,
                        (
                            f"mx,my=({tip_debug_values[0]:.1f}, {tip_debug_values[1]:.1f}) | "
                            f"r*sin(a)={tip_debug_values[2]:.1f} | "
                            f"r*cos(a)={tip_debug_values[3]:.1f}"
                        ),
                        (20, 210 if relative_xy_mm is not None else 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 150, 255),
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
                imu_roll_zeroed_deg = 0.0
                imu_pitch_zeroed_deg = 0.0
                imu_roll_avg_deg = 0.0
                imu_pitch_avg_deg = 0.0

            cv2.imshow("ArUco + UART Integration", frame)
          
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                if imu_snapshot.ready:
                    imu_roll_offset_deg = imu_snapshot.roll_deg
                    imu_pitch_offset_deg = imu_snapshot.pitch_deg
                    if pen_marker_relative_yaw is not None:
                        marker_yaw_offset = pen_marker_relative_yaw
                        marker_offsets_initialized = True
                    else:
                        print(
                            f"Marker yaw zero requested, but marker {XY_TARGET_MARKER[1]} is not visible. "
                            "Keeping the previous yaw calibration."
                        )
                    roll_history_deg.clear()
                    pitch_history_deg.clear()
                    print(
                        f"IMU zeroed at roll {imu_roll_offset_deg:.2f} deg, "
                        f"pitch {imu_pitch_offset_deg:.2f} deg, "
                        f"marker yaw zero {math.degrees(marker_yaw_offset):.2f} deg, az {imu_az_zero:.3f}"
                    )
                else:
                    print("IMU zero requested, but no UART samples are available yet.")
            elif key == ord("d"):
                print("Snapshot:")
                if imu_snapshot.ready:
                    pen_height_cm = (
                        (pen_length_mm / 10.0)
                        * math.cos(math.radians(imu_roll_avg_deg))
                        * math.cos(math.radians(imu_pitch_avg_deg))
                    )
                    print(
                        f"  IMU -> roll avg {imu_roll_avg_deg:.2f} deg, "
                        f"pitch avg {imu_pitch_avg_deg:.2f} deg "
                        f"(zeroed raw {imu_roll_zeroed_deg:.2f}, {imu_pitch_zeroed_deg:.2f}), "
                        f"(raw {imu_snapshot.roll_deg:.2f}, {imu_snapshot.pitch_deg:.2f}), "
                        f"accel xyz ({imu_snapshot.ax:.3f}, {imu_snapshot.ay:.3f}, {imu_snapshot.az:.3f})"
                    )
                    print(f"  Pen height from IMU -> {pen_height_cm:.2f} cm")
                    print(
                        f"  Sensor offset -> x {sensor_offset_x_mm:.2f} mm, "
                        f"y {sensor_offset_y_mm:.2f} mm"
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
                    corrected_pts = undistort_marker_corners(
                        entry["corners"],
                        camera_matrix,
                        dist_coeffs,
                    )
                    mx, my, pixels_per_mm = marker_center_and_scale(corrected_pts, float(marker_size_mm))
                if relative_xy_mm is not None:
                    print(
                        f"  Marker {XY_TARGET_MARKER[1]} relative to {REFERENCE_MARKER[1]} -> "
                        f"x {relative_xy_mm[0]:.1f} mm, y {relative_xy_mm[1]:.1f} mm"
                    )
            elif key == ord("8"):
                pen_length_mm += PEN_LENGTH_STEP_MM
                print(f"Pen length increased to {pen_length_mm:.1f} mm")
            elif key == ord("i"):
                sensor_offset_x_mm += SENSOR_OFFSET_STEP_MM
                print(f"Sensor offset x increased to {sensor_offset_x_mm:.2f} mm")
            elif key == ord("k"):
                sensor_offset_x_mm -= SENSOR_OFFSET_STEP_MM
                print(f"Sensor offset x decreased to {sensor_offset_x_mm:.2f} mm")
            elif key == ord("o"):
                sensor_offset_y_mm += SENSOR_OFFSET_STEP_MM
                print(f"Sensor offset y increased to {sensor_offset_y_mm:.2f} mm")
            elif key == ord("l"):
                sensor_offset_y_mm -= SENSOR_OFFSET_STEP_MM
                print(f"Sensor offset y decreased to {sensor_offset_y_mm:.2f} mm")
            elif key == ord("r"):
                with kalman_tuning_lock:
                    kalman_tuning.active_axis = "roll"
                    print("Kalman tuning mode set to ROLL")
            elif key == ord("p"):
                with kalman_tuning_lock:
                    kalman_tuning.active_axis = "pitch"
                    print("Kalman tuning mode set to PITCH")
            elif key == ord("4"):
                with kalman_tuning_lock:
                    axis = getattr(kalman_tuning, kalman_tuning.active_axis)
                    axis.q_angle += Q_ANGLE_STEP
                    print(f"{kalman_tuning.active_axis.capitalize()} q_angle increased to {axis.q_angle:.4f}")
            elif key == ord("1"):
                with kalman_tuning_lock:
                    axis = getattr(kalman_tuning, kalman_tuning.active_axis)
                    axis.q_angle = max(Q_ANGLE_STEP, axis.q_angle - Q_ANGLE_STEP)
                    print(f"{kalman_tuning.active_axis.capitalize()} q_angle decreased to {axis.q_angle:.4f}")
            elif key == ord("5"):
                with kalman_tuning_lock:
                    axis = getattr(kalman_tuning, kalman_tuning.active_axis)
                    axis.q_bias += Q_BIAS_STEP
                    print(f"{kalman_tuning.active_axis.capitalize()} q_bias increased to {axis.q_bias:.4f}")
            elif key == ord("2"):
                with kalman_tuning_lock:
                    axis = getattr(kalman_tuning, kalman_tuning.active_axis)
                    axis.q_bias = max(Q_BIAS_STEP, axis.q_bias - Q_BIAS_STEP)
                    print(f"{kalman_tuning.active_axis.capitalize()} q_bias decreased to {axis.q_bias:.4f}")
            elif key == ord("6"):
                with kalman_tuning_lock:
                    axis = getattr(kalman_tuning, kalman_tuning.active_axis)
                    axis.r_measure += R_MEASURE_STEP
                    print(f"{kalman_tuning.active_axis.capitalize()} r_measure increased to {axis.r_measure:.3f}")
            elif key == ord("3"):
                with kalman_tuning_lock:
                    axis = getattr(kalman_tuning, kalman_tuning.active_axis)
                    axis.r_measure = max(R_MEASURE_STEP, axis.r_measure - R_MEASURE_STEP)
                    print(f"{kalman_tuning.active_axis.capitalize()} r_measure decreased to {axis.r_measure:.3f}")
            elif key == ord("7"):
                pen_length_mm = max(PEN_LENGTH_STEP_MM, pen_length_mm - PEN_LENGTH_STEP_MM)
                print(f"Pen length decreased to {pen_length_mm:.1f} mm")
            elif key == ord("q"):
                with kalman_tuning_lock:
                    print("Current tuned values:")
                    print(f"  Pen length -> {pen_length_mm:.1f} mm")
                    print(f"  Sensor offset x -> {sensor_offset_x_mm:.2f} mm")
                    print(f"  Sensor offset y -> {sensor_offset_y_mm:.2f} mm")
                    print(f"  Active axis -> {kalman_tuning.active_axis}")
                    print(f"  Roll q_angle -> {kalman_tuning.roll.q_angle:.4f}")
                    print(f"  Roll q_bias -> {kalman_tuning.roll.q_bias:.4f}")
                    print(f"  Roll r_measure -> {kalman_tuning.roll.r_measure:.3f}")
                    print(f"  Pitch q_angle -> {kalman_tuning.pitch.q_angle:.4f}")
                    print(f"  Pitch q_bias -> {kalman_tuning.pitch.q_bias:.4f}")
                    print(f"  Pitch r_measure -> {kalman_tuning.pitch.r_measure:.3f}")
                break
    finally:
        stop_event.set()
        imu_thread.join(timeout=1.0)
        capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

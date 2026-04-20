from __future__ import annotations

import math

import cv2
import numpy as np

from .models import KalmanTuning, TrackerRuntimeState
from .settings import HEADER_TEXT, XY_TARGET_MARKER


def draw_header(frame: np.ndarray, total: int) -> None:
    cv2.putText(
        frame,
        f"Detected: {total} | {HEADER_TEXT}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_waiting_overlay(frame: np.ndarray) -> None:
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


def draw_runtime_overlay(
    frame: np.ndarray,
    runtime_state: TrackerRuntimeState,
    imu_roll_avg_deg: float,
    imu_pitch_avg_deg: float,
    kalman_snapshot: KalmanTuning,
    relative_xy_mm: tuple[float, float] | None,
    pen_marker_relative_yaw: float | None,
    pen_height_cm: float | None,
    tip_debug_values: tuple[float, float, float, float] | None,
) -> None:
    cv2.putText(
        frame,
        (
            f"IMU roll/pitch avg: {imu_roll_avg_deg:.1f} / {imu_pitch_avg_deg:.1f} deg | "
            f"L = {runtime_state.pen_length_mm:.1f} mm | "
            f"offset x/y = {runtime_state.sensor_offset_x_mm:.2f} / "
            f"{runtime_state.sensor_offset_y_mm:.2f} mm"
        ),
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if pen_marker_relative_yaw is not None:
        yaw_status = math.degrees(
            math.atan2(
                math.sin(pen_marker_relative_yaw - runtime_state.marker_yaw_offset),
                math.cos(pen_marker_relative_yaw - runtime_state.marker_yaw_offset),
            )
        )
        yaw_source_text = f"marker yaw rel to {XY_TARGET_MARKER[1]}: {yaw_status:.1f} deg"
    elif runtime_state.marker_offsets_initialized:
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
        (20, 120),
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

    if pen_height_cm is not None:
        text_y = 180 if relative_xy_mm is not None else 150
        cv2.putText(
            frame,
            (
                f"a = cos({pen_height_cm:.2f}/{runtime_state.pen_length_mm / 10.0:.2f})="
                f"{math.degrees(math.acos(pen_height_cm / (runtime_state.pen_length_mm / 10.0))):.2f} deg | "
                f"Radius : {math.sqrt((runtime_state.pen_length_mm / 10.0) ** 2 - pen_height_cm ** 2):.1f}cm"
            ),
            (20, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

    if tip_debug_values is not None:
        text_y = 210 if relative_xy_mm is not None else 180
        cv2.putText(
            frame,
            (
                f"mx,my=({tip_debug_values[0]:.1f}, {tip_debug_values[1]:.1f}) | "
                f"r*sin(a)={tip_debug_values[2]:.1f} | "
                f"r*cos(a)={tip_debug_values[3]:.1f}"
            ),
            (20, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 150, 255),
            2,
            cv2.LINE_AA,
        )

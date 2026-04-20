from __future__ import annotations

import math
import threading

from .models import ImuState, KalmanTuning, TrackerRuntimeState
from .settings import (
    PEN_LENGTH_STEP_MM,
    Q_ANGLE_STEP,
    Q_BIAS_STEP,
    R_MEASURE_STEP,
    SENSOR_OFFSET_STEP_MM,
    XY_TARGET_MARKER,
)


def handle_runtime_key(
    key: int,
    imu_snapshot: ImuState,
    runtime_state: TrackerRuntimeState,
    kalman_tuning: KalmanTuning,
    kalman_tuning_lock: threading.Lock,
    pen_marker_relative_yaw: float | None,
) -> bool:
    if key == ord("c"):
        if imu_snapshot.ready:
            runtime_state.imu_roll_offset_deg = imu_snapshot.roll_deg
            runtime_state.imu_pitch_offset_deg = imu_snapshot.pitch_deg
            if pen_marker_relative_yaw is not None:
                runtime_state.marker_yaw_offset = pen_marker_relative_yaw
                runtime_state.marker_offsets_initialized = True
                if runtime_state.last_pen_marker_state is not None:
                    runtime_state.last_pen_marker_state.marker_yaw_aligned = 0.0
            else:
                print(
                    f"Marker yaw zero requested, but marker {XY_TARGET_MARKER[1]} "
                    "is not visible. Keeping the previous yaw calibration."
                )
            runtime_state.roll_history_deg.clear()
            runtime_state.pitch_history_deg.clear()
            print(
                f"IMU zeroed at roll {runtime_state.imu_roll_offset_deg:.2f} deg, "
                f"pitch {runtime_state.imu_pitch_offset_deg:.2f} deg, "
                f"marker yaw zero {math.degrees(runtime_state.marker_yaw_offset):.2f} deg"
            )
        else:
            print("IMU zero requested, but no UART samples are available yet.")
        return False

    if key == ord("8"):
        runtime_state.pen_length_mm += PEN_LENGTH_STEP_MM
        print(f"Pen length increased to {runtime_state.pen_length_mm:.1f} mm")
        return False

    if key == ord("7"):
        runtime_state.pen_length_mm = max(
            PEN_LENGTH_STEP_MM,
            runtime_state.pen_length_mm - PEN_LENGTH_STEP_MM,
        )
        print(f"Pen length decreased to {runtime_state.pen_length_mm:.1f} mm")
        return False

    if key == ord("i"):
        runtime_state.sensor_offset_x_mm += SENSOR_OFFSET_STEP_MM
        print(f"Sensor offset x increased to {runtime_state.sensor_offset_x_mm:.2f} mm")
        return False

    if key == ord("k"):
        runtime_state.sensor_offset_x_mm -= SENSOR_OFFSET_STEP_MM
        print(f"Sensor offset x decreased to {runtime_state.sensor_offset_x_mm:.2f} mm")
        return False

    if key == ord("o"):
        runtime_state.sensor_offset_y_mm += SENSOR_OFFSET_STEP_MM
        print(f"Sensor offset y increased to {runtime_state.sensor_offset_y_mm:.2f} mm")
        return False

    if key == ord("l"):
        runtime_state.sensor_offset_y_mm -= SENSOR_OFFSET_STEP_MM
        print(f"Sensor offset y decreased to {runtime_state.sensor_offset_y_mm:.2f} mm")
        return False

    if key == ord("q"):
        with kalman_tuning_lock:
            print("Current tuned values:")
            print(f"  Pen length -> {runtime_state.pen_length_mm:.1f} mm")
            print(f"  Sensor offset x -> {runtime_state.sensor_offset_x_mm:.2f} mm")
            print(f"  Sensor offset y -> {runtime_state.sensor_offset_y_mm:.2f} mm")
            print(f"  Active axis -> {kalman_tuning.active_axis}")
            print(f"  Roll q_angle -> {kalman_tuning.roll.q_angle:.4f}")
            print(f"  Roll q_bias -> {kalman_tuning.roll.q_bias:.4f}")
            print(f"  Roll r_measure -> {kalman_tuning.roll.r_measure:.3f}")
            print(f"  Pitch q_angle -> {kalman_tuning.pitch.q_angle:.4f}")
            print(f"  Pitch q_bias -> {kalman_tuning.pitch.q_bias:.4f}")
            print(f"  Pitch r_measure -> {kalman_tuning.pitch.r_measure:.3f}")
        return True

    if key == ord("r"):
        with kalman_tuning_lock:
            kalman_tuning.active_axis = "roll"
            print("Kalman tuning mode set to ROLL")
        return False

    if key == ord("p"):
        with kalman_tuning_lock:
            kalman_tuning.active_axis = "pitch"
            print("Kalman tuning mode set to PITCH")
        return False

    if key == ord("4"):
        with kalman_tuning_lock:
            axis = getattr(kalman_tuning, kalman_tuning.active_axis)
            axis.q_angle += Q_ANGLE_STEP
            print(f"{kalman_tuning.active_axis.capitalize()} q_angle increased to {axis.q_angle:.4f}")
        return False

    if key == ord("1"):
        with kalman_tuning_lock:
            axis = getattr(kalman_tuning, kalman_tuning.active_axis)
            axis.q_angle = max(Q_ANGLE_STEP, axis.q_angle - Q_ANGLE_STEP)
            print(f"{kalman_tuning.active_axis.capitalize()} q_angle decreased to {axis.q_angle:.4f}")
        return False

    if key == ord("5"):
        with kalman_tuning_lock:
            axis = getattr(kalman_tuning, kalman_tuning.active_axis)
            axis.q_bias += Q_BIAS_STEP
            print(f"{kalman_tuning.active_axis.capitalize()} q_bias increased to {axis.q_bias:.4f}")
        return False

    if key == ord("2"):
        with kalman_tuning_lock:
            axis = getattr(kalman_tuning, kalman_tuning.active_axis)
            axis.q_bias = max(Q_BIAS_STEP, axis.q_bias - Q_BIAS_STEP)
            print(f"{kalman_tuning.active_axis.capitalize()} q_bias decreased to {axis.q_bias:.4f}")
        return False

    if key == ord("6"):
        with kalman_tuning_lock:
            axis = getattr(kalman_tuning, kalman_tuning.active_axis)
            axis.r_measure += R_MEASURE_STEP
            print(f"{kalman_tuning.active_axis.capitalize()} r_measure increased to {axis.r_measure:.3f}")
        return False

    if key == ord("3"):
        with kalman_tuning_lock:
            axis = getattr(kalman_tuning, kalman_tuning.active_axis)
            axis.r_measure = max(R_MEASURE_STEP, axis.r_measure - R_MEASURE_STEP)
            print(f"{kalman_tuning.active_axis.capitalize()} r_measure decreased to {axis.r_measure:.3f}")
        return False

    return False

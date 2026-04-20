from __future__ import annotations

import math
import threading
import time

from UART_reading_esp32c3 import KalmanAngle, connection, read_values

from .models import ImuState, KalmanAxisTuning, KalmanTuning


def imu_reader_worker(
    stop_event: threading.Event,
    state: ImuState,
    state_lock: threading.Lock,
    tuning: KalmanTuning,
    tuning_lock: threading.Lock,
) -> None:
    try:
        ser = connection()
    except Exception as e:
        print(f"Error initializing IMU connection: {e}")
        return
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


def start_imu_thread(
    stop_event: threading.Event,
    state: ImuState,
    state_lock: threading.Lock,
    tuning: KalmanTuning,
    tuning_lock: threading.Lock,
) -> threading.Thread:
    thread = threading.Thread(
        target=imu_reader_worker,
        args=(stop_event, state, state_lock, tuning, tuning_lock),
        name="imu-reader",
        daemon=True,
    )
    thread.start()
    return thread


def snapshot_imu_state(state: ImuState, state_lock: threading.Lock) -> ImuState:
    with state_lock:
        return ImuState(
            ax=state.ax,
            ay=state.ay,
            az=state.az,
            roll_deg=state.roll_deg,
            pitch_deg=state.pitch_deg,
            accel_roll_deg=state.accel_roll_deg,
            accel_pitch_deg=state.accel_pitch_deg,
            last_update_s=state.last_update_s,
            sample_count=state.sample_count,
            ready=state.ready,
        )


def snapshot_kalman_tuning(
    tuning: KalmanTuning,
    tuning_lock: threading.Lock,
) -> KalmanTuning:
    with tuning_lock:
        return KalmanTuning(
            roll=KalmanAxisTuning(
                q_angle=tuning.roll.q_angle,
                q_bias=tuning.roll.q_bias,
                r_measure=tuning.roll.r_measure,
            ),
            pitch=KalmanAxisTuning(
                q_angle=tuning.pitch.q_angle,
                q_bias=tuning.pitch.q_bias,
                r_measure=tuning.pitch.r_measure,
            ),
            active_axis=tuning.active_axis,
        )

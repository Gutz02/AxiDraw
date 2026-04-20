from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .settings import MOVING_AVERAGE_SAMPLES, PEN_LENGTH_MM


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


@dataclass
class DetectedMarker:
    family: str
    marker_id: int
    corners: np.ndarray
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    marker_size_mm: float | None


@dataclass
class LastPenMarkerState:
    mx: float
    my: float
    pixels_per_mm: float
    marker_yaw_aligned: float


@dataclass
class TrackerRuntimeState:
    imu_roll_offset_deg: float = 0.0
    imu_pitch_offset_deg: float = 0.0
    marker_yaw_offset: float = 0.0
    marker_offsets_initialized: bool = False
    pen_length_mm: float = PEN_LENGTH_MM
    sensor_offset_x_mm: float = 0.0
    sensor_offset_y_mm: float = 0.0
    last_pen_marker_state: LastPenMarkerState | None = None
    roll_history_deg: deque[float] = field(
        default_factory=lambda: deque(maxlen=MOVING_AVERAGE_SAMPLES)
    )
    pitch_history_deg: deque[float] = field(
        default_factory=lambda: deque(maxlen=MOVING_AVERAGE_SAMPLES)
    )

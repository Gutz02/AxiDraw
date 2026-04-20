from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .settings import CAMERA_HEIGHT_MM


@dataclass
class TipProjection:
    tip_x: float
    tip_y: float
    tip_dx: float
    tip_dy: float
    radius_px: float
    mx_corrected: float
    my_corrected: float
    pen_height_cm: float


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
    dx_global = dx * math.cos(yaw) - dy * math.sin(yaw)
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


def marker_center_and_scale(
    marker_corners: np.ndarray,
    marker_size_mm: float,
) -> tuple[float, float, float]:
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

def tip_wrt_marker(pen_length_mm: float, phi: float, theta: float):
    pen_height_mm = (pen_length_mm) * math.cos(phi) * math.cos(theta)
    radius_mm = math.sqrt(max(0.0, pen_length_mm ** 2 - pen_height_mm ** 2))
    alpha = math.atan2(-math.sin(phi) * math.cos(theta), math.sin(theta))
    tx = radius_mm*math.sin(alpha)
    ty = radius_mm*math.cos(alpha)
    tz = -pen_height_mm
    return np.array([tx, ty, tz], dtype=np.float32)


def project_tip(
    mx: float,
    my: float,
    pixels_per_mm: float,
    frame_center_x: float,
    frame_center_y: float,
    pen_length_mm: float,
    sensor_offset_x_mm: float,
    sensor_offset_y_mm: float,
    phi: float,
    theta: float,
    yaw_aligned: float,
) -> TipProjection:
    pen_height_mm = (pen_length_mm) * math.cos(phi) * math.cos(theta)
    marker_height_ratio = pen_height_mm / CAMERA_HEIGHT_MM
    mx_corrected = mx - (mx - frame_center_x) * marker_height_ratio
    my_corrected = my - (my - frame_center_y) * marker_height_ratio
    radius_mm = math.sqrt(max(0.0, pen_length_mm ** 2 - pen_height_mm ** 2))
    tip_x, tip_y, tip_dx, tip_dy = pen_tip(
        mx_corrected,
        my_corrected,
        phi,
        theta,
        yaw_aligned,
        radius_mm,
        sensor_offset_x_mm,
        sensor_offset_y_mm,
    )
    return TipProjection(
        tip_x=tip_x,
        tip_y=tip_y,
        tip_dx=tip_dx,
        tip_dy=tip_dy,
        radius_px=radius_mm,
        mx_corrected=mx_corrected,
        my_corrected=my_corrected,
        pen_height_cm=pen_height_mm,
    )

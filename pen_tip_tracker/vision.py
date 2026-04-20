from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from aruco_marker_detection import (
    MARKERS_DIR,
    REFERENCE_MARKER,
    TARGET_MARKER,
    estimate_marker_pose,
    family_to_opencv_constant,
    load_expected_markers,
    plane_distance_mm,
    relative_transform,
    transform_to_pose,
)

from .geometry import planar_yaw_from_transform
from .models import DetectedMarker
from .settings import XY_TARGET_MARKER


@dataclass
class DetectorContext:
    family: str
    expected_marker_ids: set[int]
    marker_sizes_mm: dict[tuple[str, int], float]
    ambiguous_markers: set[tuple[str, int]]
    detector: cv2.aruco.ArucoDetector


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


def load_detector_context() -> DetectorContext:
    expected_markers, marker_sizes_mm, ambiguous_markers = load_expected_markers(MARKERS_DIR)
    families = list(expected_markers)
    if len(families) != 1:
        raise RuntimeError(
            "pen_tip_tracker is configured for exactly one ArUco family, "
            f"but found: {', '.join(sorted(families))}"
        )
    family = families[0]
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(family_to_opencv_constant(family)),
        build_detector_parameters(),
    )
    return DetectorContext(
        family=family,
        expected_marker_ids=expected_markers[family],
        marker_sizes_mm=marker_sizes_mm,
        ambiguous_markers=ambiguous_markers,
        detector=detector,
    )


def detect_markers(
    frame: np.ndarray,
    gray: np.ndarray,
    detector_context: DetectorContext,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[list[DetectedMarker], int]:
    total = 0
    detected_markers: list[DetectedMarker] = []
    family = detector_context.family

    corners, ids, _rejected = detector_context.detector.detectMarkers(gray)
    if ids is None:
        return detected_markers, total

    for marker_corners, marker_id_array in zip(corners, ids):
        marker_id = int(marker_id_array[0])
        if marker_id not in detector_context.expected_marker_ids:
            continue

        total += 1
        refined_corners = refine_marker_corners(gray, marker_corners)
        marker_key = (family, marker_id)
        marker_size_mm = detector_context.marker_sizes_mm.get(marker_key)
        pose = None
        if marker_size_mm is not None:
            pose = estimate_marker_pose(
                refined_corners,
                marker_size_mm,
                camera_matrix,
                dist_coeffs,
            )

        rvec, tvec = pose if pose is not None else (None, None)
        detected_markers.append(
            DetectedMarker(
                family=family,
                marker_id=marker_id,
                corners=refined_corners,
                rvec=rvec,
                tvec=tvec,
                marker_size_mm=marker_size_mm,
            )
        )

        pts = marker_corners.reshape((4, 2)).astype(int)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        for i in range(4):
            cv2.circle(frame, tuple(pts[i]), 2, (255, 0, 0), -1)

        marker_label = f"{family}:{marker_id}"
        if tvec is not None and marker_size_mm is not None:
            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
                marker_size_mm * 0.5,
                2,
            )
        elif marker_key in detector_context.ambiguous_markers:
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

    return detected_markers, total


def find_marker(
    detected_markers: list[DetectedMarker],
    marker_key: tuple[str, int],
) -> DetectedMarker | None:
    for entry in detected_markers:
        if (
            (entry.family, entry.marker_id) == marker_key
            and entry.rvec is not None
            and entry.tvec is not None
        ):
            return entry
    return None


def compute_pose_alignment(
    detected_markers: list[DetectedMarker],
    distance_scale: float,
) -> tuple[tuple[float, float] | None, float | None]:
    pen_marker_entry = find_marker(detected_markers, TARGET_MARKER)
    reference_marker = find_marker(detected_markers, REFERENCE_MARKER)
    if pen_marker_entry is None or reference_marker is None:
        return None, None

    # this makes the pen marker in the frame of the reference marker
    pen_marker_in_reference_pose = relative_transform(# This returns a 4x4 matrix
        reference_marker.rvec,
        reference_marker.tvec,
        pen_marker_entry.rvec,
        pen_marker_entry.tvec,
    )
    
    _relative_rvec, relative_tvec = transform_to_pose(pen_marker_in_reference_pose) # This converts the 4x4 matrix to rvec and tvec, where tvec is a 3x1 vector representing translation in x, y, z
    
    relative_xy_mm = (
        float(relative_tvec[0][0]) * distance_scale,
        float(relative_tvec[1][0]) * distance_scale,
    )
    
    pen_marker_relative_yaw = planar_yaw_from_transform(pen_marker_in_reference_pose)
    return relative_xy_mm, pen_marker_relative_yaw


def print_vision_snapshot(
    detected_markers: list[DetectedMarker],
    distance_scale: float,
    relative_xy_mm: tuple[float, float] | None,
) -> None:
    if not detected_markers:
        print("  Vision -> no expected ArUco marker detected")
        return

    for entry in detected_markers:
        marker_key = (entry.family, entry.marker_id)
        if entry.marker_size_mm is None:
            print(
                f"  {entry.family}:{entry.marker_id} -> pose unavailable; "
                "set size in MARKER_SIZE_MM_BY_ID or DEFAULT_MARKER_SIZE_MM"
            )
            continue
        if entry.rvec is None or entry.tvec is None:
            print(f"  {entry.family}:{entry.marker_id} -> pose solve failed")
            continue

        raw_distance_mm = plane_distance_mm(entry.rvec, entry.tvec)
        distance_mm = raw_distance_mm * distance_scale
        z_mm = float(entry.tvec[2][0]) * distance_scale
        print(
            f"  {entry.family}:{entry.marker_id} ({entry.marker_size_mm:.1f} mm) -> "
            f"plane distance {distance_mm:.1f} mm, raw {raw_distance_mm:.1f} mm, "
            f"z {z_mm:.1f} mm"
        )

        if marker_key == REFERENCE_MARKER and relative_xy_mm is not None:
            print(
                f"  Marker {XY_TARGET_MARKER[1]} relative to {REFERENCE_MARKER[1]} -> "
                f"x {relative_xy_mm[0]:.1f} mm, y {relative_xy_mm[1]:.1f} mm"
            )

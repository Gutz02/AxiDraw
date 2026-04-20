"""
MVP ArUco detector.

- Reads expected markers from `markers/`
- Uses camera source 2
- Draws detected marker outlines and IDs
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import cv2
import numpy as np

from iphone_connection import connect_camera, read_frame


PROJECT_ROOT = Path(__file__).resolve().parent
MARKERS_DIR = PROJECT_ROOT / "markers"
CALIBRATION_FILE = PROJECT_ROOT / "camera_calibration.npz"
SOURCE = 1
MARKER_PATTERN = re.compile(
    r"(?P<family>\d+x\d+_\d+)-(?P<marker_id>\d+)(?:_(?P<size_mm>\d+)mm)?(?:_[^.]+)*\.(svg|png|jpg|jpeg)$",
    re.IGNORECASE,
)

CAMERA_HEIGHT = 895.0  # in mm
DISTANCE_SCALE_CORRECTION = 0.771
DEFAULT_MARKER_SIZE_MM: float | None = 55.0
REFERENCE_MARKER = ("4x4_1000", 4)
TARGET_MARKER = ("4x4_1000", 0)
MARKER_SIZE_MM_BY_ID: dict[tuple[str, int], float] = {
    ("4x4_1000", 4): 56.0,
    ("4x4_1000", 2): 29.0
}


def family_to_opencv_constant(family: str) -> int:
    constant_name = f"DICT_{family.upper()}"
    if not hasattr(cv2.aruco, constant_name):
        raise ValueError(f"Unsupported ArUco dictionary: {family}")
    return getattr(cv2.aruco, constant_name)


def load_expected_markers(
    markers_dir: Path,
) -> tuple[dict[str, set[int]], dict[tuple[str, int], float], set[tuple[str, int]]]:
    expected: dict[str, set[int]] = {}
    marker_sizes: dict[tuple[str, int], set[float]] = {}

    for path in markers_dir.iterdir():
        if not path.is_file():
            continue
        match = MARKER_PATTERN.fullmatch(path.name)
        if not match:
            continue

        family = match.group("family").lower()
        marker_id = int(match.group("marker_id"))
        expected.setdefault(family, set()).add(marker_id)
        marker_key = (family, marker_id)
        size_mm = match.group("size_mm")
        if size_mm is not None:
            marker_sizes.setdefault(marker_key, set()).add(float(size_mm))

    if not expected:
        raise RuntimeError("No valid marker files found in markers directory.")

    resolved_sizes: dict[tuple[str, int], float] = {}
    ambiguous_sizes: set[tuple[str, int]] = set()

    for family, marker_ids in expected.items():
        for marker_id in marker_ids:
            marker_key = (family, marker_id)
            configured_size = MARKER_SIZE_MM_BY_ID.get(marker_key)
            if configured_size is not None:
                resolved_sizes[marker_key] = configured_size
                continue

            known_sizes = marker_sizes.get(marker_key, set())
            if len(known_sizes) == 1:
                resolved_sizes[marker_key] = next(iter(known_sizes))
            elif DEFAULT_MARKER_SIZE_MM is not None:
                resolved_sizes[marker_key] = DEFAULT_MARKER_SIZE_MM
            else:
                ambiguous_sizes.add(marker_key)

    return expected, resolved_sizes, ambiguous_sizes


def build_camera_matrix(frame_width: int, frame_height: int) -> np.ndarray:
    # Replace this with real camera calibration for accurate metric pose.
    focal_length_px = float(max(frame_width, frame_height))
    return np.array(
        [
            [focal_length_px, 0.0, frame_width / 2.0],
            [0.0, focal_length_px, frame_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def load_camera_calibration() -> tuple[np.ndarray, np.ndarray] | None:
    if not CALIBRATION_FILE.exists():
        return None

    calibration = np.load(CALIBRATION_FILE)
    camera_matrix = calibration["camera_matrix"].astype(np.float32)
    dist_coeffs = calibration["dist_coeffs"].astype(np.float32)
    return camera_matrix, dist_coeffs


def marker_object_points(marker_size_mm: float) -> np.ndarray:
    half_size = marker_size_mm / 2.0
    return np.array(
        [
            [-half_size, half_size, 0.0],
            [half_size, half_size, 0.0],
            [half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0],
        ],
        dtype=np.float32,
    )


def estimate_marker_pose(
    marker_corners: np.ndarray,
    marker_size_mm: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    success, rvec, tvec = cv2.solvePnP(
        marker_object_points(marker_size_mm),
        marker_corners.reshape((4, 2)).astype(np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return None
    return rvec, tvec


def plane_distance_mm(rvec: np.ndarray, tvec: np.ndarray) -> float:
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    plane_normal = rotation_matrix[:, 2]
    return abs(float(np.dot(plane_normal, tvec.reshape(3))))


def euler_angles_deg(rvec: np.ndarray) -> tuple[float, float, float]:
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        yaw = 0.0

    return tuple(math.degrees(angle) for angle in (roll, pitch, yaw))


def pose_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = tvec.reshape(3)
    return transform


def transform_to_pose(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    tvec = transform[:3, 3].reshape(3, 1)
    return rvec, tvec


def relative_transform(
    reference_rvec: np.ndarray,
    reference_tvec: np.ndarray,
    marker_rvec: np.ndarray,
    marker_tvec: np.ndarray,
) -> np.ndarray:
    camera_from_reference = pose_to_transform(reference_rvec, reference_tvec)
    camera_from_marker = pose_to_transform(marker_rvec, marker_tvec)
    return np.linalg.inv(camera_from_reference) @ camera_from_marker


def main() -> None:
    expected_markers, marker_sizes_mm, ambiguous_markers = load_expected_markers(MARKERS_DIR)
    detectors = {
        family: cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(family_to_opencv_constant(family)),
            cv2.aruco.DetectorParameters(),
        )
        for family in expected_markers
    }

    capture = connect_camera(source=SOURCE, width=960, height=540)
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

    try:
        while True:
            frame = read_frame(capture, mirror=False)
            if camera_matrix is None:
                camera_matrix = build_camera_matrix(frame.shape[1], frame.shape[0])

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            total = 0
            detected_markers: list[dict[str, object]] = []

            for family, detector in detectors.items():
                corners, ids, rejected = detector.detectMarkers(gray)

                if ids is None:
                    continue

                for marker_corners, marker_id_array in zip(corners, ids):
                    marker_id = int(marker_id_array[0])
                    if marker_id not in expected_markers[family]:
                        continue

                    total += 1
                    marker_key = (family, marker_id)
                    pose = None
                    marker_size_mm = marker_sizes_mm.get(marker_key)
                    if marker_size_mm is not None:
                        pose = estimate_marker_pose(marker_corners, marker_size_mm, camera_matrix, dist_coeffs)

                    rvec, tvec = pose if pose is not None else (None, None)
                    detected_markers.append(
                        {
                            "family": family,
                            "marker_id": marker_id,
                            "corners": marker_corners,
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
                        distance_mm = plane_distance_mm(rvec, tvec) * distance_scale
                        marker_label += f" {distance_mm:.0f}mm"
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

            reference_entry = next(
                (
                    entry
                    for entry in detected_markers
                    if (entry["family"], entry["marker_id"]) == REFERENCE_MARKER
                    and entry["rvec"] is not None
                    and entry["tvec"] is not None
                ),
                None,
            )

            if reference_entry is not None:
                reference_rvec = reference_entry["rvec"]
                reference_tvec = reference_entry["tvec"]
                for entry in detected_markers:
                    marker_key = (entry["family"], entry["marker_id"])
                    if marker_key == REFERENCE_MARKER:
                        continue
                    if entry["rvec"] is None or entry["tvec"] is None:
                        continue

                    relative_pose = relative_transform(
                        reference_rvec,
                        reference_tvec,
                        entry["rvec"],
                        entry["tvec"],
                    )
                    relative_rvec, relative_tvec = transform_to_pose(relative_pose)
                    relative_height_mm = abs(float(relative_tvec[2][0])) * distance_scale
                    entry["relative_rvec"] = relative_rvec
                    entry["relative_tvec"] = relative_tvec
                    entry["relative_height_mm"] = relative_height_mm

                    pts = entry["corners"].reshape((4, 2)).astype(int)
                    cv2.putText(
                        frame,
                        f"ref z={relative_height_mm:.0f}mm",
                        (pts[0][0], min(pts[0][1] + 25, frame.shape[0] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            cv2.putText(
                frame,
                "Detected: "
                f"{total} | Ref {REFERENCE_MARKER[1]} | Press 'd' for pose/height | Press 'q' to quit",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("ArUco Detection MVP", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("d"):
                if not detected_markers:
                    print("No expected ArUco marker detected in the current frame.")
                    continue

                pose_distances: list[float] = []
                print("Detected marker pose:")
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
                    pose_distances.append(distance_mm)
                    roll_deg, pitch_deg, yaw_deg = euler_angles_deg(rvec)
                    z_mm = float(tvec[2][0]) * distance_scale
                    distance_delta = distance_mm - CAMERA_HEIGHT
                    print(
                        f"  {family}:{marker_id} ({marker_size_mm:.1f} mm) -> "
                        f"plane distance {distance_mm:.1f} mm, raw {raw_distance_mm:.1f} mm, z {z_mm:.1f} mm, "
                        f"rpy ({roll_deg:.1f}, {pitch_deg:.1f}, {yaw_deg:.1f}) deg, "
                        f"delta vs {CAMERA_HEIGHT:.1f} mm = {distance_delta:+.1f} mm"
                    )

                    if marker_key != REFERENCE_MARKER and "relative_tvec" in entry:
                        rel_tvec = entry["relative_tvec"]
                        rel_rvec = entry["relative_rvec"]
                        rel_roll_deg, rel_pitch_deg, rel_yaw_deg = euler_angles_deg(rel_rvec)
                        rel_x = float(rel_tvec[0][0]) * distance_scale
                        rel_y = float(rel_tvec[1][0]) * distance_scale
                        rel_z = float(rel_tvec[2][0]) * distance_scale
                        print(
                            f"    Relative to {REFERENCE_MARKER[0]}:{REFERENCE_MARKER[1]} -> "
                            f"x {rel_x:.1f} mm, y {rel_y:.1f} mm, z {rel_z:.1f} mm, "
                            f"rpy ({rel_roll_deg:.1f}, {rel_pitch_deg:.1f}, {rel_yaw_deg:.1f}) deg"
                        )

                if pose_distances:
                    average_distance = sum(pose_distances) / len(pose_distances)
                    average_delta = average_distance - CAMERA_HEIGHT
                    print(
                        f"  Average plane distance -> {average_distance:.1f} mm "
                        f"(delta {average_delta:+.1f} mm vs {CAMERA_HEIGHT:.1f} mm)"
                    )

                target_entry = next(
                    (
                        entry
                        for entry in detected_markers
                        if (entry["family"], entry["marker_id"]) == TARGET_MARKER
                        and "relative_height_mm" in entry
                    ),
                    None,
                )
                if target_entry is not None:
                    print(
                        f"  Target {TARGET_MARKER[0]}:{TARGET_MARKER[1]} height above "
                        f"{REFERENCE_MARKER[0]}:{REFERENCE_MARKER[1]} plane -> "
                        f"{target_entry['relative_height_mm']:.1f} mm"
                    )
                elif reference_entry is None:
                    print(
                        f"  Reference marker {REFERENCE_MARKER[0]}:{REFERENCE_MARKER[1]} not visible; "
                        f"cannot compute relative height."
                    )
            elif key == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

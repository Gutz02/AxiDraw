# pen_tip_tracker

This directory is the split version of the pen tip tracker that used to live in `aruco_uart_integration.py`.

Run it from the repo root with:

```bash
python -m pen_tip_tracker
```

## Files

- `app.py`: main loop and orchestration
- `imu.py`: UART reading thread and Kalman-filtered IMU state
- `vision.py`: ArUco detection, pose estimation, and relative yaw extraction
- `geometry.py`: pen-tip projection math
- `overlay.py`: OpenCV text overlay drawing
- `controls.py`: keyboard handling and tuning
- `models.py`: shared dataclasses
- `settings.py`: constants

## Big Picture

The tracker combines two independent signals:

1. Vision gives the pose of the pen marker relative to the camera and relative to marker 2.
2. IMU gives the pen tilt angles `roll` and `pitch`.

The code uses vision to answer:
- Where is the marker center in the image?
- What is the pen marker yaw relative to the reference marker?

The code uses the IMU to answer:
- How much is the pen tilted away from vertical?

Then it combines them:
- IMU tilt gives a local tip displacement direction and radius.
- Marker yaw rotates that local displacement into the image/reference frame.
- The result is drawn as the red estimated pen tip.

## Data Flow

### 1. Camera frame enters the app

In `app.py`, the loop does:

```python
frame = read_frame(capture, mirror=False)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```

That frame is sent to `vision.detect_markers(...)`.

### 2. ArUco markers become poses

In `vision.py`:

- `detect_markers(...)` finds corners with OpenCV ArUco.
- `refine_marker_corners(...)` improves corner precision with `cornerSubPix`.
- `estimate_marker_pose(...)` from `aruco_marker_detection.py` solves for:
  - `rvec`: marker orientation
  - `tvec`: marker position

Those values are stored as `DetectedMarker`.

Important:
- `tvec` is translation from camera to marker in 3D.
- `rvec` is axis-angle rotation for the marker orientation.
- `cv2.Rodrigues(rvec)` turns that into a 3x3 rotation matrix.

### 3. Relative marker pose becomes yaw

Still in `vision.py`, `compute_pose_alignment(...)` does two related things:

1. It computes marker 2 relative to the pen marker, mainly for the debug XY readout.
2. It computes the pen marker pose relative to marker 2, then extracts planar yaw.

The key line is conceptually:

```python
pen_marker_in_reference_pose = relative_transform(reference_marker, pen_marker)
```

That gives a transform whose rotation part says:
- how the pen marker axes are oriented in marker 2's frame

Then `planar_yaw_from_transform(...)` in `geometry.py` does:

- take the pen marker local X axis from the 3x3 rotation matrix
- project it into the reference XY plane
- compute `atan2(y, x)`

That gives the pen marker heading in the reference frame.

This is the pose-based replacement for the older "top edge of the square in image space" approximation.

### 4. IMU thread continuously updates tilt

`imu.py` runs `imu_reader_worker(...)` in a background thread.

It reads:
- accelerometer `ax, ay, az`
- gyro `gx, gy`

It computes:

```python
accel_roll = atan2(ay, az)
accel_pitch = atan2(-ax, sqrt(ay^2 + az^2))
```

Then the Kalman filters fuse accel + gyro into smoother:
- `roll_deg`
- `pitch_deg`

The main loop only reads snapshots of that state. It does not do serial I/O itself.

### 5. The app calibrates zero state

There are two zero references stored in `TrackerRuntimeState`:

- `imu_roll_offset_deg` and `imu_pitch_offset_deg`
- `marker_yaw_offset`

Those are set:
- automatically the first time a valid marker-relative yaw is available
- manually again when you press `c`

So the runtime uses change-from-initial-state, not absolute orientation.

That means:
- current IMU roll = measured roll - calibration roll
- current IMU pitch = measured pitch - calibration pitch
- current marker yaw = measured relative yaw - calibration yaw

This is why the initial calibration pose defines "zero yaw".

### 6. Geometry projects marker center to estimated tip

The tip projection happens in `geometry.project_tip(...)`.

Inputs:
- marker center in image pixels
- marker pixel scale
- IMU tilt `phi` and `theta`
- yaw aligned to calibration
- pen length
- sensor offsets

It computes:

1. Pen height:

```python
pen_height_cm = L * cos(phi) * cos(theta)
```

This is the vertical component of the pen length.

2. Horizontal radius from marker to tip:

```python
radius_cm = sqrt(L^2 - pen_height_cm^2)
```

This is the in-plane distance from the tracked marker to the tip.

3. Perspective-ish center correction:

```python
mx_corrected = mx - (mx - frame_center_x) * marker_height_ratio
my_corrected = my - (my - frame_center_y) * marker_height_ratio
```

This tries to compensate for the fact that the marker is elevated above the writing plane.

4. Local tip vector from tilt:

In `pen_tip(...)`:

```python
a = atan2(-sin(phi) * cos(theta), sin(theta))
dx = r * sin(a)
dy = r * cos(a)
```

This creates a 2D displacement vector in the pen's local frame.

5. Rotate local vector into the marker/reference frame:

```python
dx_global = dx * cos(yaw) - dy * sin(yaw)
dy_global = dx * sin(yaw) + dy * cos(yaw)
```

6. Subtract that vector from the marker center to get tip position:

```python
tx = mx - dx_global
ty = my - dy_global
```

That is the red point and red line you see on screen.

## Control Flow

The main loop in `app.py` is:

1. Read frame.
2. Detect markers and estimate poses.
3. Snapshot IMU and Kalman state.
4. Compute pen marker yaw relative to marker 2.
5. If IMU is ready:
   - zero against calibration offsets
   - smooth roll/pitch with a small moving average
   - project pen tip
   - draw overlays
6. If IMU is not ready:
   - draw waiting text
7. Handle keyboard input:
   - `c`: recalibrate IMU zero and marker yaw zero
   - `d`: print snapshot/debug info
   - `8/7`: adjust pen length
   - `i/k`, `o/l`: sensor offsets
   - `r/p`, `1-6`: Kalman tuning
   - `q`: print values and exit

## What Looks Trimmable

Yes, there is probably fat here.

The likely trim candidates are:

- `overlay.py`
  Lots of code here is only status text and debugging.

- `controls.py`
  This is mostly tuning/debug ergonomics, not core tracking.

- the moving-average history in `TrackerRuntimeState`
  With `MOVING_AVERAGE_SAMPLES = 1`, it currently does nothing useful.

- duplicate debug/snapshot reporting
  The printed snapshot path and the overlay path repeat some of the same information.

- cached pen marker fallback
  The "use last marker state if marker is temporarily lost" path adds complexity. You may or may not need it.

- center correction math
  The `mx_corrected/my_corrected` shift is a modeling choice, not a requirement for a minimal tracker.

If you want the smallest viable core, it is really just:

1. detect pen marker + marker 2
2. compute relative yaw
3. read IMU roll/pitch
4. calibrate zero offsets
5. project the tip
6. draw result

Everything else is support code.

## Profiling

Yes. There are several reasonable ways to profile this.

### 1. Cheap timing with `time.perf_counter()`

Best first step. Add timers around:

- frame capture
- marker detection
- pose alignment
- IMU snapshot
- tip projection
- overlay drawing

Example:

```python
start = time.perf_counter()
detected_markers, total = detect_markers(...)
detect_ms = (time.perf_counter() - start) * 1000.0
```

This is the best way to find whether your time is going into:
- camera I/O
- ArUco detection
- pose estimation
- drawing

### 2. `cProfile` for Python overhead

Run:

```bash
python -m cProfile -o pen_tip_tracker.prof -m pen_tip_tracker
```

Then inspect:

```bash
python -c "import pstats; p = pstats.Stats('pen_tip_tracker.prof'); p.sort_stats('cumtime').print_stats(30)"
```

This is useful for Python-heavy code, but less useful for time spent inside OpenCV C functions.

### 3. `py-spy` for live sampling

If you want a clearer picture of where time is going without modifying code:

```bash
py-spy top -- python -m pen_tip_tracker
```

or:

```bash
py-spy record -o profile.svg -- python -m pen_tip_tracker
```

This is often the most practical profiler for a real-time loop.

### 4. Measure frame rate directly

The simplest high-level metric is loop time / FPS.

Track:

```python
fps = 1.0 / dt
```

If FPS collapses when both markers are visible, the likely hotspot is ArUco detection or pose solve.

## Suggested Audit Order

If your goal is to trim code safely, I would read it in this order:

1. `settings.py`
2. `models.py`
3. `imu.py`
4. `vision.py`
5. `geometry.py`
6. `app.py`
7. `controls.py`
8. `overlay.py`

That order matches the actual dependency chain and gets you to the important math before the UI/debug plumbing.

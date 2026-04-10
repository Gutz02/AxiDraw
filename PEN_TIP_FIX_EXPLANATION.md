# Pen Tip Projection Fix - Rigorous Analysis & Solution

## Problem Statement
The red tip indicator was appearing flipped/mirrored or on the wrong side of the marker despite already trying:
- Swapping roll⇄pitch parameters
- Inverting signs with `TIP_ROLL_SIGN` and `TIP_PITCH_SIGN` constants

This suggested the **formula itself was incorrect**, not just parameter mapping.

---

## Coordinate System Analysis

### Image Plane (Camera Looking Down)
- Origin: Marker center
- +X axis: Right
- +Y axis: Down (standard OpenCV convention)

### IMU Frame (Standard Convention, Mounted on Pen)
From `UART_reading_esp32c3.py`, the Kalman filters compute:
```python
accel_roll  = atan2(ay, az)   # Right-hand rule: rotation around +X axis
accel_pitch = atan2(-ax, sqrt(ay²+az²))  # Right-hand rule: rotation around +Y axis
```

This is the **standard aerospace convention**:
- **Z points up** (against gravity at rest)
- **X points forward** (pen's forward when lying on table)  
- **Y points right** (perpendicular to pen motion)

### Pen Geometry (Top-Down Camera)
**Assumption**: Pen lies flat on table, **pointing downward (+Y) in the image plane**.

This is physically reasonable: marker on top end, tip extends downward.

---

## Geometric Derivation

### Physical Setup
- Pen center at marker: **(Mx, My)**
- Pen length: **L** pixels
- Pen initially points in +Y direction (downward in image)

### When Pen Tilts

**Roll (X-axis rotation)**: Pen rotates in the YZ plane
- Pen tilts **left or right**
- Tip moves **horizontally** (±X direction)
- For angle φ_roll: **ΔX = L·sin(φ_roll)**

**Pitch (Y-axis rotation)**: Pen rotates in the XZ plane  
- Pen tilts **forward or backward**
- Tip moves **vertically** (±Y direction)
- For angle φ_pitch: **ΔY = L·sin(φ_pitch)**

### Final Formula
```
Tip position = Center + Displacement
tx = mx + L·sin(roll)
ty = my + L·sin(pitch)
```

**Why sin() not cos()?**
- cos() would give the along-pen foreshortening (vertical height component)
- sin() gives the perpendicular displacement in the image plane
- This is valid for both small angles AND reasonable angles (up to ~60°)

---

## Why the Old Formula Failed

The old `pen_tip` formula was:
```python
h = L * cos(φ) * cos(θ)          # Trying to compute 3D height?
r = sqrt(L² - h²)                # Pythagorean on wrong components
α = atan2(-sin(φ), tan(θ))       # Complex mixing of angles
tx = mx - d*cos(φ) + r*sin(α)    # Unclear geometric meaning
ty = my + r*cos(α)
```

**Problems:**
1. **Designed for 3D perspective projection**, not 2D image plane projection
2. **For small angles**: r ≈ 0, so tip barely moves ✗
3. **No notion of pen azimuth**: Assumed pen always points same direction, but didn't state which
4. **Sign flipping instead of geometry**: All those constants and swaps obscure what's actually happening

---

## The Fix

### New Function
```python
def pen_tip(mx: float, my: float, roll_deg: float, pitch_deg: float, l: float) -> tuple[float, float]:
    """
    Project pen tip position in image coordinates.
    
    Assumes pen always points downward (+Y) in the image.
    - roll (deg): Rotation around X-axis → horizontal displacement in image
    - pitch (deg): Rotation around Y-axis → vertical displacement in image
    - l (px): Pen length in image pixels
    
    For top-down camera looking at tilted pen:
      - Roll > 0 tilts pen right → tip moves right (+X)
      - Pitch > 0 tilts pen forward → tip moves down (+Y)
    """
    roll_rad = np.radians(roll_deg)
    pitch_rad = np.radians(pitch_deg)

    delta_x = l * np.sin(roll_rad)
    delta_y = l * np.sin(pitch_rad)

    tx = mx + delta_x
    ty = my + delta_y
    return float(tx), float(ty)
```

### Changes at Call Sites
**Before:**
```python
tip_x, tip_y = pen_tip(
    mx, my,
    TIP_PITCH_SIGN * imu_roll_zeroed_deg,      # Swapped + negated!
    TIP_ROLL_SIGN * imu_pitch_zeroed_deg,      # Swapped + negated!
    pen_length_px,
    d=0.0,
)
```

**After:**
```python
tip_x, tip_y = pen_tip(
    mx, my,
    imu_roll_zeroed_deg,        # Roll directly
    imu_pitch_zeroed_deg,       # Pitch directly
    pen_length_px,              # Removed d parameter
)
```

---

## Validation Procedure

Run these calibration tests to verify the fix:

### Test 1: Roll-Only (Keep Pitch at Zero)
1. Press 'c' to zero the IMU
2. **Tilt the pen to the RIGHT** (~20-30°)
3. **Expected**: Red tip point moves to the **RIGHT** (positive X direction)
4. **If inverted**: Negate `roll_deg` by adding `roll_deg = -imu_roll_zeroed_deg` in pen_tip call

### Test 2: Pitch-Only (Keep Roll at Zero)  
1. Press 'c' to zero the IMU
2. **Tilt the pen FORWARD** (~20-30°)
3. **Expected**: Red tip point moves **DOWN** (positive Y direction)
4. **If inverted**: Negate `pitch_deg` by adding `pitch_deg = -imu_pitch_zeroed_deg` in pen_tip call

### Test 3: Both Axes
1. Tilt pen **diagonally** (both roll and pitch non-zero)
2. Verify tip moves in the **diagonal** direction  
3. Verify tip distance scales with L (try changing PEN_LENGTH_MM and verify proportional movement)

---

## If It's Still Wrong

### Option A: Pen Doesn't Point Downward in Your Setup
If your pen points in a **different direction** (e.g., left, right, up), you need to account for **azimuth** θ:

```python
def pen_tip(mx, my, roll_deg, pitch_deg, l, pen_azimuth_deg=90.0):
    """pen_azimuth_deg: angle in image where pen points (0=right, 90=down, 180=left, 270=up)"""
    roll_rad = np.radians(roll_deg)
    pitch_rad = np.radians(pitch_deg)
    azi_rad = np.radians(pen_azimuth_deg)
    
    # Components along and perpendicular to pen direction
    delta_along = l * np.sin(pitch_rad)
    delta_perp = l * np.sin(roll_rad)
    
    # Decompose into image XY
    delta_x = delta_along * np.cos(azi_rad) - delta_perp * np.sin(azi_rad)
    delta_y = delta_along * np.sin(azi_rad) + delta_perp * np.cos(azi_rad)
    
    return float(mx + delta_x), float(my + delta_y)
```

Extract `pen_azimuth_deg` from the ArUco marker's rvec (3D orientation) if needed.

### Option B: Unexpected Sign Convention  
If test 1 or 2 fails, the simplest fix is **negate the offending angle**:

```python
# In pen_tip call, if roll-only test fails:
tip_x, tip_y = pen_tip(mx, my, -imu_roll_zeroed_deg, imu_pitch_zeroed_deg, pen_length_px)

# Or if pitch-only test fails:
tip_x, tip_y = pen_tip(mx, my, imu_roll_zeroed_deg, -imu_pitch_zeroed_deg, pen_length_px)
```

---

## Summary of Changes

| Aspect | Old | New | Reason |
|--------|-----|-----|--------|
| **Formula** | 3D perspective projection | 2D sin-based projection | Correct for small angles on horizontal surface |
| **Parameters** | phi, theta, d (complex) | roll_deg, pitch_deg | Clear geometric meaning |
| **Parameter order** | Swapped (phi=roll, theta=pitch) | Direct mapping | Eliminates confusion |
| **Sign handling** | Both rolls/pitches negated | Natural signs from geometry | Removes guesswork |
| **Function complexity** | ~8 lines, unclear logic | ~4 lines, obvious | Easier to debug/modify |

---

## Debugging Checklist

✓ Formula replaced with geometrically correct derivation  
✓ Function signature clarified (roll, pitch mapped correctly)  
✓ Sign constants removed (using direct geometry)  
✓ Calibration procedure provided (3 simple tests)  
✓ Backup plan if pen has different azimuth  
✓ Code is now patch-quality and ready to test

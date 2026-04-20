import msvcrt
from pyaxidraw import axidraw

STEP_CM = 1.0
MIN_X_CM = 0.0
MIN_Y_CM = 0.0
MAX_X_CM = 30.0
MAX_Y_CM = 21.0
START_X_CM = 0.0
START_Y_CM = 0.0

def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))

def main() -> None:
    ad = axidraw.AxiDraw()
    ad.interactive()

    # Raise actual runtime errors if USB drops
    ad.errors.connect = True
    ad.errors.disconnect = True

    # IMPORTANT: V3/A3 = model 2
    ad.options.model = 2

    # Set units before connect if possible
    ad.options.units = 1   # centimeters

    try:
        if not ad.connect():
            print("Failed to connect to AxiDraw.")
            return

        # If you changed options after connect, call update().
        # Safe to do here as well.
        ad.update()

        # Start only if machine is physically at home corner
        current_x = START_X_CM
        current_y = START_Y_CM
        ad.moveto(current_x, current_y)
        ad.block()

        print("Arrow keys move the pen-up position.")
        print(f"Step size: {STEP_CM:.2f} cm")
        print("Press 'q' to quit.")

        while True:
            key = msvcrt.getch()

            if key in (b"q", b"Q"):
                ad.goto(0, 0)  # move back to home position
                break

            if key not in (b"\x00", b"\xe0"):
                continue

            arrow = msvcrt.getch()
            next_x = current_x
            next_y = current_y

            if arrow == b"H":      # up
                next_y += STEP_CM
            elif arrow == b"P":    # down
                next_y -= STEP_CM
            elif arrow == b"K":    # left
                next_x -= STEP_CM
            elif arrow == b"M":    # right
                next_x += STEP_CM
            else:
                continue

            next_x = clamp(next_x, MIN_X_CM, MAX_X_CM)
            next_y = clamp(next_y, MIN_Y_CM, MAX_Y_CM)

            if next_x == current_x and next_y == current_y:
                continue

            ad.moveto(next_x, next_y)
            ad.block()  # wait until move is actually complete

            # Read back physical position from the API
            phys_x, phys_y = ad.current_pos()
            turtle_x, turtle_y = ad.turtle_pos()

            current_x = next_x
            current_y = next_y

            print(
                f"requested=({current_x:.2f}, {current_y:.2f}) cm | "
                f"physical=({phys_x:.2f}, {phys_y:.2f}) cm | "
                f"turtle=({turtle_x:.2f}, {turtle_y:.2f}) cm | "
                f"err={ad.errors.code}"
            )

    except RuntimeError as e:
        print(f"Runtime error: {e} | err={ad.errors.code}")
    finally:
        try:
            ad.disconnect()
            ad.goto(0, 0)  # move back to home position
        except Exception:
            pass

if __name__ == "__main__":
    main()
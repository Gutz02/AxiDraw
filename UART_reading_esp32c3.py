import serial
import time
import math

COM_PORT = 'COM11'
BAUD_RATE = 115200


class KalmanAngle:
    def __init__(self, q_angle: float = 0.001, q_bias: float = 0.003, r_measure: float = 0.03):
        self.q_angle = q_angle
        self.q_bias = q_bias
        self.r_measure = r_measure

        self.angle = 0.0
        self.bias = 0.0
        self.rate = 0.0
        self.p = [[0.0, 0.0], [0.0, 0.0]]

    def set_angle(self, angle: float):
        self.angle = angle

    def update(self, measured_angle: float, measured_rate: float, dt: float) -> float:
        self.rate = measured_rate - self.bias
        self.angle += dt * self.rate

        self.p[0][0] += dt * (dt * self.p[1][1] - self.p[0][1] - self.p[1][0] + self.q_angle)
        self.p[0][1] -= dt * self.p[1][1]
        self.p[1][0] -= dt * self.p[1][1]
        self.p[1][1] += self.q_bias * dt

        s = self.p[0][0] + self.r_measure
        k0 = self.p[0][0] / s
        k1 = self.p[1][0] / s

        innovation = measured_angle - self.angle
        self.angle += k0 * innovation
        self.bias += k1 * innovation

        p00 = self.p[0][0]
        p01 = self.p[0][1]

        self.p[0][0] -= k0 * p00
        self.p[0][1] -= k0 * p01
        self.p[1][0] -= k1 * p00
        self.p[1][1] -= k1 * p01

        return self.angle

def connection():
    return serial.Serial(COM_PORT, BAUD_RATE, timeout=0.05)

def average_frequency(ser: serial.Serial, N: int = 50):
    prev_time = None
    intervals = []

    N = 50

    while True:
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if not line:
                continue

            now = time.time()

            if prev_time is not None:
                dt = now - prev_time
                intervals.append(dt)

            prev_time = now

            if len(intervals) == N:
                avg_dt = sum(intervals) / N
                freq = 1.0 / avg_dt

                print(f"AVG dt: {avg_dt*1000:.2f} ms | AVG freq: {freq:.2f} Hz")

                intervals.clear()

        except KeyboardInterrupt:
            break

    ser.close()

def read_values(ser: serial.Serial):
    try:
        line = ser.readline().decode(errors='ignore').strip()
        return line
    except KeyboardInterrupt:
        return None
    
def main():
    gyro_roll = 0.0
    gyro_pitch = 0.0
    accel_roll = 0.0
    accel_pitch = 0.0
    kalman_roll = KalmanAngle()
    kalman_pitch = KalmanAngle()
    kalman_roll_angle = 0.0
    kalman_pitch_angle = 0.0
    filters_initialized = False
    prev_time = None
    ser = connection()
    while True:
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if not line:
                continue

            values = line.split()
            if len(values) < 5:
                continue

            #Lines are as follows: ax, ay, az, gx, gy
            ax = float(values[0])
            ay = float(values[1])
            az = float(values[2])
            gx = float(values[3])
            gy = float(values[4])

            now = time.perf_counter()
            dt = 0.0 if prev_time is None else now - prev_time
            prev_time = now

            gyro_roll  += gx * dt
            gyro_pitch += gy * dt

            accel_roll  = math.atan2(ay, az)
            accel_pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az))

            if not filters_initialized:
                kalman_roll.set_angle(accel_roll)
                kalman_pitch.set_angle(accel_pitch)
                kalman_roll_angle = accel_roll
                kalman_pitch_angle = accel_pitch
                filters_initialized = True
            elif dt > 0.0:
                kalman_roll_angle = kalman_roll.update(accel_roll, gx, dt)
                kalman_pitch_angle = kalman_pitch.update(accel_pitch, gy, dt)

            print(
                # f"Gyro Roll: {math.degrees(gyro_roll):.2f} | "
                # f"Gyro Pitch: {math.degrees(gyro_pitch):.2f} | "
                # f"Accel Roll: {math.degrees(accel_roll):.2f} | "
                # f"Accel Pitch: {math.degrees(accel_pitch):.2f} | "
                f"Kalman Roll: {math.degrees(kalman_roll_angle):.2f} | "
                f"Kalman Pitch: {math.degrees(kalman_pitch_angle):.2f}"
            )
        except ValueError:
            continue
        except KeyboardInterrupt:
            print("Interrupted")
            break 
        

if __name__ == "__main__":
    main()

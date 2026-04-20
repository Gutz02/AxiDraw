import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt, QRect
from PySide6.QtGui import QCloseEvent, QImage, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget
from pyaxidraw import axidraw


PAPER_WIDTH_CM = 42.0
PAPER_HEIGHT_CM = 29.0
FLIP_X = False
FLIP_Y = True
DISPLAY_INDEX = 0

PEN_TIP_THRESHOLD = 0.01
PROCESS_INTERVAL_MS = 5
MIN_POINT_DISTANCE_CM = 0.0
WINDOW_NAME = "Tablet Drawing"

USE_MOVING_AVERAGE = False
USE_EXPONENTIAL_SMOOTHING = False
MOVING_AVG_WINDOW = 5
EXP_ALPHA = 0.25


@dataclass
class PlotCommand:
    kind: str
    x_cm: float | None = None
    y_cm: float | None = None
    pen_down: bool | None = None


@dataclass
class TabletSample:
    x_px: float = 0.0
    y_px: float = 0.0
    pressure: float = 0.0
    tip_down: bool = False
    inside_paper: bool = False
    seen: bool = False


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def compute_paper_rect(display_width: int, display_height: int) -> tuple[int, int, int, int]:
    paper_aspect = PAPER_WIDTH_CM / PAPER_HEIGHT_CM
    paper_width_px = display_width
    paper_height_px = int(round(paper_width_px / paper_aspect))
    if paper_height_px > display_height:
        paper_height_px = display_height
        paper_width_px = int(round(paper_height_px * paper_aspect))
    paper_left_px = (display_width - paper_width_px) // 2
    paper_top_px = (display_height - paper_height_px) // 2
    return paper_left_px, paper_top_px, paper_width_px, paper_height_px


def map_widget_to_axidraw(
    x_px: float,
    y_px: float,
    paper_left_px: int,
    paper_top_px: int,
    paper_width_px: int,
    paper_height_px: int,
) -> tuple[float, float]:
    normalized_x = clamp((x_px - paper_left_px) / paper_width_px, 0.0, 1.0)
    normalized_y = clamp((y_px - paper_top_px) / paper_height_px, 0.0, 1.0)

    if FLIP_X:
        normalized_x = 1.0 - normalized_x
    if FLIP_Y:
        normalized_y = 1.0 - normalized_y

    ax_x = normalized_x * PAPER_WIDTH_CM
    ax_y = (1.0 - normalized_y) * PAPER_HEIGHT_CM
    return ax_x, ax_y


def map_axidraw_to_canvas(
    x_cm: float,
    y_cm: float,
    canvas_width_px: int,
    canvas_height_px: int,
) -> tuple[int, int]:
    normalized_x = x_cm / PAPER_WIDTH_CM
    normalized_y = 1.0 - (y_cm / PAPER_HEIGHT_CM)

    if FLIP_X:
        normalized_x = 1.0 - normalized_x
    if FLIP_Y:
        normalized_y = 1.0 - normalized_y

    canvas_x = int(round(normalized_x * (canvas_width_px - 1)))
    canvas_y = int(round(normalized_y * (canvas_height_px - 1)))
    canvas_x = int(clamp(canvas_x, 0, canvas_width_px - 1))
    canvas_y = int(clamp(canvas_y, 0, canvas_height_px - 1))
    return canvas_x, canvas_y


def move_or_draw_segment(
    ad: axidraw.AxiDraw,
    pen_down: bool,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> tuple[float, float]:
    if pen_down and not ad.current_pen():
        ad.lineto(x1, y1)
    else:
        ad.moveto(x1, y1)
    return x1, y1


def axidraw_worker(
    ad: axidraw.AxiDraw,
    command_queue: deque[PlotCommand],
    queue_condition: threading.Condition,
    stop_event: threading.Event,
    machine_state: dict[str, float | bool],
    machine_state_lock: threading.Lock,
) -> None:
    current_x_cm = 0.0
    current_y_cm = 0.0
    current_pen_down = False

    while True:
        with queue_condition:
            while not command_queue and not stop_event.is_set():
                queue_condition.wait(timeout=0.05)
            if not command_queue and stop_event.is_set():
                break
            command = command_queue.popleft()

        if command.kind == "pen":
            next_pen_down = bool(command.pen_down)
            if next_pen_down != current_pen_down:
                if next_pen_down:
                    ad.pendown()
                else:
                    ad.penup()
                current_pen_down = next_pen_down
                with machine_state_lock:
                    machine_state["pen_down"] = current_pen_down
            continue

        if command.kind != "move" or command.x_cm is None or command.y_cm is None:
            continue

        current_x_cm, current_y_cm = move_or_draw_segment(
            ad,
            current_pen_down,
            current_x_cm,
            current_y_cm,
            command.x_cm,
            command.y_cm,
        )
        with machine_state_lock:
            machine_state["x_cm"] = current_x_cm
            machine_state["y_cm"] = current_y_cm


class TabletDrawingWidget(QWidget):
    def __init__(
        self,
        display_geometry: QRect,
        command_queue: deque[PlotCommand],
        queue_condition: threading.Condition,
        machine_state: dict[str, float | bool],
        machine_state_lock: threading.Lock,
        stop_callback,
    ) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_NAME)
        self.setGeometry(display_geometry)
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.display_width = display_geometry.width()
        self.display_height = display_geometry.height()
        self.paper_left_px, self.paper_top_px, self.paper_width_px, self.paper_height_px = compute_paper_rect(
            self.display_width,
            self.display_height,
        )
        self.canvas = np.full((self.paper_height_px, self.paper_width_px, 3), 255, dtype=np.uint8)

        self.command_queue = command_queue
        self.queue_condition = queue_condition
        self.machine_state = machine_state
        self.machine_state_lock = machine_state_lock
        self.stop_callback = stop_callback

        self.tablet_lock = threading.Lock()
        self.latest_sample = TabletSample()
        self.last_pen_state = False
        self.last_buffered_x_cm = 0.0
        self.last_buffered_y_cm = 0.0
        self.prev_canvas_point: tuple[int, int] | None = None
        self.point_buffer: deque[tuple[float, float]] = deque(maxlen=MOVING_AVG_WINDOW)
        self.exp_initialized = False
        self.exp_x = 0.0
        self.exp_y = 0.0
        self.cursor_canvas_point: tuple[int, int] | None = None
        self.current_pressure = 0.0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_tablet_state)
        self.timer.start(PROCESS_INTERVAL_MS)

    def tabletEvent(self, event) -> None:
        pos = event.position()
        x_px = float(pos.x())
        y_px = float(pos.y())
        inside_paper = (
            self.paper_left_px <= x_px < self.paper_left_px + self.paper_width_px
            and self.paper_top_px <= y_px < self.paper_top_px + self.paper_height_px
        )
        sample = TabletSample(
            x_px=x_px,
            y_px=y_px,
            pressure=float(event.pressure()),
            tip_down=bool(event.pressure() > PEN_TIP_THRESHOLD),
            inside_paper=inside_paper,
            seen=True,
        )
        with self.tablet_lock:
            self.latest_sample = sample
        event.accept()
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.stop_callback()
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.stop_callback()
        event.accept()

    def apply_selected_smoothing(self, x: float, y: float) -> tuple[float, float]:
        if USE_MOVING_AVERAGE and USE_EXPONENTIAL_SMOOTHING:
            raise RuntimeError(
                "Choose only one smoothing filter: moving average OR exponential smoothing."
            )

        if USE_MOVING_AVERAGE:
            self.point_buffer.append((x, y))
            sx = sum(px for px, _ in self.point_buffer) / len(self.point_buffer)
            sy = sum(py for _, py in self.point_buffer) / len(self.point_buffer)
            return sx, sy

        if USE_EXPONENTIAL_SMOOTHING:
            if not self.exp_initialized:
                self.exp_x = x
                self.exp_y = y
                self.exp_initialized = True
                return x, y
            self.exp_x = EXP_ALPHA * x + (1.0 - EXP_ALPHA) * self.exp_x
            self.exp_y = EXP_ALPHA * y + (1.0 - EXP_ALPHA) * self.exp_y
            return self.exp_x, self.exp_y

        return x, y

    def process_tablet_state(self) -> None:
        with self.tablet_lock:
            sample = TabletSample(
                x_px=self.latest_sample.x_px,
                y_px=self.latest_sample.y_px,
                pressure=self.latest_sample.pressure,
                tip_down=self.latest_sample.tip_down,
                inside_paper=self.latest_sample.inside_paper,
                seen=self.latest_sample.seen,
            )

        if not sample.seen:
            self.update()
            return

        self.current_pressure = sample.pressure

        if not sample.inside_paper:
            if self.last_pen_state:
                with self.queue_condition:
                    self.command_queue.append(PlotCommand(kind="pen", pen_down=False))
                    self.queue_condition.notify_all()
                self.last_pen_state = False
                self.prev_canvas_point = None
            self.cursor_canvas_point = None
            self.update()
            return

        target_x_cm, target_y_cm = map_widget_to_axidraw(
            sample.x_px,
            sample.y_px,
            self.paper_left_px,
            self.paper_top_px,
            self.paper_width_px,
            self.paper_height_px,
        )
        target_x_cm, target_y_cm = self.apply_selected_smoothing(target_x_cm, target_y_cm)
        self.cursor_canvas_point = map_axidraw_to_canvas(
            target_x_cm,
            target_y_cm,
            self.paper_width_px,
            self.paper_height_px,
        )

        if sample.tip_down != self.last_pen_state:
            with self.queue_condition:
                if sample.tip_down:
                    self.command_queue.append(
                        PlotCommand(kind="move", x_cm=target_x_cm, y_cm=target_y_cm)
                    )
                    self.command_queue.append(PlotCommand(kind="pen", pen_down=True))
                else:
                    self.command_queue.append(PlotCommand(kind="pen", pen_down=False))
                self.queue_condition.notify_all()
            if sample.tip_down:
                self.last_buffered_x_cm = target_x_cm
                self.last_buffered_y_cm = target_y_cm
            else:
                self.prev_canvas_point = None
            self.last_pen_state = sample.tip_down

        dx = target_x_cm - self.last_buffered_x_cm
        dy = target_y_cm - self.last_buffered_y_cm
        move_distance_cm = math.hypot(dx, dy)
        if move_distance_cm >= MIN_POINT_DISTANCE_CM:
            self.last_buffered_x_cm = target_x_cm
            self.last_buffered_y_cm = target_y_cm
            if sample.tip_down:
                with self.queue_condition:
                    self.command_queue.append(
                        PlotCommand(kind="move", x_cm=target_x_cm, y_cm=target_y_cm)
                    )
                    self.queue_condition.notify()

            if sample.tip_down and self.prev_canvas_point is not None:
                cv2.line(self.canvas, self.prev_canvas_point, self.cursor_canvas_point, (0, 0, 0), 2)
            elif not sample.tip_down:
                cv2.circle(self.canvas, self.cursor_canvas_point, 1, (220, 220, 220), -1)
            self.prev_canvas_point = self.cursor_canvas_point

        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.darkGray)

        paper_rect = QRect(
            self.paper_left_px,
            self.paper_top_px,
            self.paper_width_px,
            self.paper_height_px,
        )
        painter.fillRect(paper_rect, Qt.white)

        rgb_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
        image = QImage(
            rgb_canvas.data,
            self.paper_width_px,
            self.paper_height_px,
            rgb_canvas.strides[0],
            QImage.Format_RGB888,
        )
        painter.drawImage(paper_rect.topLeft(), image)

        painter.setPen(QPen(Qt.gray, 2))
        painter.drawRect(paper_rect)

        if self.cursor_canvas_point is not None:
            painter.setPen(Qt.NoPen)
            painter.setBrush(Qt.red)
            painter.drawEllipse(
                self.paper_left_px + self.cursor_canvas_point[0] - 4,
                self.paper_top_px + self.cursor_canvas_point[1] - 4,
                8,
                8,
            )

        with self.queue_condition:
            queue_len = len(self.command_queue)
        with self.machine_state_lock:
            actual_x_cm = float(self.machine_state["x_cm"])
            actual_y_cm = float(self.machine_state["y_cm"])
            actual_pen_down = bool(self.machine_state["pen_down"])

        painter.setPen(Qt.white)
        painter.drawText(
            20,
            30,
            (
                f"pressure={self.current_pressure:.3f} | "
                f"tip_down={self.last_pen_state} | "
                f"actual=({actual_x_cm:.2f}, {actual_y_cm:.2f}) cm | "
                f"queued={queue_len} | pen={actual_pen_down}"
            ),
        )


def main() -> None:
    app = QApplication(sys.argv)
    screens = app.screens()
    if DISPLAY_INDEX >= len(screens):
        print(f"Display index {DISPLAY_INDEX} not available. Found {len(screens)} screen(s).")
        return

    ad = axidraw.AxiDraw()
    ad.interactive()
    ad.errors.connect = True
    ad.errors.disconnect = True
    ad.options.model = 2
    ad.options.units = 1
    ad.options.speed_pendown = 100
    ad.options.speed_penup = 100
    ad.options.accel = 1000

    if not ad.connect():
        print("Failed to connect to AxiDraw.")
        return

    ad.update()
    ad.penup()
    ad.moveto(0.0, 0.0)
    ad.block()

    command_queue: deque[PlotCommand] = deque()
    queue_condition = threading.Condition()
    stop_event = threading.Event()
    machine_state_lock = threading.Lock()
    machine_state: dict[str, float | bool] = {
        "x_cm": 0.0,
        "y_cm": 0.0,
        "pen_down": False,
    }

    worker_thread = threading.Thread(
        target=axidraw_worker,
        args=(ad, command_queue, queue_condition, stop_event, machine_state, machine_state_lock),
        name="axidraw-worker",
        daemon=True,
    )
    worker_thread.start()

    stop_requested = {"value": False}

    def request_stop() -> None:
        stop_requested["value"] = True

    screen_geometry = screens[DISPLAY_INDEX].geometry()
    widget = TabletDrawingWidget(
        screen_geometry,
        command_queue,
        queue_condition,
        machine_state,
        machine_state_lock,
        request_stop,
    )
    widget.showFullScreen()
    widget.activateWindow()
    widget.setFocus()

    print("Tablet-only tracking active.")
    print("Q or Esc = quit")
    print(f"Using screen {DISPLAY_INDEX}: {screen_geometry}")

    exit_code = 0
    try:
        exit_code = app.exec()
    finally:
        with queue_condition:
            command_queue.clear()
            stop_event.set()
            queue_condition.notify_all()
        try:
            worker_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            ad.penup()
            ad.moveto(0, 0)
            ad.block()
            ad.disconnect()
        except Exception:
            pass

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

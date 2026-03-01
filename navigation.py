import os
os.getlogin = lambda: "nvidia"  # Fix for Docker: pal.products.qcar requires this
import signal
import time
from threading import Thread, Lock

import cv2
import numpy as np
import pathlib

from pal.products.qcar import QCar, QCarGPS, QCarCameras
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap

from qcar_detector import load_model, infer_on_frame

# ===================== Experiment Configuration =====================
tf = 6000
startDelay = 1
controllerUpdateRate = 100

# Speed controller (tuned for stability)
K_p = 0.15
K_i = 0.1

# Adaptive Speed Control
V_MAX = 0.8
V_MIN = 0.25
CURVATURE_THRESHOLD = 2.0
CURVATURE_LOOKAHEAD = 35

enableSteeringControl = True
nodeSequence = [10, 1, 13, 19, 17, 20, 22, 9, 10]

calibrate = False

# Camera / Display
CONTROLLER_UPDATE_RATE = controllerUpdateRate
CSI_WIDTH, CSI_HEIGHT = 820, 410
DISPLAY_WIDTH, DISPLAY_HEIGHT = 820, 410
PREFERRED_CAM = "LEFT"
FAILOVER_AFTER_SEC = 1.0

# ===================== Stop Targets =====================
STOP_TARGETS = [
    (0.125, 4.395),
    (-0.905, 0.800),
]
STOP_RADIUS = 0.20   # metres
STOP_DURATION = 3.0  # seconds to pause at each target

# ===================== Traffic Light Configuration =====================
# YOLO label names for traffic light states (adjust to match your model's labels)
RED_LIGHT_LABELS    = {"red_light", "red light", "stop_light_red"}
YELLOW_LIGHT_LABELS = {"yellow_light", "yellow light", "stop_light_yellow"}
GREEN_LIGHT_LABELS  = {"green_light", "green light", "stop_light_green"}

# Minimum detection confidence to act on a traffic light
TRAFFIC_LIGHT_CONF_THRESHOLD = 0.50

# How many consecutive frames a red/yellow must be seen before stopping
# (avoids false positives from a single bad detection)
RED_CONFIRM_FRAMES = 3

# Once we stop for a red, we won't move until green is seen for this many frames
GREEN_CONFIRM_FRAMES = 2

# ---- Relevant Traffic Light ROI ----
# Only traffic lights whose bounding-box centre falls inside this horizontal
# band (as a fraction of frame width) will be acted on.  Lights detected at
# the left or right periphery (e.g. a light at a cross-street visible to the
# side) are ignored.  Tune TL_ROI_X_MIN / TL_ROI_X_MAX to taste.
#
#   |<-- ignore -->|<------ act on ------>|<-- ignore -->|
#   0.0           TL_ROI_X_MIN       TL_ROI_X_MAX       1.0
#
TL_ROI_X_MIN = 0.25   # ignore lights whose centre is left  of 25 % of frame
TL_ROI_X_MAX = 0.75   # ignore lights whose centre is right of 75 % of frame
#
# Vertical band: ignore lights that are too low in the frame (likely very
# close / already passed) or too high (likely far background noise).
TL_ROI_Y_MIN = 0.10   # top    10 % of frame → ignore
TL_ROI_Y_MAX = 0.80   # bottom 20 % of frame → ignore

# ===================== Roadmap Setup =====================
if enableSteeringControl:
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(nodeSequence)
    initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
    calibrationPose = initialPose
else:
    initialPose = [0, 0, 0]


# ===================== Shared State =====================
class State:
    def __init__(self):
        self.kill = False
        self.lock = Lock()
        self.raw_frame = None
        self.latest_detections = []
        self.cam_name = "NONE"
        self.throttle = 0.0
        self.steering = 0.0
        self.speed = 0.0
        self.pose = (0.0, 0.0, 0.0)
        self.path_complete = False

        # ---- Traffic light shared state ----
        # "unknown" | "red" | "yellow" | "green"
        self.traffic_light_state = "unknown"

state = State()


def sig_handler(*args):
    state.kill = True

signal.signal(signal.SIGINT, sig_handler)


# ===================== Traffic Light Helper =====================
def classify_traffic_light(detections, frame_w=CSI_WIDTH, frame_h=CSI_HEIGHT):
    """
    Given a list of detections [(x1,y1,x2,y2,conf,cls_id,label), ...],
    return the highest-priority traffic light state seen above threshold
    AND within the forward-facing ROI.

    Priority: red > yellow > green > unknown

    ROI filter
    ----------
    Only detections whose bounding-box *centre* falls in the normalised
    rectangle [TL_ROI_X_MIN, TL_ROI_X_MAX] x [TL_ROI_Y_MIN, TL_ROI_Y_MAX]
    are considered.  This rejects traffic lights visible at the sides of the
    frame (e.g. a cross-street light) so the car only obeys the light that
    is directly ahead of it.
    """
    best = "unknown"
    for det in detections:
        x1, y1, x2, y2, conf, _, label = det
        label_lower = label.lower().strip()

        # 1. Confidence gate
        if conf < TRAFFIC_LIGHT_CONF_THRESHOLD:
            continue

        # 2. Is it even a traffic-light class?
        is_tl = (label_lower in RED_LIGHT_LABELS or
                 label_lower in YELLOW_LIGHT_LABELS or
                 label_lower in GREEN_LIGHT_LABELS)
        if not is_tl:
            continue

        # 3. ROI gate — normalised bounding-box centre must be inside the
        #    forward zone.  Peripheral lights (left/right of car at an
        #    intersection) will have centres near the frame edges and are
        #    rejected here.
        cx_norm = ((x1 + x2) / 2.0) / frame_w
        cy_norm = ((y1 + y2) / 2.0) / frame_h

        if not (TL_ROI_X_MIN <= cx_norm <= TL_ROI_X_MAX and
                TL_ROI_Y_MIN <= cy_norm <= TL_ROI_Y_MAX):
            continue   # outside forward-facing zone — ignore

        # 4. Priority classification
        if label_lower in RED_LIGHT_LABELS:
            return "red"      # highest priority — short-circuit
        elif label_lower in YELLOW_LIGHT_LABELS and best != "red":
            best = "yellow"
        elif label_lower in GREEN_LIGHT_LABELS and best not in ("red", "yellow"):
            best = "green"

    return best


# ===================== Controllers =====================
class SpeedController:
    def __init__(self, kp=0, ki=0):
        self.maxThrottle = 0.3
        self.kp = kp
        self.ki = ki
        self.ei = 0

    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += dt * e
        self.ei = np.clip(self.ei, -1.0, 1.0)
        return np.clip(self.kp * e + self.ki * self.ei,
                       -self.maxThrottle, self.maxThrottle)


class PurePursuitController:
    WHEELBASE = 0.256

    def __init__(self, waypoints, cyclic=False):
        self.maxSteeringAngle = np.pi / 6
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.cyclic = cyclic
        self.finished = False
        self.p_ref = (0, 0)
        self.th_ref = 0

    def update(self, p_car, th, speed, dt):
        if self.finished:
            return 0.0

        wp_1 = self.wp[:, np.mod(self.wpi, self.N - 1)]
        wp_2 = self.wp[:, np.mod(self.wpi + 1, self.N - 1)]
        seg = wp_2 - wp_1
        seg_len = np.linalg.norm(seg)
        if seg_len > 1e-6:
            seg_uv = seg / seg_len
            s = np.dot(p_car - wp_1, seg_uv)
            if s >= seg_len:
                if self.cyclic or self.wpi < self.N - 2:
                    self.wpi += 1
                elif self.wpi >= self.N - 2:
                    self.finished = True
                    print(">> Pure Pursuit: Path complete!")
                    return 0.0

        self.th_ref = np.arctan2(seg[1], seg[0])
        Ld = max(0.25 + 0.6 * abs(speed), 0.25)

        goal = None
        for i in range(80):
            idx = np.mod(self.wpi + i, self.N - 1)
            wp = self.wp[:, idx]
            if np.linalg.norm(wp - p_car) >= Ld:
                goal = wp
                break

        if goal is None:
            goal = self.wp[:, np.mod(self.wpi + 1, self.N - 1)]

        self.p_ref = goal

        dx = goal[0] - p_car[0]
        dy = goal[1] - p_car[1]
        alpha = wrap_to_pi(np.arctan2(dy, dx) - th)

        Ld_actual = max(np.sqrt(dx**2 + dy**2), 0.05)
        delta = np.arctan2(2.0 * self.WHEELBASE * np.sin(alpha), Ld_actual)

        return np.clip(delta, -self.maxSteeringAngle, self.maxSteeringAngle)

    def is_finished(self):
        return self.finished

    def get_curvature_ahead(self, lookahead_points=35):
        max_curvature = 0.0
        for i in range(lookahead_points):
            idx0 = np.mod(self.wpi + i, self.N - 1)
            idx1 = np.mod(self.wpi + i + 1, self.N - 1)
            idx2 = np.mod(self.wpi + i + 2, self.N - 1)

            v1 = self.wp[:, idx1] - self.wp[:, idx0]
            v2 = self.wp[:, idx2] - self.wp[:, idx1]

            len1 = np.linalg.norm(v1)
            if len1 < 1e-6:
                continue

            heading1 = np.arctan2(v1[1], v1[0])
            heading2 = np.arctan2(v2[1], v2[0])

            angle_change = abs(wrap_to_pi(heading2 - heading1))
            curvature = angle_change / len1
            max_curvature = max(max_curvature, curvature)

        return max_curvature


# ===================== Camera Helpers =====================
def get_stream(cameras, name: str):
    name = name.upper()
    if name == "FRONT":
        return getattr(cameras, "csiFront", None)
    if name == "LEFT":
        return getattr(cameras, "csiLeft", None)
    if name == "RIGHT":
        return getattr(cameras, "csiRight", None)
    return None


def handle_keypress(k):
    if k is None or k == -1:
        return
    if k in (ord('q'), ord('Q'), 27):
        state.kill = True


# ===================== Vision Thread =====================
def yolo_worker(yolo_model):
    """
    Runs YOLO inference on every captured frame and writes detections +
    the inferred traffic-light state back into shared state.
    """
    print(">> Vision Thread Started.")

    red_counter   = 0
    green_counter = 0

    while not state.kill:
        with state.lock:
            frame = state.raw_frame.copy() if state.raw_frame is not None else None

        if frame is None:
            time.sleep(0.01)
            continue

        _, dets = infer_on_frame(yolo_model, frame)

        # ---- Classify traffic light from this frame ----
        raw_tl = classify_traffic_light(dets)

        # Debounce: require RED_CONFIRM_FRAMES consecutive detections before
        # declaring red/yellow, and GREEN_CONFIRM_FRAMES before clearing it.
        if raw_tl in ("red", "yellow"):
            red_counter   = min(red_counter + 1, RED_CONFIRM_FRAMES)
            green_counter = 0
        elif raw_tl == "green":
            green_counter = min(green_counter + 1, GREEN_CONFIRM_FRAMES)
            red_counter   = max(red_counter - 1, 0)
        else:
            # No light detected — decay counters gradually
            red_counter   = max(red_counter - 1, 0)
            green_counter = max(green_counter - 1, 0)

        if red_counter >= RED_CONFIRM_FRAMES:
            tl_state = raw_tl   # "red" or "yellow"
        elif green_counter >= GREEN_CONFIRM_FRAMES:
            tl_state = "green"
        else:
            tl_state = "unknown"

        with state.lock:
            state.latest_detections   = dets
            state.traffic_light_state = tl_state


# ===================== Main Integrated Loop =====================
def integratedLoop():
    cv2.namedWindow("QCar View - Pure Pursuit", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("QCar View - Pure Pursuit", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    # Controllers
    speedController = SpeedController(kp=K_p, ki=K_i)
    steeringController = None
    if enableSteeringControl:
        steeringController = PurePursuitController(
            waypoints=waypointSequence,
            cyclic=False
        )

    # Hardware
    qcar = QCar(readMode=0, frequency=controllerUpdateRate)

    cameras = QCarCameras(
        frameWidth=CSI_WIDTH,
        frameHeight=CSI_HEIGHT,
        frameRate=30,
        enableFront=True,
        enableLeft=True,
        enableRight=True,
    )

    if enableSteeringControl or calibrate:
        ekf = QCarEKF(x_0=initialPose)
        gps = QCarGPS(initialPose=initialPose, calibrate=calibrate)
    else:
        gps = memoryview(b'')

    preferred_name = PREFERRED_CAM.upper()
    active_name = preferred_name
    last_good_frame_time = time.time()

    u = 0.0
    delta = 0.0

    # ---- Stop-point state ----
    stop_stage = 0
    paused = False
    pause_start = 0.0

    # ---- Traffic light stop state ----
    # Tracks whether we are currently halted due to a red/yellow light
    tl_halted = False

    with qcar, cameras, gps:
        t0 = time.time()
        t = 0.0

        while (t < tf + startDelay) and (not state.kill) and (not state.path_complete):
            tp = t
            t = time.time() - t0
            dt = max(t - tp, 1e-6)

            # ----------- Read sensors -----------
            qcar.read()
            v = float(qcar.motorTach)

            # ----------- State estimation -----------
            if enableSteeringControl:
                if gps.readGPS():
                    y_gps = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
                    ekf.update([qcar.motorTach, delta], dt, y_gps, qcar.gyroscope[2])
                else:
                    ekf.update([qcar.motorTach, delta], dt, None, qcar.gyroscope[2])

                x = float(ekf.x_hat[0, 0])
                y = float(ekf.x_hat[1, 0])
                th = float(ekf.x_hat[2, 0])
                p = np.array([x, y])
            else:
                x = y = th = 0.0
                p = np.array([0.0, 0.0])

            # ----------- Read traffic light state (thread-safe) -----------
            with state.lock:
                tl_state = state.traffic_light_state

            # ----------- Traffic Light Logic -----------
            # We halt for red or yellow; we only resume when green is confirmed.
            # tl_halted persists across frames so the car stays stopped until
            # the light actually turns green (avoids reacting to momentary
            # "unknown" gaps between red and green detections).
            if tl_state in ("red", "yellow"):
                if not tl_halted:
                    tl_halted = True
                    speedController.ei = 0   # reset integral to avoid windup
                    print(f">> Traffic light: {tl_state.upper()} — stopping car")
            elif tl_state == "green" and tl_halted:
                tl_halted = False
                print(">> Traffic light: GREEN — resuming")
            # "unknown" while halted: stay halted (fail-safe behaviour)

            # ----------- Stop-point logic -----------
            if stop_stage < len(STOP_TARGETS) and not paused and not tl_halted:
                target_x, target_y = STOP_TARGETS[stop_stage]
                dist_to_stop = np.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)

                if dist_to_stop < STOP_RADIUS:
                    paused = True
                    pause_start = t
                    u = 0.0
                    delta = 0.0
                    speedController.ei = 0
                    qcar.write(u, delta)
                    print(f">> Stopped at target {stop_stage + 1}: "
                          f"({target_x}, {target_y}) — waiting {STOP_DURATION:.0f}s")

            if paused:
                if (t - pause_start) >= STOP_DURATION:
                    paused = False
                    stop_stage += 1
                    print(f">> Resuming — stop_stage now {stop_stage}")
                else:
                    qcar.write(0.0, 0.0)

            # ----------- Control update -----------
            # Car is stationary if: in start delay, paused at stop-point, or
            # halted at a red/yellow light.
            blocked = paused or tl_halted

            if not blocked:
                if t < startDelay:
                    u = 0.0
                    delta = 0.0
                else:
                    if enableSteeringControl:
                        delta = float(steeringController.update(p, th, v, dt))

                        curvature = steeringController.get_curvature_ahead(CURVATURE_LOOKAHEAD)
                        curvature_factor = min(curvature / CURVATURE_THRESHOLD, 1.0)
                        predictive_v = V_MAX - (V_MAX - V_MIN) * curvature_factor

                        steering_factor = abs(delta) / (np.pi / 6)
                        reactive_v = V_MAX - (V_MAX - V_MIN) * steering_factor

                        adaptive_v_ref = max(min(predictive_v, reactive_v), V_MIN)
                        u = float(speedController.update(v, adaptive_v_ref, dt))

                        if steeringController.is_finished():
                            print(">> PATH COMPLETE - Stopping car")
                            state.path_complete = True
                            u = 0.0
                            delta = 0.0
                    else:
                        delta = 0.0
                        u = float(speedController.update(v, V_MAX, dt))

                qcar.write(u, delta)
            else:
                # Ensure the car is fully stopped while blocked
                u = 0.0
                # Hold steering angle to avoid drifting but do not advance
                qcar.write(0.0, delta)

            # ----------- Camera capture -----------
            img = None
            active_stream = get_stream(cameras, active_name)
            if active_stream is not None and active_stream.read():
                img = active_stream.imageData
                last_good_frame_time = time.time()

            if img is None and (time.time() - last_good_frame_time) > FAILOVER_AFTER_SEC:
                for candidate in ["FRONT", "LEFT", "RIGHT"]:
                    stream = get_stream(cameras, candidate)
                    if stream is not None and stream.read():
                        img = stream.imageData
                        active_name = candidate
                        last_good_frame_time = time.time()
                        break

            if img is not None:
                with state.lock:
                    state.raw_frame = img.copy()
                    state.cam_name = active_name

            # ----------- Display -----------
            with state.lock:
                display_img = state.raw_frame.copy() if state.raw_frame is not None else None
                dets = list(state.latest_detections)
                tl_disp = state.traffic_light_state
                state.throttle = u
                state.steering = delta
                state.speed = v
                state.pose = (x, y, th)
                cam_name_disp = state.cam_name

            if display_img is None:
                display_img = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

            # Draw detections
            for det in dets:
                x1, y1, x2, y2, conf, cls_id, label = det
                # Colour-code bounding box based on traffic light class
                label_lower = label.lower().strip()
                if label_lower in RED_LIGHT_LABELS:
                    box_color = (0, 0, 255)      # red
                elif label_lower in YELLOW_LIGHT_LABELS:
                    box_color = (0, 215, 255)    # yellow (BGR)
                elif label_lower in GREEN_LIGHT_LABELS:
                    box_color = (0, 255, 0)      # green
                else:
                    box_color = (255, 0, 0)      # blue for other objects

                # Dim detections that fall outside the forward ROI
                cx_norm = ((x1 + x2) / 2.0) / DISPLAY_WIDTH
                cy_norm = ((y1 + y2) / 2.0) / DISPLAY_HEIGHT
                in_roi = (TL_ROI_X_MIN <= cx_norm <= TL_ROI_X_MAX and
                          TL_ROI_Y_MIN <= cy_norm <= TL_ROI_Y_MAX)
                thickness = 2 if in_roi else 1
                alpha_label = label if in_roi else f"[ignored] {label}"

                cv2.rectangle(display_img, (x1, y1), (x2, y2), box_color, thickness)
                label_text = f"{alpha_label} {conf:.2f}"
                text_y = y1 - 10 if y1 > 20 else y1 + 20
                cv2.putText(display_img, label_text, (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, thickness)

            # Draw forward-facing ROI rectangle (dashed appearance via two rects)
            roi_x1 = int(TL_ROI_X_MIN * DISPLAY_WIDTH)
            roi_y1 = int(TL_ROI_Y_MIN * DISPLAY_HEIGHT)
            roi_x2 = int(TL_ROI_X_MAX * DISPLAY_WIDTH)
            roi_y2 = int(TL_ROI_Y_MAX * DISPLAY_HEIGHT)
            cv2.rectangle(display_img, (roi_x1, roi_y1), (roi_x2, roi_y2),
                          (200, 200, 0), 1)   # dim yellow border
            cv2.putText(display_img, "ROI", (roi_x1 + 4, roi_y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

            WHITE  = (255, 255, 255)
            CYAN   = (0, 255, 255)
            GREEN  = (0, 255, 0)
            YELLOW = (0, 255, 255)
            RED    = (0, 0, 255)

            cv2.putText(display_img, f"PURE PURSUIT | CAM: FRONT", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

            # ---- Traffic light HUD (only shown when a light is actively detected) ----
            if tl_disp != "unknown":
                tl_color_map = {
                    "red":    RED,
                    "yellow": YELLOW,
                    "green":  GREEN,
                }
                tl_hud_color = tl_color_map.get(tl_disp, WHITE)
                tl_hud_text  = f"TRAFFIC LIGHT: {tl_disp.upper()}"
                if tl_halted:
                    tl_hud_text += "  [HALTED]"
                cv2.putText(display_img, tl_hud_text, (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, tl_hud_color, 2)
            elif tl_halted:
                # Still halted but light temporarily lost — show minimal warning
                cv2.putText(display_img, "TRAFFIC LIGHT: HALTED", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

            # Show stop-point status with custom messages per target
            STOP_MESSAGES = [
                "Picking up passenger",   # target 1
                "Dropping off passenger", # target 2
            ]
            if paused and stop_stage < len(STOP_TARGETS):
                remaining = max(0.0, STOP_DURATION - (t - pause_start))
                msg = STOP_MESSAGES[stop_stage] if stop_stage < len(STOP_MESSAGES) else f"Stopped at target {stop_stage + 1}"
                cv2.putText(display_img,
                            f"{msg} — resuming in {remaining:.1f}s",
                            (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)

            if enableSteeringControl and steeringController:
                curvature = steeringController.get_curvature_ahead(CURVATURE_LOOKAHEAD)
                curvature_factor = min(curvature / CURVATURE_THRESHOLD, 1.0)
                adaptive_v_ref = max(V_MAX - (V_MAX - V_MIN) * curvature_factor, V_MIN)

                cv2.putText(display_img,
                            f"THR: {u:.2f}  STR: {np.degrees(delta):.1f}°  V: {v:.2f} m/s",
                            (20, DISPLAY_HEIGHT - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

                cv2.putText(display_img,
                            f"Target: {adaptive_v_ref:.2f} m/s | Curvature: {curvature:.2f}",
                            (20, DISPLAY_HEIGHT - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
            else:
                cv2.putText(display_img,
                            f"THR: {u:.2f}  STR: {np.degrees(delta):.1f}°  V: {v:.2f} m/s",
                            (20, DISPLAY_HEIGHT - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

            cv2.putText(display_img,
                        f"Objects: {len(dets)} | WP: {steeringController.wpi if steeringController else 0}",
                        (20, DISPLAY_HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)

            cv2.imshow("QCar View - Pure Pursuit", display_img)

            k = cv2.waitKeyEx(1)
            handle_keypress(k)

        # Safe stop
        qcar.read_write_std(throttle=0, steering=0)

    cv2.destroyAllWindows()


# ===================== Main =====================
if __name__ == "__main__":
    print("--- STARTING INTEGRATED QCAR SYSTEM (PURE PURSUIT + TRAFFIC LIGHTS) ---")
    pathlib.WindowsPath = pathlib.PosixPath

    print(">> Loading YOLOv5 Model...")
    model = load_model()

    if model is None:
        print("[WARNING] YOLO model not loaded. Vision features disabled.")
        vision_enabled = False
    else:
        print(">> Warming up model...")
        dummy = np.zeros((CSI_HEIGHT, CSI_WIDTH, 3), dtype=np.uint8)
        infer_on_frame(model, dummy)
        vision_enabled = True

    if vision_enabled:
        vision_thread = Thread(target=yolo_worker, args=(model,), daemon=True)
        vision_thread.start()
    else:
        print(">> Vision thread disabled")

    print(">> Pure Pursuit controller ready")
    print(f">> Following path: {nodeSequence}")
    print(f">> Adaptive Speed: V_MAX={V_MAX} m/s, V_MIN={V_MIN} m/s")
    print(f">> Stop targets: {STOP_TARGETS}")
    print(f">> Traffic light labels — RED: {RED_LIGHT_LABELS}")
    print(f"                          YELLOW: {YELLOW_LIGHT_LABELS}")
    print(f"                          GREEN: {GREEN_LIGHT_LABELS}")

    control_thread = Thread(target=integratedLoop)
    control_thread.start()

    control_thread.join()
    state.kill = True
    print("--- SYSTEM SHUTDOWN COMPLETE ---")
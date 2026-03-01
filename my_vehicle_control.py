# import os
# import signal
# import time
# from threading import Thread, Lock

# import cv2
# import numpy as np
# import pathlib

# from pal.products.qcar import QCar, QCarGPS, QCarCameras
# from pal.utilities.math import wrap_to_pi
# from hal.content.qcar_functions import QCarEKF
# from hal.products.mats import SDCSRoadMap

# from qcar_detector import load_model, infer_on_frame

# # ===================== Experiment Configuration =====================
# tf = 6000
# startDelay = 1
# controllerUpdateRate = 100          # Hz  → dt = 0.01 s
# VISION_UPDATE_RATE   = 20           # Hz  → camera + YOLO + display

# # Speed controller
# v_ref = 0.5
# K_p   = 0.2
# K_i   = 0.25

# # Steering controller (Stanley)
# enableSteeringControl = True
# K_stanley    = 0.28
# nodeSequence = [10, 4, 20, 10]

# # GPS / EKF
# calibrate = False

# # calibrationPose = [0, 0, -np.pi/2]
# calibrationPose = [0, 2, -np.pi/2]

# # Camera / Display
# CSI_WIDTH,     CSI_HEIGHT     = 820, 410
# DISPLAY_WIDTH, DISPLAY_HEIGHT = 820, 410
# PREFERRED_CAM      = "LEFT"      # "FRONT" | "LEFT" | "RIGHT"
# FAILOVER_AFTER_SEC = 1.0

# # ===================== Roadmap Setup =====================
# if enableSteeringControl:
#     roadmap         = SDCSRoadMap(leftHandTraffic=False)
#     waypointSequence = roadmap.generate_path(nodeSequence)
#     initialPose     = roadmap.get_node_pose(nodeSequence[0]).squeeze()
# else:
#     initialPose = [0, 0, 0]


# # ===================== Shared State =====================
# class State:
#     def __init__(self):
#         self.kill = False
#         self.lock = Lock()

#         # vision (written by vision thread, read by display)
#         self.raw_frame         = None
#         self.latest_detections = []
#         self.cam_name          = "NONE"

#         # telemetry (written by control thread, read by display)
#         self.throttle = 0.0
#         self.steering = 0.0
#         self.speed    = 0.0
#         self.pose     = (0.0, 0.0, 0.0)   # x, y, th

# state = State()


# def sig_handler(*args):
#     state.kill = True

# signal.signal(signal.SIGINT, sig_handler)


# # ===================== Controllers =====================
# class SpeedController:
#     """PI speed controller with back-calculation anti-windup."""

#     def __init__(self, kp=0.0, ki=0.0):
#         self.max_throttle = 0.3
#         self.kp = kp
#         self.ki = ki
#         self.ei = 0.0
#         # Anti-windup: back-calculation gain  (1/Tt ≈ 1/Ti = ki/kp)
#         self.kb = ki / kp if kp > 1e-9 else 0.0

#     def update(self, v: float, v_ref: float, dt: float) -> float:
#         e   = v_ref - v
#         raw = self.kp * e + self.ki * self.ei

#         # Saturate output
#         u_sat = np.clip(raw, -self.max_throttle, self.max_throttle)

#         # Back-calculation: integrator correction proportional to saturation error
#         self.ei += dt * (e + self.kb * (u_sat - raw))

#         return u_sat


# class SteeringController:
#     def __init__(self, waypoints, k=1, cyclic=True):
#         self.maxSteeringAngle = np.pi / 6
#         self.wp     = waypoints
#         self.N      = len(waypoints[0, :])
#         self.wpi    = 0
#         self.k      = k
#         self.cyclic = cyclic

#         self.p_ref  = (0, 0)
#         self.th_ref = 0

#     def update(self, p, th, speed):
#         wp_1 = self.wp[:, np.mod(self.wpi,     self.N - 1)]
#         wp_2 = self.wp[:, np.mod(self.wpi + 1, self.N - 1)]

#         v     = wp_2 - wp_1
#         v_mag = np.linalg.norm(v)
#         if v_mag < 1e-9:
#             return 0.0

#         v_uv    = v / v_mag
#         tangent = np.arctan2(v_uv[1], v_uv[0])

#         s = np.dot(p - wp_1, v_uv)
#         if s >= v_mag:
#             if self.cyclic or self.wpi < self.N - 2:
#                 self.wpi += 1

#         ep        = wp_1 + v_uv * s
#         ct        = ep - p
#         direction = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)

#         ect = np.linalg.norm(ct) * np.sign(direction)
#         psi = wrap_to_pi(tangent - th)

#         self.p_ref  = ep
#         self.th_ref = tangent

#         spd = max(abs(speed), 0.05)     # avoid divide-by-zero
#         return np.clip(
#             wrap_to_pi(psi + np.arctan2(self.k * ect, spd)),
#             -self.maxSteeringAngle,
#             self.maxSteeringAngle,
#         )


# # ===================== Camera Helpers =====================
# def get_stream(cameras, name: str):
#     name = name.upper()
#     if name == "FRONT": return getattr(cameras, "csiFront", None)
#     if name == "LEFT":  return getattr(cameras, "csiLeft",  None)
#     if name == "RIGHT": return getattr(cameras, "csiRight", None)
#     return None


# # ===================== Thread A: Control + EKF (fixed-rate) =====================
# def control_loop(qcar, cameras_ref, gps, ekf):
#     """
#     Runs at exactly `controllerUpdateRate` Hz using a deadline scheduler.
#     Never calls imshow / waitKey / camera-read — those live in the vision thread.
#     """
#     speedController   = SpeedController(kp=K_p, ki=K_i)
#     steeringController = (
#         SteeringController(waypoints=waypointSequence, k=K_stanley)
#         if enableSteeringControl else None
#     )

#     dt_target = 1.0 / controllerUpdateRate
#     u         = 0.0
#     delta     = 0.0
#     x = y = th = 0.0

#     t0        = time.perf_counter()
#     next_tick = t0

#     while not state.kill:
#         # ---- Fixed-rate sleep (deadline scheduler) ----
#         next_tick += dt_target
#         sleep_dur  = next_tick - time.perf_counter()
#         if sleep_dur > 0:
#             time.sleep(sleep_dur)
#         # If we overran, next_tick stays ahead so we self-correct

#         t  = time.perf_counter() - t0
#         dt = dt_target          # use nominal dt for controller math

#         if t > tf + startDelay:
#             break

#         # ---- Read sensors ----
#         qcar.read()

#         if enableSteeringControl:
#             if gps.readGPS():
#                 y_gps = np.array([
#                     gps.position[0],
#                     gps.position[1],
#                     gps.orientation[2],
#                 ])
#                 ekf.update([qcar.motorTach, delta], dt, y_gps, qcar.gyroscope[2])
#             else:
#                 ekf.update([qcar.motorTach, delta], dt, None, qcar.gyroscope[2])

#             x  = ekf.x_hat[0, 0]
#             y  = ekf.x_hat[1, 0]
#             th = ekf.x_hat[2, 0]
#             p  = (np.array([x, y])
#                   + np.array([np.cos(th), np.sin(th)]) * 0.2)

#         v = float(qcar.motorTach)

#         # ---- Update controllers ----
#         if t < startDelay:
#             u     = 0.0
#             delta = 0.0
#         else:
#             u = speedController.update(v, v_ref, dt)
#             delta = steeringController.update(p, th, v) if enableSteeringControl else 0.0

#         qcar.write(u, delta)

#         # ---- Publish telemetry for display thread ----
#         with state.lock:
#             state.throttle = u
#             state.steering = delta
#             state.speed    = v
#             state.pose     = (x, y, th)

#     # Safe stop
#     qcar.read_write_std(throttle=0, steering=0)
#     state.kill = True


# # ===================== Thread B: YOLO inference (background) =====================
# def yolo_worker(yolo_model):
#     print(">> Vision/YOLO Thread Started.")
#     while not state.kill:
#         with state.lock:
#             frame = state.raw_frame.copy() if state.raw_frame is not None else None

#         if frame is None:
#             time.sleep(0.01)
#             continue

#         _, dets = infer_on_frame(yolo_model, frame)

#         with state.lock:
#             state.latest_detections = dets


# # ===================== Thread C: Camera capture + Display =====================
# def vision_display_loop(cameras):
#     """
#     Runs at ~VISION_UPDATE_RATE Hz.
#     Handles camera reads, frame sharing, YOLO overlay, and imshow/waitKey.
#     Control loop is never blocked by anything here.
#     """
#     cv2.namedWindow("QCar View", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("QCar View", DISPLAY_WIDTH, DISPLAY_HEIGHT)

#     dt_vision         = 1.0 / VISION_UPDATE_RATE
#     preferred_name    = PREFERRED_CAM.upper()
#     active_name       = preferred_name
#     last_good_frame_t = time.time()
#     next_tick         = time.perf_counter()

#     while not state.kill:
#         # ---- Fixed-rate sleep ----
#         next_tick += dt_vision
#         sleep_dur  = next_tick - time.perf_counter()
#         if sleep_dur > 0:
#             time.sleep(sleep_dur)

#         # ---- Camera read + failover ----
#         img = None
#         active_stream = get_stream(cameras, active_name)
#         if active_stream is not None and active_stream.read():
#             img = active_stream.imageData
#             last_good_frame_t = time.time()

#         if img is None and (time.time() - last_good_frame_t) > FAILOVER_AFTER_SEC:
#             for candidate in ["FRONT", "LEFT", "RIGHT"]:
#                 stream = get_stream(cameras, candidate)
#                 if stream is not None and stream.read():
#                     img         = stream.imageData
#                     active_name = candidate
#                     last_good_frame_t = time.time()
#                     break

#         if img is not None:
#             with state.lock:
#                 state.raw_frame = img.copy()
#                 state.cam_name  = active_name

#         # ---- Build display frame ----
#         with state.lock:
#             display_img   = state.raw_frame.copy() if state.raw_frame is not None else None
#             dets          = list(state.latest_detections)
#             u             = state.throttle
#             delta         = state.steering
#             v             = state.speed
#             cam_name_disp = state.cam_name

#         if display_img is None:
#             display_img = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

#         # ---- Draw detections ----
#         for det in dets:
#             x1, y1, x2, y2, conf, cls_id, label = det
#             cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             label_text = f"{label} {conf:.2f}"
#             text_y = y1 - 10 if y1 > 20 else y1 + 20
#             cv2.putText(display_img, label_text, (x1, text_y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#         # ---- UI overlays ----
#         WHITE = (255, 255, 255)
#         CYAN  = (0, 255, 255)
#         cv2.putText(display_img,
#                     f"CAM: {cam_name_disp} (pref: {preferred_name})",
#                     (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
#         cv2.putText(display_img,
#                     f"THR: {u:.2f}  STR: {delta:.2f}  V: {v:.2f} m/s",
#                     (20, DISPLAY_HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
#         cv2.putText(display_img,
#                     f"Objects Detected: {len(dets)}",
#                     (20, DISPLAY_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)

#         cv2.imshow("QCar View", display_img)

#         k = cv2.waitKeyEx(1)
#         if k in (ord('q'), ord('Q'), 27):
#             state.kill = True

#     cv2.destroyAllWindows()


# # ===================== Main =====================
# if __name__ == "__main__":
#     print("--- STARTING INTEGRATED QCAR SYSTEM ---")
#     pathlib.WindowsPath = pathlib.PosixPath

#     print(">> Loading YOLOv5 Model...")
#     model = load_model()
#     if model is None:
#         print("[FATAL] Could not load model. Exiting.")
#         raise SystemExit(1)

#     print(">> Warming up model...")
#     dummy = np.zeros((CSI_HEIGHT, CSI_WIDTH, 3), dtype=np.uint8)
#     infer_on_frame(model, dummy)

#     # ---- Bring up hardware ----
#     qcar    = QCar(readMode=1, frequency=controllerUpdateRate)
#     cameras = QCarCameras(
#         frameWidth=CSI_WIDTH,
#         frameHeight=CSI_HEIGHT,
#         frameRate=30,
#         enableFront=True,
#         enableLeft=True,
#         enableRight=True,
#     )

#     if enableSteeringControl or calibrate:
#         ekf = QCarEKF(x_0=initialPose)
#         gps = QCarGPS(initialPose=calibrationPose, calibrate=calibrate)
#     else:
#         ekf = None
#         gps = memoryview(b'')

#     with qcar, cameras, gps:
#         # Thread 1 – YOLO (lowest priority, pure background)
#         t_yolo = Thread(target=yolo_worker, args=(model,), daemon=True)
#         t_yolo.start()

#         # Thread 2 – Camera capture + display (~20 Hz)
#         t_vision = Thread(target=vision_display_loop, args=(cameras,), daemon=True)
#         t_vision.start()

#         # Thread 3 (main-ish) – Control + EKF (100 Hz, fixed-rate)
#         t_control = Thread(target=control_loop, args=(qcar, cameras, gps, ekf))
#         t_control.start()

#         t_control.join()        # block until control finishes or kill
#         state.kill = True       # signal vision threads to stop
#         t_vision.join(timeout=2.0)

#     print("--- SYSTEM SHUTDOWN COMPLETE ---")

import os
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

# Speed controller
v_ref = 0.5
K_p = 0.2
K_i = 0.25

# Steering controller (Stanley)
enableSteeringControl = True
K_stanley = 0.28
nodeSequence = [10, 4, 20, 10]

# GPS / EKF
calibrate = False

# Define the calibration pose
# Calibration pose is either [0,0,-pi/2] or [0,2,-pi/2]
# Comment out the one that is not used
# calibrationPose = [0,0,-np.pi/2]
calibrationPose = [0, 2, -np.pi/2]

# Camera / Display
CONTROLLER_UPDATE_RATE = controllerUpdateRate
CSI_WIDTH, CSI_HEIGHT = 820, 410
DISPLAY_WIDTH, DISPLAY_HEIGHT = 820, 410
PREFERRED_CAM = "LEFT"          # "FRONT" or "LEFT" or "RIGHT"
FAILOVER_AFTER_SEC = 1.0        # if preferred cam fails, try others

# ===================== Roadmap Setup =====================
if enableSteeringControl:
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(nodeSequence)
    initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
else:
    initialPose = [0, 0, 0]


# ===================== Shared State =====================
class State:
    def __init__(self):
        self.kill = False
        self.lock = Lock()

        # vision
        self.raw_frame = None
        self.latest_detections = []
        self.cam_name = "NONE"

        # vehicle telemetry (for overlay)
        self.throttle = 0.0
        self.steering = 0.0
        self.speed = 0.0
        self.pose = (0.0, 0.0, 0.0)  # x, y, th

state = State()


def sig_handler(*args):
    state.kill = True

signal.signal(signal.SIGINT, sig_handler)


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
        output = np.clip(self.kp * e + self.ki * self.ei,
                         -self.maxThrottle, self.maxThrottle)
        # FIX 1: Anti-windup — undo integral accumulation when output is saturated
        if abs(output) >= self.maxThrottle:
            self.ei -= dt * e
        return output


class SteeringController:
    def __init__(self, waypoints, k=1, cyclic=True):
        self.maxSteeringAngle = np.pi / 6
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.k = k
        self.cyclic = cyclic

        self.p_ref = (0, 0)
        self.th_ref = 0

    def update(self, p, th, speed):
        wp_1 = self.wp[:, np.mod(self.wpi, self.N - 1)]
        wp_2 = self.wp[:, np.mod(self.wpi + 1, self.N - 1)]

        v = wp_2 - wp_1
        v_mag = np.linalg.norm(v)
        if v_mag < 1e-9:
            return 0.0

        v_uv = v / v_mag
        tangent = np.arctan2(v_uv[1], v_uv[0])

        s = np.dot(p - wp_1, v_uv)
        if s >= v_mag:
            if self.cyclic or self.wpi < self.N - 2:
                self.wpi += 1

        ep = wp_1 + v_uv * s
        ct = ep - p
        direction = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)

        ect = np.linalg.norm(ct) * np.sign(direction)
        psi = wrap_to_pi(tangent - th)

        self.p_ref = ep
        self.th_ref = tangent

        spd = max(abs(speed), 0.05)

        return np.clip(
            wrap_to_pi(psi + np.arctan2(self.k * ect, spd)),
            -self.maxSteeringAngle,
            self.maxSteeringAngle
        )


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
    print(">> Vision Thread Started.")
    while not state.kill:
        with state.lock:
            frame = state.raw_frame.copy() if state.raw_frame is not None else None

        if frame is None:
            time.sleep(0.01)
            continue

        _, dets = infer_on_frame(yolo_model, frame)

        with state.lock:
            state.latest_detections = dets


# ===================== FIX 3: Dedicated Camera Thread =====================
def camera_worker(cameras):
    """Captures frames in a dedicated thread so camera I/O never blocks the control loop."""
    print(">> Camera Thread Started.")
    preferred_name = PREFERRED_CAM.upper()
    active_name = preferred_name
    last_good_frame_time = time.time()

    while not state.kill:
        img = None
        active_stream = get_stream(cameras, active_name)
        if active_stream is not None and active_stream.read():
            img = active_stream.imageData
            last_good_frame_time = time.time()

        # Failover to another camera if preferred is silent
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

        # Yield briefly to avoid spinning hot
        time.sleep(0.001)


# ===================== Display Thread =====================
def display_worker():
    """Renders detections and telemetry overlays; keeps all cv2 calls off the control loop."""
    print(">> Display Thread Started.")
    cv2.namedWindow("QCar View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("QCar View", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    preferred_name = PREFERRED_CAM.upper()

    while not state.kill:
        with state.lock:
            display_img = state.raw_frame.copy() if state.raw_frame is not None else None
            dets = list(state.latest_detections)
            u = state.throttle
            delta = state.steering
            v = state.speed
            cam_name_disp = state.cam_name

        if display_img is None:
            display_img = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

        # Draw detections
        for det in dets:
            x1, y1, x2, y2, conf, cls_id, label = det
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label_text = f"{label} {conf:.2f}"
            text_y = y1 - 10 if y1 > 20 else y1 + 20
            cv2.putText(display_img, label_text, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        WHITE = (255, 255, 255)
        CYAN = (0, 255, 255)

        cv2.putText(display_img, f"CAM: {cam_name_disp} (pref: {preferred_name})", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.putText(display_img, f"THR: {u:.2f}  STR: {delta:.2f}  V: {v:.2f} m/s",
                    (20, DISPLAY_HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        cv2.putText(display_img, f"Objects Detected: {len(dets)}", (20, DISPLAY_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)

        cv2.imshow("QCar View", display_img)
        k = cv2.waitKeyEx(1)
        handle_keypress(k)

        time.sleep(0.033)  # ~30 fps display cap

    cv2.destroyAllWindows()


# ===================== Control Loop (lean — no camera/display work) =====================
def controlLoop(qcar, gps, ekf):
    global KILL_THREAD

    speedController = SpeedController(kp=K_p, ki=K_i)
    steeringController = None
    if enableSteeringControl:
        steeringController = SteeringController(waypoints=waypointSequence, k=K_stanley)

    u = 0.0
    delta = 0.0
    x = initialPose[0]
    y = initialPose[1]
    th = initialPose[2]
    p = np.array([x, y])

    t0 = time.time()
    t = 0.0

    while (t < tf + startDelay) and (not state.kill):
        tp = t
        t = time.time() - t0
        # FIX 2: Clamp dt — prevents integral windup from large stalls
        dt = np.clip(t - tp, 1e-6, 0.05)

        # ----------- Read sensors -----------
        qcar.read()
        if enableSteeringControl:
            if gps.readGPS():
                y_gps = np.array([
                    gps.position[0],
                    gps.position[1],
                    gps.orientation[2]
                ])
                ekf.update([qcar.motorTach, delta], dt, y_gps, qcar.gyroscope[2])
            else:
                ekf.update([qcar.motorTach, delta], dt, None, qcar.gyroscope[2])

            x = ekf.x_hat[0, 0]
            y = ekf.x_hat[1, 0]
            th = ekf.x_hat[2, 0]
            p = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2

        v = qcar.motorTach
        print(f"v={float(v):+.3f}")

        # ----------- Controllers -----------
        if t < startDelay:
            u = 0.0
            delta = 0.0
        else:
            u = speedController.update(v, v_ref, dt)
            if enableSteeringControl:
                delta = steeringController.update(p, th, v)
            else:
                delta = 0.0

        qcar.write(u, delta)

        # Share telemetry with display thread
        with state.lock:
            state.throttle = u
            state.steering = delta
            state.speed = v
            state.pose = (x, y, th)

    qcar.read_write_std(throttle=0, steering=0)


# ===================== Main =====================
if __name__ == "__main__":
    print("--- STARTING INTEGRATED QCAR SYSTEM ---")
    pathlib.WindowsPath = pathlib.PosixPath

    print(">> Loading YOLOv5 Model...")
    model = load_model()
    if model is None:
        print("[FATAL] Could not load model. Exiting.")
        raise SystemExit(1)

    print(">> Warming up model...")
    dummy = np.zeros((CSI_HEIGHT, CSI_WIDTH, 3), dtype=np.uint8)
    infer_on_frame(model, dummy)

    qcar = QCar(readMode=1, frequency=CONTROLLER_UPDATE_RATE)
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
        gps = QCarGPS(initialPose=calibrationPose, calibrate=calibrate)
    else:
        ekf = None
        gps = memoryview(b'')

    with qcar, cameras, gps:
        vision_thread = Thread(target=yolo_worker, args=(model,), daemon=True)
        camera_thread = Thread(target=camera_worker, args=(cameras,), daemon=True)
        display_thread = Thread(target=display_worker, daemon=True)
        control_thread = Thread(target=controlLoop, args=(qcar, gps, ekf))

        vision_thread.start()
        camera_thread.start()
        display_thread.start()
        control_thread.start()

        control_thread.join()
        state.kill = True

    print("--- SYSTEM SHUTDOWN COMPLETE ---")
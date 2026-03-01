import os
import signal
import numpy as np
from threading import Lock
from contextlib import contextmanager, nullcontext
import time
import cv2
import pyqtgraph as pg

# Patch os.getlogin() for container environments where no TTY is available
_original_getlogin = os.getlogin
def _safe_getlogin():
    try:
        return _original_getlogin()
    except OSError:
        return os.environ.get('USER', os.environ.get('LOGNAME', 'unknown'))
os.getlogin = _safe_getlogin

from pal.products.qcar import QCar, QCarGPS, QCarLidar, IS_PHYSICAL_QCAR
from pal.utilities.scope import MultiScope
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
import pal.resources.images as images
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2

from path_generation import build_graph, find_path, ROAD_LIST_RIGHT, ROAD_LIST_LEFT

# Experiment Configuration
tf = 150
startDelay = 0.1
controllerUpdateRate = 100

# Speed profile: fast on straights, slow at corners
V_MAX = 0.8                # Maximum speed on straights (m/s)
V_MIN = 0.25               # Minimum speed in tight corners (m/s)
CURVATURE_THRESHOLD = 2.0  # Curvature (rad/m) at which speed drops to V_MIN
CURVATURE_LOOKAHEAD = 35   # Waypoints to look ahead (brake earlier)
v_ref = V_MAX              # Default reference (overridden by curvature control)

K_p = 0.15
K_i = 0.3

enableSteeringControl = True

# LiDAR (disabled -- not functional in current QLabs environment)
ENABLE_LIDAR = False

#Build the graph
graph = build_graph()

#Find a path from node 15 to 13
# nodeSequence = find_path(graph, 10, 6)

# nodeSequence = [10, 12, 15, 4, 5, 6, 4, 15, 20, 21, 22, 20, 10]
nodeSequence = [10, 4, 20, 12, 15, 5, 6, 21, 22, 10]


targets = [
    [-1.949, 3.869],
    [0.844, -1.041],
    [2.273, 1.731],
    [0.699, 4.540],
    [0.002, -0.001]
]

initialPose = [-1.205, -0.83, -0.78]
calibrate = False
calibrationPose = [-1.205, -0.83, -0.78]

# LiDAR Configuration
lidar_num_measurements = 384  # Must match QCarLidar default (was 1000, which is invalid)
lidar_measurement_mode = 2
lidar_interpolation_mode = 0
lidar_save_file = 'lidar_data.npz'

# Initialize waypointSequence globally
if enableSteeringControl:
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(nodeSequence)
else:
    waypointSequence = np.array([[0], [0]])

class SpeedController:
    def __init__(self, kp=0, ki=0):
        self.maxThrottle = 0.6
        self.kp = kp
        self.ki = ki
        self.ei = 0

    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += dt * e
        self.ei = np.clip(self.ei, -1.0, 1.0)  # Anti-windup
        return np.clip(
            self.kp * e + self.ki * self.ei,
            -self.maxThrottle,
            self.maxThrottle
        )

class SteeringController:
    """Pure pursuit path-tracking controller.

    Inherently smooth -- computes a continuous arc toward a goal point
    on the path. No post-processing (rate limiting, EMA) needed.
    """

    WHEELBASE = 0.256  # QCar wheelbase (m)

    def __init__(self, waypoints, cyclic=False):
        self.maxSteeringAngle = np.pi / 6
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.cyclic = cyclic
        self.p_ref = (0, 0)
        self.th_ref = 0

    def update(self, p_car, th, speed, dt):
        """Compute steering angle using pure pursuit.

        Args:
            p_car: Actual car position [x, y] (rear axle)
            th:    Car heading (rad)
            speed: Current speed (m/s)
            dt:    Time step (s)

        Returns:
            Steering angle (rad), clipped to maxSteeringAngle.
        """
        # Advance waypoint index if car has passed the current segment
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

        # Store reference heading (tangent of current path segment)
        self.th_ref = np.arctan2(seg[1], seg[0])

        # Speed-dependent lookahead: close at low speed, far at high speed
        Ld = max(0.25 + 0.6 * abs(speed), 0.25)

        # Find goal point: first waypoint at distance >= Ld ahead on path
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

        # Angle from car heading to goal point
        dx = goal[0] - p_car[0]
        dy = goal[1] - p_car[1]
        alpha = wrap_to_pi(np.arctan2(dy, dx) - th)

        # Pure pursuit steering law
        Ld_actual = max(np.sqrt(dx**2 + dy**2), 0.05)
        delta = np.arctan2(2.0 * self.WHEELBASE * np.sin(alpha), Ld_actual)

        return np.clip(delta, -self.maxSteeringAngle, self.maxSteeringAngle)

    def get_curvature_ahead(self, lookahead_points=35):
        """Calculate max curvature over the next N waypoints (rad/m)."""
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

def sig_handler(*args):
    print("Received termination signal. Shutting down...")
    raise SystemExit

signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)

def controlLoop():
    u = 0
    delta = 0
    adaptive_v_ref = V_MAX
    countMax = controllerUpdateRate / 10
    count = 0
    last_lidar_time = 0
    lidar_fail_count = 0
    time_log = []
    delta_log = []
    x_log = []
    waypoint_x_log = []
    lidar_distance_log = []

    qlabs = QuanserInteractiveLabs()
    if not qlabs.open("localhost"):
        print("Failed to connect to QLabs.")
        return

    qcar_qlabs = QLabsQCar2(qlabs)
    qcar_qlabs.actorNumber = 0
    qcar_qlabs.set_led_strip_uniform(color=[0, 1, 0])

    speedController = SpeedController(kp=K_p, ki=K_i)
    steeringController = SteeringController(waypoints=waypointSequence, cyclic=True)
    qcar = QCar(readMode=0, frequency=controllerUpdateRate)
    qcar_lock = Lock()
    ekf = QCarEKF(x_0=initialPose)
    gps = QCarGPS(initialPose=calibrationPose, calibrate=calibrate)

    if ENABLE_LIDAR:
        lidar = QCarLidar(
            numMeasurements=lidar_num_measurements,
            rangingDistanceMode=lidar_measurement_mode,
            interpolationMode=lidar_interpolation_mode
        )
    else:
        lidar = nullcontext()

    angles_list = []
    distances_list = []
    timestamps = []

    try:
        with qcar, gps, lidar:
            t0 = time.time()
            t = 0
            stage = 0
            paused = False
            pause_start = 0
            last_led_color = [0, 1, 0]

            print(f"Starting control loop and LiDAR data collection for {tf:.1f} seconds...")

            while t < tf + startDelay:
                tp = t
                t = time.time() - t0
                dt = t - tp

                with qcar_lock:
                    qcar.read()
                    if gps.readGPS():
                        y_gps = np.array([
                            gps.position[0],
                            gps.position[1],
                            gps.orientation[2]
                        ])
                        ekf.update(
                            [qcar.motorTach, delta],
                            dt,
                            y_gps,
                            qcar.gyroscope[2],
                        )
                    else:
                        ekf.update(
                            [qcar.motorTach, delta],
                            dt,
                            None,
                            qcar.gyroscope[2],
                        )

                    x = ekf.x_hat[0,0]
                    y = ekf.x_hat[1,0]
                    th = ekf.x_hat[2,0]
                    v = qcar.motorTach
                    p = np.array([x, y])  # Actual position (pure pursuit handles its own lookahead)

                lidar_delta = None
                current_v_ref = v_ref
                if ENABLE_LIDAR and t >= startDelay and (t - last_lidar_time) >= 0.1:
                    success = lidar.read()
                    if success and lidar.angles is not None and lidar.distances is not None:
                        lidar_fail_count = 0
                        angles_list.append(lidar.angles.copy())
                        distances_list.append(lidar.distances.copy())
                        timestamps.append(t)
                        right_distances = lidar.distances[(lidar.angles >= -0.2) & (lidar.angles <= 0.2)]
                        right_distances = right_distances[right_distances > 0]
                        if len(right_distances) > 0:
                            min_distance = np.min(right_distances)
                            lidar_distance_log.append(min_distance)
                            if min_distance < 0.7:
                                lidar_delta = -np.pi / 6
                        else:
                            lidar_distance_log.append(np.inf)
                    else:
                        lidar_fail_count += 1
                        if lidar_fail_count <= 3 or lidar_fail_count % 50 == 0:
                            print(f"LiDAR read failed at t={t:.1f}s (failures: {lidar_fail_count})")
                        lidar_distance_log.append(np.inf)
                    last_lidar_time = t

                target = targets[stage]
                target_x, target_y = target
                distance_to_target = np.sqrt((x - target_x)**2 + (y - target_y)**2)

                if stage == 0:
                    led_color = [1, 0, 0]
                elif stage == 1:
                    led_color = [0, 0, 1]
                elif stage == 2:
                    led_color = [1, 0, 0]
                elif stage == 3:
                    led_color = [0, 0, 1]
                else:
                    led_color = [0, 1, 0]

                if not paused and distance_to_target < 0.1:
                    paused = True
                    pause_start = t
                    u = 0
                    delta = 0
                    speedController.ei = 0  # Reset integral on pause
                    with qcar_lock:
                        qcar.write(u, delta)
                    if led_color != last_led_color:
                        qcar_qlabs.set_led_strip_uniform(color=led_color)
                        last_led_color = led_color

                elif paused and (t - pause_start) >= 3:
                    paused = False
                    stage += 1
                    if stage < len(targets) and led_color != [0, 1, 0]:
                        qcar_qlabs.set_led_strip_uniform(color=[0, 1, 0])
                        last_led_color = [0, 1, 0]
                elif paused:
                    u = 0
                    delta = 0
                    with qcar_lock:
                        qcar.write(u, delta)
                    speedScope.refresh()
                    continue

                if not paused:
                    if t < startDelay:
                        u = 0
                        delta = 0
                    else:
                        # Pure pursuit steering (inherently smooth, no post-processing needed)
                        delta = steeringController.update(p, th, v, dt)

                        # Curvature-based speed: fast on straights, slow before corners
                        curvature = steeringController.get_curvature_ahead(CURVATURE_LOOKAHEAD)
                        curvature_factor = min(curvature / CURVATURE_THRESHOLD, 1.0)
                        predictive_v = V_MAX - (V_MAX - V_MIN) * curvature_factor

                        # Reactive safety: also slow down when actively steering hard
                        steering_factor = abs(delta) / (np.pi / 6)
                        reactive_v = V_MAX - (V_MAX - V_MIN) * steering_factor

                        adaptive_v_ref = max(min(predictive_v, reactive_v), V_MIN)
                        u = speedController.update(v, adaptive_v_ref, dt)

                with qcar_lock:
                    qcar.write(u, delta)

                # Log control inputs
                with open('control_log.txt', 'a') as f:
                    f.write(f"t={t:.2f}, Throttle={u:.4f}, Steering={delta:.4f}\n")

                time_log.append(t)
                delta_log.append(delta)
                x_log.append(x)
                waypoint_x_log.append(steeringController.p_ref[0])

                if stage == len(targets):
                    break

                count += 1
                if count >= countMax and t > startDelay:
                    t_plot = t - startDelay
                    speedScope.axes[0].sample(t_plot, [v, adaptive_v_ref])
                    speedScope.axes[1].sample(t_plot, [adaptive_v_ref - v])
                    speedScope.axes[2].sample(t_plot, [u])
                    steeringScope.axes[4].sample(t_plot, [[p[0], p[1]]])
                    p[0] = ekf.x_hat[0,0]
                    p[1] = ekf.x_hat[1,0]
                    x_ref = steeringController.p_ref[0]
                    y_ref = steeringController.p_ref[1]
                    th_ref = steeringController.th_ref
                    x_gps = gps.position[0]
                    y_gps = gps.position[1]
                    th_gps = gps.orientation[2]
                    steeringScope.axes[0].sample(t_plot, [p[0], x_gps])
                    steeringScope.axes[1].sample(t_plot, [p[1], y_gps])
                    steeringScope.axes[2].sample(t_plot, [th, th_gps])
                    steeringScope.axes[3].sample(t_plot, [delta])
                    arrow.setPos(p[0], p[1])
                    arrow.setStyle(angle=180 - th * 180 / np.pi)
                    count = 0

                # Refresh all scope plots (transfers buffered samples to
                # plot curves and processes Qt events internally)
                speedScope.refresh()

            with qcar_lock:
                qcar.read_write_std(throttle=0, steering=0)
                qcar_qlabs.set_led_strip_uniform(color=[0, 1, 0])

        if ENABLE_LIDAR and angles_list:
            try:
                np.savez(lidar_save_file, angles=angles_list, distances=distances_list, timestamps=timestamps)
                print(f"LiDAR data saved to '{lidar_save_file}'")
            except Exception as e:
                print(f"Failed to save LiDAR data: {repr(e)}")

        import matplotlib.pyplot as plt
        if delta_log and time_log:
            plt.figure(figsize=(8, 6))
            plt.plot(time_log, delta_log, 'r-', label='Steering Angle')
            plt.title('Steering Angle vs. Time')
            plt.xlabel('Time [s]')
            plt.ylabel('Steering Angle [rad]')
            plt.grid(True)
            plt.legend()
            plt.savefig('delta_plot.png')
            print("Delta plot saved to 'delta_plot.png'")

        if x_log and waypoint_x_log and time_log:
            plt.figure(figsize=(8, 6))
            plt.plot(time_log, x_log, 'b-', label='EKF x')
            plt.plot(time_log, waypoint_x_log, 'g--', label='Waypoint x')
            plt.title('EKF x vs. Waypoint x')
            plt.xlabel('Time [s]')
            plt.ylabel('x Position [m]')
            plt.grid(True)
            plt.legend()
            plt.savefig('x_vs_waypoint_plot.png')
            print("x vs. waypoint plot saved to 'x_vs_waypoint_plot.png'")

        if lidar_distance_log and time_log:
            plt.figure(figsize=(8, 6))
            plt.plot(time_log[:len(lidar_distance_log)], lidar_distance_log, 'm-', label='Min LiDAR Distance')
            plt.title('Minimum LiDAR Distance vs. Time')
            plt.xlabel('Time [s]')
            plt.ylabel('Distance [m]')
            plt.grid(True)
            plt.legend()
            plt.savefig('lidar_distance_plot.png')
            print("LiDAR distance plot saved to 'lidar_distance_plot.png'")

    except SystemExit:
        print("Program terminated by signal.")
    finally:
        cv2.destroyAllWindows()
        qlabs.close()
        print("Test complete.")

if __name__ == '__main__':
    if IS_PHYSICAL_QCAR:
        fps = 10
    else:
        fps = 30

    speedScope = MultiScope(
        rows=3,
        cols=1,
        title='Vehicle Speed Control',
        fps=fps
    )
    speedScope.addAxis(
        row=0,
        col=0,
        timeWindow=tf,
        yLabel='Vehicle Speed [m/s]',
        yLim=(0, 1.0)
    )
    speedScope.axes[0].attachSignal(name='v_meas', width=2)
    speedScope.axes[0].attachSignal(name='v_ref')

    speedScope.addAxis(
        row=1,
        col=0,
        timeWindow=tf,
        yLabel='Speed Error [m/s]',
        yLim=(-0.5, 0.5)
    )
    speedScope.axes[1].attachSignal()

    speedScope.addAxis(
        row=2,
        col=0,
        timeWindow=tf,
        xLabel='Time [s]',
        yLabel='Throttle Command [%]',
        yLim=(-0.5, 0.5)
    )
    speedScope.axes[2].attachSignal()

    steeringScope = MultiScope(
        rows=4,
        cols=2,
        title='Vehicle Steering Control',
        fps=fps
    )

    steeringScope.addAxis(
        row=0,
        col=0,
        timeWindow=tf,
        yLabel='x Position [m]',
        yLim=(-2.5, 2.5)
    )
    steeringScope.axes[0].attachSignal(name='x_meas')
    steeringScope.axes[0].attachSignal(name='x_ref')

    steeringScope.addAxis(
        row=1,
        col=0,
        timeWindow=tf,
        yLabel='y Position [m]',
        yLim=(-1, 5)
    )
    steeringScope.axes[1].attachSignal(name='y_meas')
    steeringScope.axes[1].attachSignal(name='y_ref')

    steeringScope.addAxis(
        row=2,
        col=0,
        timeWindow=tf,
        yLabel='Heading Angle [rad]',
        yLim=(-3.5, 3.5)
    )
    steeringScope.axes[2].attachSignal(name='th_meas')
    steeringScope.axes[2].attachSignal(name='th_ref')

    steeringScope.addAxis(
        row=3,
        col=0,
        timeWindow=tf,
        yLabel='Steering Angle [rad]',
        yLim=(-0.6, 0.6)
    )
    steeringScope.axes[3].attachSignal()
    steeringScope.axes[3].xLabel = 'Time [s]'

    steeringScope.addXYAxis(
        row=0,
        col=1,
        rowSpan=4,
        xLabel='x Position [m]',
        yLabel='y Position [m]',
        xLim=(-2.5, 2.5),
        yLim=(-1, 5)
    )

    im = cv2.imread(
        images.SDCS_CITYSCAPE,
        cv2.IMREAD_GRAYSCALE
    )

    steeringScope.axes[4].attachImage(
        scale=(-0.002035, 0.002035),
        offset=(1125, 2365),
        rotation=180,
        levels=(0, 255)
    )
    steeringScope.axes[4].images[0].setImage(image=im)

    referencePath = pg.PlotDataItem(
        pen={'color': (85, 168, 104), 'width': 2},
        name='Reference'
    )
    steeringScope.axes[4].plot.addItem(referencePath)
    referencePath.setData(waypointSequence[0, :], waypointSequence[1, :])

    steeringScope.axes[4].attachSignal(name='Estimated', width=2)

    arrow = pg.ArrowItem(
        angle=180,
        tipAngle=60,
        headLen=10,
        tailLen=10,
        tailWidth=5,
        pen={'color': 'w', 'fillColor': [196, 78, 82], 'width': 1},
        brush=[196, 78, 82]
    )
    arrow.setPos(initialPose[0], initialPose[1])
    steeringScope.axes[4].plot.addItem(arrow)

    try:
        controlLoop()
    except SystemExit:
        pass
    input('Experiment complete. Press any key to exit...')

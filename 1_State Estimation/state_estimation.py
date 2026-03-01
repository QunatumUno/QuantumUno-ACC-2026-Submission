import signal
import time
import numpy as np

from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.scope import MultiScope
from pal.utilities.math import wrap_to_pi

# ================= Experiment Configuration =================
tf = 10
controllerUpdateRate = 100

def sig_handler(*args):
    print("Received termination signal. Shutting down...")
    raise SystemExit

signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)
#endregion


# ---------------------------------------------------------------------------

class QcarEKF:
    def __init__(self, x0, P0, Q, R):
        self.L = 0.257
        self.I = np.eye(3)
        self.xHat = x0
        self.P = P0
        self.Q = Q
        self.R = R

        self.C = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

    def f(self, X, u, dt):
        v = float(u[0])
        delta = float(u[1])

        x = float(X[0, 0])
        y = float(X[1, 0])
        th = float(X[2, 0])

        x_next = x + v * np.cos(th) * dt
        y_next = y + v * np.sin(th) * dt
        th_next = th + (v / self.L) * np.tan(delta) * dt
        th_next = wrap_to_pi(th_next)

        return np.array([[x_next], [y_next], [th_next]])

    def Jf(self, X, u, dt):
        v = float(u[0])
        th = float(X[2, 0])

        F = np.eye(3)
        F[0, 2] = -v * np.sin(th) * dt
        F[1, 2] =  v * np.cos(th) * dt
        return F

    def prediction(self, dt, u):
        # Predict state
        self.xHat = self.f(self.xHat, u, dt)

        # Predict covariance
        F = self.Jf(self.xHat, u, dt)
        self.P = F @ self.P @ F.T + self.Q

    def correction(self, y):
        # Dead-reckoning EKF: skip correction
        if self.R is None:
            return

        y = np.array(y).reshape(3, 1)

        y_hat = self.C @ self.xHat
        r = y - y_hat
        r[2, 0] = wrap_to_pi(r[2, 0])

        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)

        self.xHat = self.xHat + K @ r
        self.xHat[2, 0] = wrap_to_pi(self.xHat[2, 0])

        self.P = (self.I - K @ self.C) @ self.P


class GyroKF:
    """
    Simple discrete KF for heading + gyro bias:
      state x = [theta, bias]^T

    model:
      theta_{k+1} = theta_k + dt*(u - bias_k)
      bias_{k+1}  = bias_k

    measurement (GPS heading):
      y = theta + noise
    """

    def __init__(self, x0, P0, Q, R):
        self.I = np.eye(2)
        self.xHat = x0
        self.P = P0
        self.Q = Q
        self.R = R

        # C maps [theta, bias] -> theta
        self.C = np.array([[1.0, 0.0]])

    def prediction(self, dt, u):
        u = float(u)

        theta = float(self.xHat[0, 0])
        bias = float(self.xHat[1, 0])

        # State prediction
        theta_next = theta + dt * (u - bias)
        theta_next = wrap_to_pi(theta_next)
        bias_next = bias

        self.xHat = np.array([[theta_next], [bias_next]])

        # Linearized A for this discrete model:
        # theta_next = theta - dt*bias + dt*u
        # bias_next  = bias
        A = np.array([
            [1.0, -dt],
            [0.0,  1.0]
        ])

        self.P = A @ self.P @ A.T + self.Q

    def correction(self, y):
        y = float(y)
        y = wrap_to_pi(y)

        # Innovation
        y_hat = float((self.C @ self.xHat)[0, 0])
        r = wrap_to_pi(y - y_hat)

        # Innovation covariance
        S = self.C @ self.P @ self.C.T + self.R

        # Kalman gain
        K = self.P @ self.C.T @ np.linalg.inv(S)

        # Update
        self.xHat = self.xHat + K * r
        self.xHat[0, 0] = wrap_to_pi(self.xHat[0, 0])

        self.P = (self.I - K @ self.C) @ self.P


def controlLoop():
    # used to limit data sampling to 10hz
    countMax = controllerUpdateRate / 10
    count = 0

    # ---------------- Estimators Setup ----------------
    x0 = np.zeros((3, 1))
    P0 = np.eye(3)

    ekf_dr = QcarEKF(
        x0=x0.copy(),
        P0=P0.copy(),
        Q=np.diagflat([0.01, 0.01, 0.01]),
        R=None
    )

    ekf_gps = QcarEKF(
        x0=x0.copy(),
        P0=P0.copy(),
        Q=np.diagflat([0.01, 0.01, 0.01]),
        R=np.diagflat([0.2, 0.2, 0.1])
    )

    kf = GyroKF(
        x0=np.zeros((2, 1)),
        P0=np.eye(2),
        Q=np.diagflat([0.00001, 0.00001]),
        R=np.diagflat([0.1])
    )

    C_combined = np.eye(3)
    R_combined = np.diagflat([0.8, 0.8, 0.01])

    ekf_sf = QcarEKF(
        x0=x0.copy(),
        P0=P0.copy(),
        Q=np.diagflat([0.0001, 0.0001, 0.0001]),
        R=R_combined
    )

    # ---------------- Main Loop ----------------
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    gps = QCarGPS(initialPose=x0[:, 0])

    with qcar, gps:
        t0 = time.time()
        t = 0.0

        while t < tf:
            tp = t
            t = time.time() - t0
            dt = t - tp
            if dt <= 0:
                continue

            # Read sensors
            qcar.read()
            speed_tach = qcar.motorTach
            th_gyro = qcar.gyroscope[2]

            # Simple constant commands (as in your original)
            u = 0.1
            delta = np.pi / 12
            qcar.write(u, delta)

            # --------- Choose which estimators run ---------
            ekf_dr.prediction(dt, [speed_tach, delta])

            # Uncomment to enable:
            ekf_gps.prediction(dt, [speed_tach, delta])
            kf.prediction(dt, th_gyro)
            ekf_sf.prediction(dt, [speed_tach, delta])

            # --------- Corrections ---------
            if gps.readGPS():
                x_gps = gps.position[0]
                y_gps = gps.position[1]
                th_gps = gps.orientation[2]
                y = np.array([[x_gps], [y_gps], [th_gps]])

                # Estimator 2 correction
                ekf_gps.correction(y)

                # Estimator 3 example wiring (only if you enable kf/ekf_sf above):
                kf.correction(th_gps)
                ekf_sf.C = C_combined
                ekf_sf.R = R_combined
                ekf_sf.correction([[x_gps], [y_gps], [kf.xHat[0,0]]])

            # --------- Scopes ---------
            count += 1
            if count >= countMax:
                scope.axes[0].sample(t, [
                    ekf_dr.xHat[0, 0],
                    ekf_gps.xHat[0, 0],
                    ekf_sf.xHat[0, 0]
                ])
                scope.axes[1].sample(t, [
                    ekf_dr.xHat[1, 0],
                    ekf_gps.xHat[1, 0],
                    ekf_sf.xHat[1, 0]
                ])
                scope.axes[2].sample(t, [
                    ekf_dr.xHat[2, 0],
                    ekf_gps.xHat[2, 0],
                    ekf_sf.xHat[2, 0]
                ])
                scope.axes[3].sample(t, [
                    [ekf_dr.xHat[0, 0], ekf_dr.xHat[1, 0]],
                    [ekf_gps.xHat[0, 0], ekf_gps.xHat[1, 0]],
                    [ekf_sf.xHat[0, 0], ekf_sf.xHat[1, 0]],
                ])

                # KF scope (safe even if you didn't enable prediction/correction)
                biasScope.axes[0].sample(t, [kf.xHat[0, 0]])
                biasScope.axes[1].sample(t, [kf.xHat[1, 0]])

                count = 0

            # refresh GUI events (same “system style” as your first script)
            MultiScope.refreshAll()

    # stop the car on exit
    qcar.read_write_std(throttle=0, steering=0)


# ---------------------------------------------------------------------------

#region : Setup and run experiment
if __name__ == '__main__':
    if IS_PHYSICAL_QCAR:
        fps = 10
    else:
        fps = 30

    # Scope for displaying estimated gyroscope bias
    biasScope = MultiScope(
        rows=2,
        cols=1,
        title='Heading Kalman Filter',
        fps=fps
    )
    biasScope.addAxis(
        row=0,
        col=0,
        xLabel='Time [s]',
        yLabel='Heading Angle [rad]',
        timeWindow=tf
    )
    biasScope.axes[0].attachSignal()

    biasScope.addAxis(
        row=1,
        col=0,
        xLabel='Time [s]',
        yLabel='Gyroscope Bias [rad/s]',
        timeWindow=tf
    )
    biasScope.axes[1].attachSignal()

    # Scope for comparing performance of various estimator types
    scope = MultiScope(rows=3, cols=2, title='QCar State Estimation', fps=fps)

    scope.addAxis(row=0, col=0, timeWindow=tf, yLabel='x Position [m]')
    scope.axes[0].attachSignal(name='x_dr')
    scope.axes[0].attachSignal(name='x_ekf_gps')
    scope.axes[0].attachSignal(name='x_ekf_sf')

    scope.addAxis(row=1, col=0, timeWindow=tf, yLabel='y Position [m]')
    scope.axes[1].attachSignal(name='y_dr')
    scope.axes[1].attachSignal(name='y_ekf_gps')
    scope.axes[1].attachSignal(name='y_ekf_sf')

    scope.addAxis(row=2, col=0, timeWindow=tf, yLabel='Heading Angle [rad]')
    scope.axes[2].xLabel = 'Time [s]'
    scope.axes[2].attachSignal(name='th_dr')
    scope.axes[2].attachSignal(name='th_ekf_gps')
    scope.axes[2].attachSignal(name='th_ekf_sf')

    scope.addXYAxis(
        row=0,
        col=1,
        rowSpan=3,
        xLabel='x Position [m]',
        yLabel='y Position [m]',
        xLim=(-1.5, 1.5),
        yLim=(-0.5, 2.5)
    )
    scope.axes[3].attachSignal(name='ekf_dr')
    scope.axes[3].attachSignal(name='ekf_gps')
    scope.axes[3].attachSignal(name='ekf_sf')

    try:
        controlLoop()
    except SystemExit:
        pass

    input('Experiment complete. Press any key to exit...')
#endregion
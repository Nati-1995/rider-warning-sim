"""Warning rules, same as Task 5.

Warn when time to crash < time the rider needs + margin and the bike is in the
door zone, two updates in a row.
"""
import math
from collections import deque

import numpy as np

import config as C


def t_req(v):
    """Time the rider needs at speed v: react + brake."""
    return C.T_REACT + v / (2 * C.A_BRAKE)


def slope_sigma(sig, dt, n):
    """Slope error of a line fit through n points, dt apart."""
    return sig / (dt * math.sqrt(n * (n * n - 1) / 12))


class Mode:
    """One way to warn (camera, radar, ...)."""

    def __init__(self, name, rate_hz, proc_s, n_speed, kind, geom):
        self.name, self.rate, self.proc, self.n, self.kind = name, rate_hz, proc_s, n_speed, kind
        self.geom = geom      # car_half, bike_half, rear_to_crash in m
        self.reset()

    def reset(self):
        self.hist = deque(maxlen=max(self.n, C.N_LATERAL))
        self.hits = 0
        self.warned_at = None
        self.updates = 0
        self.trace = []       # (t, ttc, speed, gap) every update, used to check the modes

    def margin(self, v):
        """Margin = 2 sigma of the TTC error + 2 frames."""
        ttc = t_req(v)
        d = v * ttc
        if self.kind == "camera":
            h = C.CAM_FPX * C.BOX_HEIGHT_M / d
            sig = slope_sigma(C.BOX_JITTER_PX, 1 / self.rate, self.n) / (h / ttc) * ttc
        else:
            sv = C.RADAR_SIGMA_V / math.sqrt(self.n)
            sig = ttc * math.sqrt((C.RADAR_SIGMA_D / d) ** 2 + (sv / v) ** 2)
        return 2 * sig + 2 / self.rate

    def decide(self, t, ttc, v_est, gap):
        self.updates += 1
        self.trace.append((t, ttc, v_est, gap))
        if self.warned_at is not None:
            return False
        if ttc is None or v_est is None or v_est <= 0.1 or gap is None:
            self.hits = 0
            return False
        inside = ttc <= t_req(v_est) + self.margin(v_est) and gap < C.GAP_LIMIT
        self.hits = self.hits + 1 if inside else 0
        if self.hits >= 2:
            self.warned_at = t + self.proc
            return True
        return False


class RadarMode(Mode):
    """Radar gives distance and speed every frame."""

    def measure(self, points):
        """Distance, speed and side position of the bike, None if missed."""
        if points is None or len(points) == 0:
            return None
        points = np.asarray(points, dtype=float)
        moving = points[np.abs(points[:, 3]) > C.RADAR_MIN_SPEED]
        if len(moving) == 0:
            return None
        depth, az, alt, vel = moving.T
        x = depth * np.cos(alt) * np.cos(az)          # distance behind the car
        y = depth * np.cos(alt) * np.sin(az)          # keep the sign, abs only after averaging
        near = x <= x.min() + C.RADAR_GROUP_M          # nearest points = the bike
        x_m = float(np.median(x[near]))
        y_m = float(np.median(y[near]))
        v_m = float(np.median(np.abs(vel[near])))      # only the bike moves so abs is fine
        return x_m, v_m, y_m

    def update(self, t, points, gap_from_camera=None, hand=True):
        """points: (depth, azimuth, altitude, speed) rows from one frame."""
        meas = self.measure(points)
        if meas is None:
            return self.decide(t, None, None, None)
        x_m, v_m, y_m = meas
        self.hist.append((t, x_m, v_m, y_m))
        if len(self.hist) < self.n:
            return self.decide(t, None, None, None)
        recent = list(self.hist)[-self.n:]
        v_est = float(np.mean([q[2] for q in recent]))
        d_crash = x_m + self.geom["rear_to_crash"] - v_est * self.proc
        ttc = d_crash / v_est if v_est > 0.1 else None
        gap = abs(float(np.mean([q[3] for q in self.hist]))) - self.geom["car_half"]
        if gap_from_camera is not None:
            gap = gap_from_camera
        return self.decide(t, ttc, v_est, gap)


class Kalman:
    """Kalman filter (constant speed) on radar distance and speed, it keeps guessing when a frame is missed."""

    def __init__(self):
        self.x = None                                  # [distance, speed]
        self.P = None
        self.t = None
        self.R = np.diag([C.RADAR_SIGMA_D ** 2, C.RADAR_SIGMA_V ** 2])
        self.fed = 0

    def predict(self, t):
        if self.x is None:
            return
        dt = t - self.t
        self.t = t
        F = np.array([[1.0, -dt], [0.0, 1.0]])         # distance goes down by v*dt
        q = C.KF_ACCEL_SIGMA ** 2
        Q = q * np.array([[dt ** 4 / 4, -dt ** 3 / 2], [-dt ** 3 / 2, dt ** 2]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def correct(self, t, d, v):
        z = np.array([d, v])
        if self.x is None:
            self.x, self.P, self.t = z, self.R.copy(), t
        else:
            K = self.P @ np.linalg.inv(self.P + self.R)
            self.x = self.x + K @ (z - self.x)
            self.P = (np.eye(2) - K) @ self.P
        self.fed += 1


class NewMode(RadarMode):
    """Our design: radar says when, camera says the gap, Kalman fills the gaps.

    Caution comes 1 s before Danger, Danger only if someone holds the door handle.
    proc includes the Bluetooth delay to the phone.
    """

    def reset(self):
        super().reset()
        self.kf = Kalman()
        self.c_hits = 0
        self.caution_at = None
        self.caution_now = False

    def update(self, t, points, gap_from_camera=None, hand=True):
        self.caution_now = False
        self.updates += 1
        self.kf.predict(t)
        meas = self.measure(points)
        if meas is not None:
            x_m, v_m, y_m = meas
            self.kf.correct(t, x_m, v_m)
            self.hist.append((t, x_m, v_m, y_m))
        if self.kf.fed < self.n or not self.hist:
            self.hits = self.c_hits = 0
            return False
        d_hat, v_hat = float(self.kf.x[0]), float(self.kf.x[1])
        if v_hat <= 0.1:
            self.hits = self.c_hits = 0
            return False
        ttc = (d_hat + self.geom["rear_to_crash"] - v_hat * self.proc) / v_hat
        gap = abs(float(np.mean([q[3] for q in self.hist]))) - self.geom["car_half"]
        if gap_from_camera is not None:
            gap = gap_from_camera
        limit = t_req(v_hat) + self.margin(v_hat)
        self.trace.append((t, ttc, v_hat, gap, d_hat))
        near_door = gap < C.GAP_LIMIT
        if self.caution_at is None:
            self.c_hits = self.c_hits + 1 if near_door and ttc <= limit + C.CAUTION_LEAD else 0
            if self.c_hits >= 2:
                self.caution_at = t + self.proc
                self.caution_now = True
        if self.warned_at is None:
            self.hits = self.hits + 1 if near_door and hand and ttc <= limit else 0
            if self.hits >= 2:
                self.warned_at = t + self.proc
                return True
        return False


class CameraMode(Mode):
    """Camera gets the TTC from how fast the box grows (line fit on 1/h)."""

    def reset(self):
        super().reset()
        self.last_gap = None
        self.last_seen = None

    def update(self, t, box):
        """box: (x1, y1, x2, y2) in px, None if nothing found."""
        if box is None:
            return self.decide(t, None, None, None)
        h = box[3] - box[1]
        xc = (box[0] + box[2]) / 2
        if h < C.MIN_BOX_PX:
            return self.decide(t, None, None, None)     # too small
        if box[0] <= 1 or box[2] >= C.CAM_W - 1:
            return self.decide(t, None, None, None)     # box cut by the edge
        y_side = (xc - C.CAM_W / 2) * C.BOX_HEIGHT_M / h   # side position
        self.hist.append((t, h, y_side))
        if len(self.hist) < self.n:
            return self.decide(t, None, None, None)
        recent = list(self.hist)[-self.n:]
        ts = np.array([q[0] for q in recent])
        inv = 1 / np.array([q[1] for q in recent])
        k, b = np.polyfit(ts, inv, 1)
        inv_end = k * ts[-1] + b
        if k >= 0 or inv_end <= 0:
            return self.decide(t, None, None, None)
        ttc_cam = -inv_end / k                         # time until it reaches the camera
        d_est = C.CAM_FPX * C.BOX_HEIGHT_M * inv_end
        v_est = d_est / max(ttc_cam, 0.05)
        ttc = ttc_cam + self.geom["rear_to_crash"] / v_est - self.proc
        y = float(np.mean([abs(q[2]) for q in recent[-3:]]))
        gap = y - self.geom["car_half"] - self.geom["bike_half"]
        self.last_gap, self.last_seen = gap, t
        return self.decide(t, ttc, v_est, gap)


def make_modes(geom):
    modes = []
    for name, rate, proc, n, kind in C.MODES:
        cls = {"camera": CameraMode, "new": NewMode}.get(kind, RadarMode)
        modes.append(cls(name, rate, proc, n, kind, geom))
    return modes


def pick_bike_box(boxes):
    """Biggest bike box, or biggest person box if no bike. boxes: (cls, conf, x1, y1, x2, y2)."""
    if not boxes:
        return None
    bikes = [b for b in boxes if int(b[0]) in (1, 3)]
    pool = bikes if bikes else [b for b in boxes if int(b[0]) == 0]
    if not pool:
        return None
    best = max(pool, key=lambda b: (b[4] - b[2]) * (b[5] - b[3]))
    return tuple(best[2:6])


def score(rows, mode_names):
    """One summary line per mode."""
    out = []
    for name in mode_names:
        r = [q for q in rows if q["mode"] == name]
        danger = [q for q in r if q["path"] == "danger"]
        safe = [q for q in r if q["path"] == "safe"]
        nohand = [q for q in r if q["path"] == "nohand"]
        warned_danger = sum(q["warned"] for q in danger)
        in_time = sum(q["in_time"] for q in danger)
        false_alarm = sum(q["warned"] or q["caution_warned"] for q in safe)
        nohand_alarm = sum(q["warned"] for q in nohand)
        caution_in_time = sum(q["caution_in_time"] for q in danger)
        precision = warned_danger / (warned_danger + false_alarm) if warned_danger + false_alarm else 0.0
        recall = warned_danger / len(danger) if danger else 0.0
        leads = [q["lead_s"] for q in danger if q["warned"]]
        out.append(dict(
            mode=name, danger_runs=len(danger), safe_runs=len(safe), nohand_runs=len(nohand),
            in_time=in_time, warned=warned_danger, false_alarms=false_alarm, nohand_alarms=nohand_alarm,
            in_time_pct=100 * in_time / len(danger) if danger else 0.0,
            false_alarm_pct=100 * false_alarm / len(safe) if safe else 0.0,
            nohand_alarm_pct=100 * nohand_alarm / len(nohand) if nohand else 0.0,
            caution_in_time=caution_in_time, has_caution=any(q["caution_warned"] for q in r),
            precision=precision, recall=recall,
            mean_lead_s=float(np.mean(leads)) if leads else float("nan"),
        ))
    return out

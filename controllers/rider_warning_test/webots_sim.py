"""Webots part: moves the bike and reads the radar and camera."""
import math
import os

import numpy as np

import config as C

CAR_HALF_WIDTH = 0.97           # half width of the BMW X5, m
CAR_CENTRE_FROM_ORIGIN = 1.425  # origin is the rear axle, centre is this far ahead, m
PARK = {"SCOOTER": -500.0, "MOTORBIKE": -520.0}     # the other bike waits here
NAMES = {"SCOOTER": "scooter", "MOTORBIKE": "motorbike"}
# Webots gives one radar point in the middle of the bike, a real radar sees the front wheel
# first so we move it forward by half the bike length
RADAR_POINT_TO_FRONT = 1.2      # m
PICTURE_AT = (60.0, 30.0, 15.0)  # save pictures at these distances, m

GREEN, RED, ORANGE, GREY, WHITE = 0x3CDC78, 0xFF5A50, 0xFFB020, 0xE6E6E6, 0xFFFFFF


class WebotsSim:
    def __init__(self, sup, light, use_camera, detector_kind, picture_dir=None):
        self.sup = sup
        self.dt_ms = int(sup.getBasicTimeStep())
        C.FIXED_DT = self.dt_ms / 1000.0
        self.light = light
        self.detector_kind = detector_kind
        self.picture_dir = picture_dir
        self.rng = np.random.default_rng(C.SEED)
        self.cam_drop = C.CAM_DROP_NIGHT if light == "night" else C.CAM_DROP_DAY

        self.radar = sup.getDevice("rear radar")
        self.radar.enable(self.dt_ms)
        self.radar_max = self.radar.getMaxRange()
        self.cam = None
        if use_camera:
            self.cam = sup.getDevice("rear camera")
            self.cam.enable(2 * self.dt_ms)
            if detector_kind == "recognition":
                self.cam.recognitionEnable(2 * self.dt_ms)
            C.CAM_W, C.CAM_H = self.cam.getWidth(), self.cam.getHeight()
            C.CAM_FPX = (C.CAM_W / 2) / math.tan(self.cam.getFov() / 2)

        car = sup.getFromDef("CAR").getField("translation").getSFVec3f()
        rig = sup.getFromDef("RIG").getField("translation").getSFVec3f()
        self.car_y = car[1]
        self.crash_x = car[0] + CAR_CENTRE_FROM_ORIGIN + C.CRASH_POINT_FROM_CENTRE
        self.geom = dict(car_half=CAR_HALF_WIDTH, bike_half=C.BIKE_HALF_WIDTH,
                         rear_to_crash=self.crash_x - rig[0])
        self.bikes = ["SCOOTER", "MOTORBIKE"]
        self.tr = {b: sup.getFromDef(b).getField("translation") for b in self.bikes}
        self.body_ids = self._bike_body_ids()
        lamp = sup.getFromDef("LAMP")
        self.lamp = lamp.getField("emissiveColor") if lamp is not None else None
        for b in self.bikes:
            self.tr[b].setSFVec3f([PARK[b], self.car_y + 2.0, 0.0])
        self.bike = None
        self.status = {}
        self.pictures_taken = set()

    def _bike_body_ids(self):
        """Ids of the bike models so the camera ignores everything else."""
        ids = set()
        for b in self.bikes:
            kids = self.sup.getFromDef(b).getField("children")
            for i in range(kids.getCount()):
                node = kids.getMFNode(i)
                if node.getTypeName() != "Pose":
                    continue
                inner = node.getField("children")
                for j in range(inner.getCount()):
                    if inner.getMFNode(j).getBaseTypeName() == "Solid":
                        ids.add(inner.getMFNode(j).getId())
        return ids

    # text on the 3D view, with a dark copy behind so you can read it
    def _label(self, i, text, y, colour=WHITE):
        self.sup.setLabel(20 + i, text, 0.012, y + 0.003, 0.05, 0x000000, 0.0, "Arial")
        self.sup.setLabel(i, text, 0.01, y, 0.05, colour, 0.0, "Arial")

    def _lamp(self, rgb):
        if self.lamp is not None:
            self.lamp.setSFColor(rgb)

    def _draw_status(self):
        for i, (name, *_rest) in enumerate(C.MODES):
            text, colour = self.status.get(name, ("watching", GREY))
            self._label(1 + i, f"{name}: {text}", 0.06 + 0.042 * i, colour)

    # protocol.py calls these
    def on_pass_start(self, run_id, bike, speed_kmh, path):
        kind = {"danger": "door zone pass (must warn)",
                "safe": "safe pass (must stay quiet)",
                "nohand": "door zone, nobody at the door (no Danger)"}[path]
        self._label(0, f"Run {run_id}   {NAMES[bike]}   {speed_kmh:g} km/h   {kind}   {self.light}", 0.01)
        self.status = {}
        self._draw_status()

    def on_caution(self, mode_name, lead_s, in_time, path):
        if path == "safe":
            self.status[mode_name] = ("false alarm (Caution)", ORANGE)
        else:
            self.status[mode_name] = (f"Caution {lead_s:.2f} s before the crash point", GREEN if in_time else RED)
        self._lamp([1, 0.6, 0])
        self._draw_status()

    def on_warning(self, mode_name, lead_s, in_time, path):
        if path == "safe":
            self.status[mode_name] = ("false alarm", ORANGE)
        elif path == "nohand":
            self.status[mode_name] = ("Danger with nobody at the door", ORANGE)
        elif mode_name in self.status and self.status[mode_name][0].startswith("Caution"):
            caution = self.status[mode_name][0].split(" before")[0]
            self.status[mode_name] = (f"{caution}, Danger {lead_s:.2f} s, {'in time' if in_time else 'late'}",
                                      GREEN if in_time else RED)
        else:
            self.status[mode_name] = (f"warned {lead_s:.2f} s before the crash point, {'in time' if in_time else 'late'}",
                                      GREEN if in_time else RED)
        if mode_name.startswith("New system"):
            self._lamp([1, 0, 0])
        self._draw_status()

    def finished(self, text):
        for i in range(1 + len(C.MODES)):
            self._label(i, "", 0.0)
        self._label(0, text, 0.01)

    def start_pass(self, bike, speed_ms, gap_m):
        self.bike = bike
        self.v = speed_ms
        self.gap = gap_m
        self.y = self.car_y + CAR_HALF_WIDTH + gap_m + C.BIKE_HALF_WIDTH
        self.x0 = self.crash_x - C.START_DISTANCE    # bike origin = front of the bike
        self.t = 0.0
        self.tr[bike].setSFVec3f([self.x0, self.y, 0.0])
        self._lamp([0, 0, 0])
        if self.sup.step(self.dt_ms) == -1:
            raise SystemExit

    def step(self):
        self.t += C.FIXED_DT
        x = self.x0 + self.v * self.t
        self.tr[self.bike].setSFVec3f([x, self.y, 0.0])
        if self.sup.step(self.dt_ms) == -1:
            raise SystemExit
        truth = dict(d=self.crash_x - x,
                     gap=(self.y - C.BIKE_HALF_WIDTH) - (self.car_y + CAR_HALF_WIDTH),
                     v=self.v)
        image, boxes = None, None
        if self.cam is not None:
            if self.detector_kind == "recognition":
                boxes = self._recognition_boxes()
            else:
                raw = self.cam.getImage()
                if raw:
                    image = np.frombuffer(raw, dtype=np.uint8).reshape((C.CAM_H, C.CAM_W, 4))[:, :, :3]
        if self.picture_dir:
            self._pictures(truth["d"])
        return dict(t=self.t, radar=self._radar_rows(), image=image, boxes=boxes, truth=truth)

    def _radar_rows(self):
        rows = []
        if C.NOISE and self.rng.random() < C.RADAR_DROP:
            return np.zeros((0, 4))     # missed frame
        for tg in self.radar.getTargets():
            if not (math.isfinite(tg.distance) and math.isfinite(tg.speed) and math.isfinite(tg.azimuth)):
                continue            # first frame has no speed
            if tg.distance >= self.radar_max - 0.01:
                continue            # out of range
            if tg.distance > C.RADAR_RANGE_BIKE:
                continue            # bike only seen to 84 m
            along = tg.distance * math.cos(tg.azimuth) - RADAR_POINT_TO_FRONT
            side = tg.distance * math.sin(tg.azimuth)
            rows.append((math.hypot(along, side), math.atan2(side, along), 0.0, tg.speed))
        return np.array(rows, dtype=float).reshape(-1, 4)

    def _recognition_boxes(self):
        """Boxes from Webots recognition, same format as YOLO."""
        boxes = []
        if C.NOISE and self.rng.random() < self.cam_drop:
            return boxes                # missed picture
        for obj in self.cam.getRecognitionObjects():
            if obj.getId() not in self.body_ids:
                continue
            pos = obj.getPosition()
            if self.light == "night" and math.sqrt(sum(p * p for p in pos)) > C.CAMERA_RANGE_NIGHT:
                continue            # night range cut
            px, py = obj.getPositionOnImage()
            w, h = obj.getSizeOnImage()
            if C.NOISE:                 # boxes jump a bit like a real detector
                px += self.rng.normal(0, C.BOX_JITTER_PX)
                h += self.rng.normal(0, C.BOX_JITTER_PX)
            boxes.append((3, 1.0, px - w / 2, py - h / 2, px + w / 2, py + h / 2))
        return boxes

    def _pictures(self, d):
        for want in PICTURE_AT:
            key = (self.bike, self.gap, want)
            if key in self.pictures_taken or not (d <= want < d + self.v * C.FIXED_DT):
                continue
            self.pictures_taken.add(key)
            path = "door_zone" if self.gap < C.GAP_LIMIT else "safe"
            tag = os.path.join(self.picture_dir, f"{self.light}_{NAMES[self.bike]}_{path}_{int(want)}m")
            self.sup.exportImage(tag + "_view.jpg", 90)
            if self.cam is not None:
                self.cam.saveImage(tag + "_camera.jpg", 90)

    def end_pass(self):
        self.tr[self.bike].setSFVec3f([PARK[self.bike], self.car_y + 2.0, 0.0])
        self.sup.step(self.dt_ms)

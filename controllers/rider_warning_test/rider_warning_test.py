"""Webots controller for the rider warning test (CEN451 Capstone I).

Settings come from controllerArgs of "sensor rig" in the world file:
  --light=day or --light=night   only changes the night camera range, the world file sets the lights
  --passes=10                    passes per bike, like the competition
  --speeds=20,40,60              bike speeds in km/h, used in turn
  --detector=recognition         camera detector: recognition (built into Webots, fast),
                                 yolo (YOLO on the real camera picture, needs ultralytics) or none
  --noise=on or --noise=off      sensor errors and missed frames (off = perfect sensors)
  --seed=1                       different number = different random errors
  --hand-test=on or off          extra passes with nobody at the door (60 runs, off = 40)
  --pictures                     save the 3D view and the camera picture at 60, 30 and 15 m
  --quit                         close Webots when the test ends
"""
import os
import sys

from controller import Supervisor

import config as C
from protocol import run_protocol
from webots_sim import WebotsSim


def read_args(argv):
    opts = dict(light="day", passes=C.PASSES_PER_BIKE, speeds=list(C.SPEEDS_KMH), detector="recognition",
                quit=False, pictures=False, noise="on", seed=C.SEED, hand_test="on")
    for a in argv[1:]:
        if a in ("--quit", "--pictures"):
            opts[a[2:]] = True
        elif "=" in a:
            k, v = a.lstrip("-").split("=", 1)
            k = k.replace("-", "_")
            if k in ("passes", "seed"):
                opts[k] = int(v)
            elif k == "speeds":
                opts[k] = [float(s) for s in v.split(",") if s]
            elif k in opts:
                opts[k] = v
    return opts


def make_yolo():
    """YOLO on one picture, returns (cls, conf, x1, y1, x2, y2) boxes."""
    import numpy as np
    from ultralytics import YOLO
    model = YOLO(C.YOLO_MODEL)

    def detect(image_bgr):
        res = model.predict(np.ascontiguousarray(image_bgr), imgsz=640, conf=C.YOLO_CONF,
                            classes=C.YOLO_CLASSES, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        return [(int(c), float(p), *map(float, b)) for c, p, b in zip(cls, conf, xyxy)]
    return detect


def pick_detector(kind):
    if kind == "none":
        return "none", None
    if kind == "yolo":
        try:
            return "yolo", make_yolo()
        except ImportError:
            print("ultralytics is not installed (pip install ultralytics), using the Webots recognition camera instead")
    return "recognition", None


def main():
    opts = read_args(sys.argv)
    C.NOISE = opts["noise"] != "off"
    C.SEED = opts["seed"]
    C.HAND_TEST = opts["hand_test"] != "off"
    sup = Supervisor()
    kind, detector = pick_detector(opts["detector"])
    print(f"Rider warning test, {opts['light']}, {opts['passes']} passes per bike, "
          f"speeds {', '.join(f'{s:g}' for s in opts['speeds'])} km/h, camera detector: {kind}, "
          f"noise {'on' if C.NOISE else 'off'}, nobody-at-the-door passes {'on' if C.HAND_TEST else 'off'}")
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.normpath(os.path.join(here, "..", "..", "results"))
    pictures = os.path.join(out, "pictures") if opts["pictures"] else None
    if pictures:
        os.makedirs(pictures, exist_ok=True)
    sim = WebotsSim(sup, opts["light"], kind != "none", kind, pictures)
    run_protocol(sim, opts["light"], opts["speeds"], opts["passes"], out, detector=detector, use_camera=kind != "none")
    sim.finished(f"Test finished. Open the results folder: {out}")
    for _ in range(10):      # few more steps so the last lines print
        sup.step(int(sup.getBasicTimeStep()))
    if opts["quit"]:
        sup.simulationQuit(0)
    else:
        sup.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
        sup.step(int(sup.getBasicTimeStep()))


if __name__ == "__main__":
    main()

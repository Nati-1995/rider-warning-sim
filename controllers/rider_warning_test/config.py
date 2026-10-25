"""All the numbers for the rider warning test are here, change them here not in the code."""
import math

# simulation
FIXED_DT = 0.016             # one sim step in s, same as basicTimeStep in the world file

# rider (from Task 5)
T_REACT = 1.5                # reaction time in s (Green 2000)
A_BRAKE = 5.0                # braking in m/s^2, our guess

# test plan like the competition, 2 bikes pass the parked car 10 times each
PASSES_PER_BIKE = 10
SPEEDS_KMH = [20, 40, 60]    # km/h, used one after the other
DANGER_GAP = 0.5             # gap between car and bike in a door zone pass, m
SAFE_GAP = 2.5               # gap for a safe pass, m
GAP_LIMIT = 1.5              # closer than this = door zone, m
HAND_TEST = True             # extra door zone passes with no hand on the door handle
START_DISTANCE = 110.0       # where the bike starts, m behind the crash point
END_DISTANCE = -6.0          # pass stops when the bike is this far past it, m
CRASH_POINT_FROM_CENTRE = 0.2  # front door edge, m ahead of the car centre

# radar (the Radar node is in the world file, 100 m and 40 deg like the TI 77 GHz board)
RADAR_MIN_SPEED = 0.5        # slower than this = road or parked car, ignore it, m/s
RADAR_GROUP_M = 2.5          # points this close to the nearest one belong to the bike, m
RADAR_SIGMA_D = 0.23         # m, from the TI design (Task 5), same as rangeNoise in the world
RADAR_SIGMA_V = 0.09         # m/s, same as speedNoise in the world
RADAR_SIGMA_AZ = 0.035       # rad (2 deg), same as angularNoise in the world
RADAR_RANGE_BIKE = 84.0      # radar sees a bike up to 84 m (Task 5)
RADAR_DROP = 0.05            # 5% of radar frames miss the bike
N_LATERAL = 30               # frames we average for the side position (about 0.5 s)

# camera
CAM_W, CAM_H = 1280, 720      # read again from Webots at the start
CAM_FOV = 60.0               # deg
CAM_FPX = (CAM_W / 2) / math.tan(math.radians(CAM_FOV / 2))   # focal length in px
BOX_HEIGHT_M = 1.55          # bike + rider height, m (we measured it in Webots)
BOX_JITTER_PX = 2.0          # boxes jump about 2 px
MIN_BOX_PX = 30              # smaller boxes are too noisy, so we skip them (about 57 m at 720p)
BIKE_HALF_WIDTH = 0.4        # m
YOLO_MODEL = "yolo11n.pt"    # ultralytics downloads it the first time
YOLO_CLASSES = [0, 1, 3]     # COCO person, bicycle, motorcycle
YOLO_CONF = 0.3
CAMERA_RANGE_NIGHT = 25.0    # m, Webots recognition sees in the dark so we cut it here
CAM_DROP_DAY = 0.05          # 5% of pictures miss the bike in the day
CAM_DROP_NIGHT = 0.15        # and 15% at night

# noise on = random sensor errors and missed frames, off = perfect sensors
NOISE = True
SEED = 1                     # same seed = same random errors

# new system (Task 5 and Task 7)
CAUTION_LEAD = 1.0           # Caution comes this many s before Danger
BT_DELAY = 0.1               # Bluetooth to the phone, s (our guess in Task 7)
KF_ACCEL_SIGMA = 2.0         # how much the bike speed can change, for the Kalman filter

# warning modes: name, rate (Hz), processing time (s), frames needed for the speed, type
MODES = [
    ("Camera now (10 Hz)", 10, 0.100, 5, "camera"),
    ("Camera on Jetson (30 fps)", 30, 0.005, 13, "camera"),
    ("Radar 77 GHz (60 Hz)", 60, 0.005, 3, "radar"),
    ("Radar + camera", 60, 0.005, 3, "fusion"),
    ("New system", 60, 0.005 + BT_DELAY, 3, "new"),   # our design
]

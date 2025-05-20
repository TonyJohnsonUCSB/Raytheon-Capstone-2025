import time
import math
import numpy as np
import cv2
from picamera2 import Picamera2

# — Calibration (hard‑coded) —
INTRINSIC = np.array([
    [653.1070007239106,   0.0,               339.2952147845755],
    [0.0,                 650.7753992788821, 258.1165494889447],
    [0.0,                 0.0,               1.0]
], dtype=np.float32)
DIST_COEFFS = np.array([
    -0.03887864427953473,
     0.6888798469690414,
     0.00815702400928161,
     0.010438854120041072,
    -1.713270699000528
], dtype=np.float32)

# — Params —
RESOLUTION   = (640, 480)
MARKER_SIZE  = 0.06611   # meters
DROP_ZONE_ID = 1
ARUCO_DICT   = cv2.aruco.DICT_6X6_250

# — Camera GPS (provide your camera’s known location) —
CAM_LAT = 37.4219999     # degrees
CAM_LON = -122.0840575   # degrees
CAM_ALT = 10.0           # meters above sea level

# — Earth model for small‑angle ENU→LL conversion —
R_EARTH = 6378137.0                      # equatorial radius in meters
def enu_to_geodetic(east, north, up):
    dlat = north / R_EARTH               # in radians
    dlon = east / (R_EARTH * math.cos(math.radians(CAM_LAT)))
    lat = CAM_LAT + math.degrees(dlat)
    lon = CAM_LON + math.degrees(dlon)
    alt = CAM_ALT + up
    return lat, lon, alt

# — Frame‑to‑frame stabilization —
prev_gray = None
def stabilize(frame):
    global prev_gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        prev_gray = gray
        return frame
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200,
                                  qualityLevel=0.01, minDistance=30)
    if pts is not None:
        curr, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
        good = st.ravel()==1
        if np.count_nonzero(good) >= 6:
            src, dst = pts[good], curr[good]
            M, _ = cv2.estimateAffinePartial2D(src, dst)
            if M is not None:
                frame = cv2.warpAffine(frame, M, RESOLUTION)
    prev_gray = gray
    return frame

# — Pose estimation with GPS computation —
def pose_estimation(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(
        gray, cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    )
    if ids is None:
        return

    for idx, mid in enumerate(ids.flatten()):
        if mid != DROP_ZONE_ID:
            continue
        c = corners[idx]
        _, _, tvecs = cv2.aruco.estimatePoseSingleMarkers(
            [c], MARKER_SIZE, INTRINSIC, DIST_COEFFS
        )
        if not len(tvecs):
            continue

        # convert to ENU (meters)
        x_east, y_north, z_down = tvecs[0][0]
        z_up = -z_down

        # compute marker geodetic coords
        lat, lon, alt = enu_to_geodetic(x_east, y_north, z_up)

        # overlay
        pts = c.reshape(4, 2)
        cX, cY = int(pts[:,0].mean()), int(pts[:,1].mean())
        txt = f"Lat:{lat:.12f}, Lon:{lon:.12f}, Alt:{alt:.1f}m"
        cv2.putText(frame, txt, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

# — Main loop —
picam2 = Picamera2()
cfg = picam2.create_preview_configuration(
    main={"format":"RGB888", "size":RESOLUTION}
)
picam2.configure(cfg)
picam2.start()
time.sleep(2)

try:
    while True:
        frame = picam2.capture_array()
        stab = stabilize(frame)
        pose_estimation(stab)
        cv2.imshow("Stabilized Feed", stab)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()

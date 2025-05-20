import time
import numpy as np
import cv2
from picamera2 import Picamera2

# — Calibration (hard‑coded) —
INTRINSIC = np.array([
    [653.1070007239106,   0.0,               339.2952147845755],
    [0.0,                 650.7753992788821, 258.1165494889447],
    [0.0,                 0.0,               1.0]
])
DIST_COEFFS = np.array([
    -0.03887864427953473,
     0.6888798469690414,
     0.00815702400928161,
     0.010438854120041072,
    -1.713270699000528
])

# — Other params —
RESOLUTION   = (640, 480)
MARKER_SIZE  = 0.06611  # meters
DROP_ZONE_ID = 1
METERS_TO_CM = 100
ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250

# — Initialize camera —
picam2 = Picamera2()
cfg = picam2.create_preview_configuration(
    main={"format":"RGB888","size":RESOLUTION}
)
picam2.configure(cfg)
picam2.start()
time.sleep(2)  # warm‑up

# — Video writer to record preview —
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out    = cv2.VideoWriter(
    "/home/rtxcapstone/Desktop/testVideo.avi",
    fourcc,
    20.0,
    RESOLUTION
)

def pose_estimation(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None:
        return

    for idx, marker_id in enumerate(ids.flatten()):
        if marker_id != DROP_ZONE_ID:
            continue

        c = corners[idx]
        _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [c], MARKER_SIZE, INTRINSIC, DIST_COEFFS
        )
        if not len(tvecs):
            continue

        x_cm, y_cm, z_cm = tvecs[0][0] * METERS_TO_CM
        pts = c.reshape((4, 2))
        cX, cY = int(pts[:,0].mean()), int(pts[:,1].mean())
        text = f"X:{x_cm:.1f}cm Y:{y_cm:.1f}cm Z:{z_cm:.1f}cm"
        cv2.putText(frame, text, (cX-100, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

try:
    while True:
        frame = picam2.capture_array()
        pose_estimation(frame)
        cv2.imshow("Live Feed", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    out.release()
    picam2.stop()
    cv2.destroyAllWindows()


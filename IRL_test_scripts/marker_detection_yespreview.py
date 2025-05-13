import os
# Suppress libpng incorrect sRGB profile warnings
os.environ['OPENCV_IO_ENABLE_PNG_WARNINGS'] = '0'

import numpy as np
import cv2
import time
import traceback
from picamera2 import Picamera2

# Top-level video file name
VIDEO_NAME = '/home/rtxcapstone/Desktop/testVideo.avi'
RESOLUTION = (640, 480)

# Conversion factor
METERS_TO_CM = 100

# --- Vibration Mitigation Parameters ---
ALPHA = 0.5                    # pose smoothing factor (0 < ALPHA <= 1)
prev_tvecs = {}                # store previous tvecs per marker
prev_gray = None               # for frame-to-frame stabilization

def pose_estimation(frame, aruco_dict_type, matrix_coeffs, dist_coeffs, drop_zoneID, marker_size):
    """
    Detects the drop zone marker and overlays x, y, z offsets in centimeters.
    """
    global prev_tvecs
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        params = cv2.aruco.DetectorParameters_create()
        params.adaptiveThreshConstant = 7
        params.minMarkerPerimeterRate = 0.03

        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        if ids is None:
            return

        for idx, marker_id in enumerate(ids.flatten()):
            if marker_id != drop_zoneID:
                continue

            c = corners[idx]
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [c], marker_size, matrix_coeffs, dist_coeffs
            )
            if tvecs is None or len(tvecs) == 0:
                continue

            # Smooth translation vector
            tvec = tvecs[0][0]
            if marker_id in prev_tvecs:
                tvec = ALPHA * tvec + (1 - ALPHA) * prev_tvecs[marker_id]
            prev_tvecs[marker_id] = tvec

            # Compute offsets and convert to centimeters
            x, y, z = tvec
            x_cm, y_cm, z_cm = x * METERS_TO_CM, y * METERS_TO_CM, z * METERS_TO_CM

            # Overlay text at marker center
            pts = c.reshape((4, 2))
            cX = int(pts[:, 0].mean())
            cY = int(pts[:, 1].mean())
            text = f"X:{x_cm:.1f}cm Y:{y_cm:.1f}cm Z:{z_cm:.1f}cm"
            cv2.putText(frame, text, (cX - 100, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    except Exception:
        print("-- Error in pose_estimation:")
        traceback.print_exc()

# --- Camera & Recorder Setup ---
ARUCO_DICT = {
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_6X6_250":      cv2.aruco.DICT_6X6_250
}
aruco_type = "DICT_6X6_250"
intrinsic = np.array([[933.15867, 0, 657.59],
                     [0, 933.1586, 400.36993],
                     [0, 0, 1]])
distortion = np.array([-0.43948, 0.18514, 0, 0])
drop_zoneID = 1
marker_size = 0.06611  # meters

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": RESOLUTION}
)
picam2.configure(config)

# Start camera and warm up sensor
picam2.start()
print("-- Camera started, warming up...")
time.sleep(2)

# Setup recording
FRAME_SIZE = RESOLUTION
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
default_fps = 30.0
writer = cv2.VideoWriter(VIDEO_NAME, fourcc, default_fps, FRAME_SIZE)
if not writer.isOpened():
    raise RuntimeError(f"Failed to open VideoWriter at {default_fps:.2f} FPS")
print(f"-- Recording at default {default_fps:.2f} FPS")

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            stab = frame.copy()
        else:
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200,
                                           qualityLevel=0.01, minDistance=30)
            M = None
            if pts is not None:
                curr, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                good = status.reshape(-1) == 1
                if np.count_nonzero(good) >= 6:
                    src = pts[good]
                    dst = curr[good]
                    M, _ = cv2.estimateAffinePartial2D(src, dst)
            stab = cv2.warpAffine(frame, M, FRAME_SIZE) if M is not None else frame.copy()
        prev_gray = gray

        # Pose estimation (X, Y, Z offsets in cm)
        pose_estimation(stab, ARUCO_DICT[aruco_type], intrinsic, distortion, drop_zoneID, marker_size)

        cv2.imshow("Stabilized Feed", stab)
        writer.write(stab)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("-- Interrupted by user.")
finally:
    print("-- Cleaning up...")
    picam2.stop()
    picam2.close()
    writer.release()
    cv2.destroyAllWindows()

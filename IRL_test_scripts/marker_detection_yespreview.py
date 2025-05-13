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
RESOLUTION = (640,480)
#RESOLUTION = (2304, 1296)
# --- Vibration Mitigation Parameters ---
ALPHA = 0.5                    # pose smoothing factor (0 < ALPHA <= 1)
prev_tvecs = {}                # store previous tvecs per marker
prev_gray = None               # for frame-to-frame stabilization

# --- Recording Settings ---
REQUESTED_FPS = 35            # desired camera capture rate (Hz)
FRAME_INTERVAL = None          # will be set after calibration

# --- Pose Estimation with Smoothing ---
def pose_estimation(frame, aruco_dict_type, matrix_coeffs, dist_coeffs, drop_zoneID, marker_size):
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
            c = corners[idx]
            if c is None or len(c) == 0:
                continue

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [c], marker_size, matrix_coeffs, dist_coeffs
            )
            if tvecs is None or len(tvecs) == 0:
                continue

            tvec = tvecs[0][0]
            if marker_id in prev_tvecs:
                tvec = ALPHA * tvec + (1 - ALPHA) * prev_tvecs[marker_id]
            prev_tvecs[marker_id] = tvec

            x, y, z = tvec
            pts = c.reshape((4,2))
            cX = int(pts[:,0].mean())
            cY = int(pts[:,1].mean())

            text1 = f"ID:{marker_id} X:{x:.2f}m Y:{y:.2f}m Z:{z:.2f}m"
            cv2.putText(frame, text1, (cX-80, cY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            if marker_id == drop_zoneID:
                dist = np.linalg.norm(tvec)
                ang = np.degrees(np.arctan2(x, z))
                dz_t = f"DZ:{dist:.2f}m Ang:{ang:.1f}\u00b0"
                cv2.putText(frame, dz_t, (cX-80, cY+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
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
    main={"format":"RGB888","size":RESOLUTION}
)
picam2.configure(config)

# Start camera and warm up sensor
picam2.start()
print("-- Camera started, warming up...")
time.sleep(2)

# --- Calibration: measure achievable FPS ---
warmup_frames = 15
timestamps = []
print(f"-- Calibrating over {warmup_frames} frames...")
for _ in range(warmup_frames):
    timestamps.append(time.time())
    _ = picam2.capture_array()
timestamps.append(time.time())
intervals = [t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:])]
avg_dt = sum(intervals) / len(intervals)
max_fps = 1.0 / avg_dt if avg_dt > 0 else REQUESTED_FPS
effective_fps = min(REQUESTED_FPS, max_fps)
if effective_fps < REQUESTED_FPS:
    print(f"-- Warning: max achievable fps ~{max_fps:.1f}, using {effective_fps:.1f} fps instead of requested {REQUESTED_FPS} fps")
else:
    print(f"-- Using requested fps: {REQUESTED_FPS} fps")

# Set capture parameters based on calibration
CAPTURE_FPS = effective_fps
FRAME_INTERVAL = 1.0 / CAPTURE_FPS
# Lock sensor frame duration
dur = int(1e6 / CAPTURE_FPS)
picam2.set_controls({"FrameDurationLimits":(dur, dur)})

# Setup recording at calibrated FPS
FRAME_SIZE = (640,480)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter(VIDEO_NAME, fourcc, CAPTURE_FPS, FRAME_SIZE)
if not writer.isOpened():
    raise RuntimeError(f"Failed to open VideoWriter at {CAPTURE_FPS:.2f} Hz")

print(f"-- Recording at {CAPTURE_FPS:.2f} fps")

try:
    while True:
        loop_start = time.time()

        # Capture and stabilize
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

        # Pose estimation
        pose_estimation(stab, ARUCO_DICT[aruco_type], intrinsic, distortion, drop_zoneID, marker_size)

        # Display & write
        cv2.imshow("Stabilized Feed", stab)
        writer.write(stab)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Throttle loop to calibrated FPS
        elapsed = time.time() - loop_start
        if elapsed < FRAME_INTERVAL:
            time.sleep(FRAME_INTERVAL - elapsed)

except KeyboardInterrupt:
    print("-- Interrupted by user.")
finally:
    print("-- Cleaning up...")
    picam2.stop()
    picam2.close()
    writer.release()
    cv2.destroyAllWindows()

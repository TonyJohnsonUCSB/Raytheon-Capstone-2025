import numpy as np
import cv2
import time
import traceback
from picamera2 import Picamera2

# --- Vibration Mitigation Parameters ---
ALPHA = 0.5                       # pose smoothing factor (0 < ALPHA <= 1)
prev_tvecs = {}                  # store previous tvecs per marker
prev_gray = None                 # for frame-to-frame stabilization

# --- Pose Estimation with Smoothing ---
def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, drop_zoneID, marker_size):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters_create()
        # Tweak detector parameters for robustness
        parameters.adaptiveThreshConstant = 7
        parameters.minMarkerPerimeterRate = 0.03

        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
        if ids is None:
            return

        ids = ids.flatten()
        for idx, marker_id in enumerate(ids):
            marker_corners = corners[idx]
            if marker_corners is None or len(marker_corners) == 0:
                continue

            # Estimate pose
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [marker_corners], marker_size, matrix_coefficients, distortion_coefficients
            )
            if tvecs is None or len(tvecs) == 0:
                continue

            tvec = tvecs[0][0]
            # Smooth translation vectors
            if marker_id in prev_tvecs:
                tvec = ALPHA * tvec + (1 - ALPHA) * prev_tvecs[marker_id]
            prev_tvecs[marker_id] = tvec

            x, y, z = tvec
            # Overlay pose text at marker center
            corners_arr = marker_corners.reshape((4, 2))
            cX = int(corners_arr[:, 0].mean())
            cY = int(corners_arr[:, 1].mean())

            pose_text = f"ID:{int(marker_id)} X:{x:.2f}m Y:{y:.2f}m Z:{z:.2f}m"
            cv2.putText(frame, pose_text, (cX - 80, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if marker_id == drop_zoneID:
                distance = np.linalg.norm(tvec)
                angle_x = np.degrees(np.arctan2(x, z))
                dz_text = f"DZ:{distance:.2f}m Ang:{angle_x:.1f}\u00b0"
                cv2.putText(frame, dz_text, (cX - 80, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    except Exception:
        print("-- Error in pose_estimation:")
        traceback.print_exc()

# --- Camera & Recorder Setup ---
ARUCO_DICT = {
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250
}
aruco_type = "DICT_6X6_250"

intrinsic_camera = np.array([[933.15867, 0, 657.59], [0, 933.1586, 400.36993], [0, 0, 1]])
distortion = np.array([-0.43948, 0.18514, 0, 0])

drop_zoneID = 1
marker_size = 0.06611  # meters

picam2 = Picamera2()
# configure with manual exposure for faster shutter
camera_config = picam2.create_preview_configuration(
    raw={"size": (1640, 1232)},
    main={"format": 'RGB888', "size": (640, 480)}
)
picam2.configure(camera_config)
# manual controls: lower exposure time, increase gain
picam2.set_controls({
    "FrameRate": 60.0   #FPS
})

print("-- Camera starting...")
picam2.start()
# allow sensor to adjust
time.sleep(2)
print("-- Camera started")

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20.0
frame_size = (640, 480)
out = cv2.VideoWriter('/home/rtxcapstone/Desktop/testVideo.mp4', fourcc, fps, frame_size)

try:
    while True:
        img = picam2.capture_array()
        h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            stabilized = img.copy()
        else:
            # feature-based stabilization
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
            M = None
            if prev_pts is not None:
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
                # select good matches
                idx = status.reshape(-1) == 1
                if np.count_nonzero(idx) >= 6:
                    src = prev_pts[idx]
                    dst = curr_pts[idx]
                    M, _ = cv2.estimateAffinePartial2D(src, dst)
            if M is not None:
                stabilized = cv2.warpAffine(img, M, (w, h))
            else:
                stabilized = img.copy()

        prev_gray = gray

        # detect & overlay on stabilized frame
        pose_estimation(stabilized, ARUCO_DICT[aruco_type], intrinsic_camera, distortion, drop_zoneID, marker_size)

        cv2.imshow("Stabilized Feed", stabilized)
        out.write(stabilized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("-- Interrupted by user.")
finally:
    print("-- Cleaning up...")
    picam2.stop()
    picam2.close()
    out.release()
    cv2.destroyAllWindows()

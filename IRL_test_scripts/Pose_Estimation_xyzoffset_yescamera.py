import numpy as np
import cv2
import time
import traceback
from picamera2 import Picamera2

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, drop_zoneID, marker_size):
    if frame is None or frame.size == 0:
        return

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

        if ids is not None:
            ids = ids.flatten()
            for marker_index, marker_id in enumerate(ids):
                if marker_index >= len(corners):
                    continue

                marker_corners = corners[marker_index]
                if marker_corners is None or len(marker_corners) == 0:
                    continue

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [marker_corners], marker_size, matrix_coefficients, distortion_coefficients
                )

                if tvecs is not None and len(tvecs) > 0:
                    tvec = tvecs[0][0]
                    x, y, z = tvec
                    print(f"-- [Marker ID: {int(marker_id)}] X: {x:.3f} m | Y: {y:.3f} m | Z: {z:.3f} m")

                    if marker_id == drop_zoneID:
                        distance = np.linalg.norm(tvec)
                        angle_x = np.degrees(np.arctan2(x, z))
                        print(f"--   ↳ Drop-Zone → Distance: {distance:.2f} m | Angle X: {angle_x:.2f}°")

    except Exception:
        print("-- Unexpected error in pose_estimation:")
        traceback.print_exc()

# --- Camera & Settings ---
ARUCO_DICT = {
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250
}
aruco_type = "DICT_6X6_250"

intrinsic_camera = np.array([[933.15867, 0, 657.59], [0, 933.1586, 400.36993], [0, 0, 1]])
distortion = np.array([-0.43948, 0.18514, 0, 0])

drop_zoneID = 1
marker_size = 0.06611  # meters

print("-- Initializing camera...")
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    raw={"size": (1640, 1232)},
    main={"format": 'RGB888', "size": (640, 480)}
))
picam2.start()
time.sleep(2)
print("-- Camera started")

try:
    while True:
        img = picam2.capture_array()
        pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion, drop_zoneID, marker_size)
        cv2.imshow("Live Feed", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("-- Interrupted by user.")
finally:
    print("-- Cleaning up...")
    picam2.stop()
    picam2.close()
    cv2.destroyAllWindows()

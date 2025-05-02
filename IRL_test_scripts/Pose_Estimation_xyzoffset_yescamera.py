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

                # Estimate pose
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [marker_corners], marker_size, matrix_coefficients, distortion_coefficients
                )

                if tvecs is not None and len(tvecs) > 0:
                    tvec = tvecs[0][0]
                    x, y, z = tvec

                    # Calculate center of marker in image
                    corners_arr = marker_corners.reshape((4, 2))
                    cX = int(corners_arr[:, 0].mean())
                    cY = int(corners_arr[:, 1].mean())

                    # Overlay pose information
                    text = f"ID:{int(marker_id)} X:{x:.2f}m Y:{y:.2f}m Z:{z:.2f}m"
                    cv2.putText(frame, text, (cX - 60, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if marker_id == drop_zoneID:
                        distance = np.linalg.norm(tvec)
                        angle_x = np.degrees(np.arctan2(x, z))
                        dz_text = f"DZ Dist:{distance:.2f}m Ang:{angle_x:.2f}\u00b0"
                        cv2.putText(frame, dz_text, (cX - 60, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20.0
frame_size = (640, 480) 
out = cv2.VideoWriter('aruco_vision.mp4', fourcc, fps, frame_size)

try:
    while True:
        img = picam2.capture_array()
        pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion, drop_zoneID, marker_size)

        # Display
        cv2.imshow("Live Feed", img)
        # Write frame to file
        out.write(img)

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

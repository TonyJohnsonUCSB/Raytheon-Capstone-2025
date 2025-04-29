import time
import cv2
import numpy as np
from picamera2 import Picamera2

# Helper functions
def compute_velocity(pos):
    if abs(pos) < 0.01:
        return 0
    elif abs(pos) < 0.3:
        return -np.sign(pos)
    else:
        return -pos

# --- ArUco & camera calibration setup ---
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

camera_matrix = np.array([[933.15867, 0, 657.59],
                          [0, 933.1586, 400.36993],
                          [0, 0, 1]])
dist_coeffs = np.array([-0.43948, 0.18514, 0, 0])

marker_size = 0.06611  # meters
drop_zone_id = 1

# Control gains & limits
k_x = 1.0
k_y = 1.0
max_vel = 1.0  # m/s

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    raw={"size": (1640, 1232)},
    main={"format": 'RGB888', "size": (1280, 960)}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Preview", 1280, 960)
print("Starting marker tracking. Press 'q' to exit.")

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=parameters)
        vel_east = vel_north = 0.0

        if ids is not None:
            ids = ids.flatten()
            for idx, marker_id in enumerate(ids):
                if marker_id != drop_zone_id:
                    continue

                cv2.aruco.drawDetectedMarkers(frame, [corners[idx]])

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[idx]], marker_size, camera_matrix, dist_coeffs
                )
                xyadjust = np.array([-0.06204336, -0.02906156, 0])
                tvec = np.array(tvecs[0][0])
                x_cam, y_cam, z_cam = tvec
                
                if abs(x_cam) < 0.01:
                    vel_east = 0
                elif abs(x_cam) < 0.3:
                    vel_east = -np.sign(x_cam)
                else:
                    vel_east = -x_cam
                    
                if abs(y_cam) < 0.01:
                    vel_north = 0
                elif abs(y_cam) < 0.3:
                    vel_north = np.sign(y_cam)
                else:
                    vel_north = y_cam

                break

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2

            cv2.putText(frame, f"x = {x_cam:.4f} m", (10, 40), font, font_scale, (0, 255, 0), thickness)
            cv2.putText(frame, f"y = {y_cam:.4f} m", (10, 90), font, font_scale, (0, 255, 0), thickness)
            cv2.putText(frame, f"z = {z_cam:.4f} m", (10, 140), font, font_scale, (0, 255, 0), thickness)
            cv2.putText(frame, f"vel_east = {vel_east:.4f} m/s", (10, 190), font, font_scale, (255, 0, 0), thickness)
            cv2.putText(frame, f"vel_north = {vel_north:.4f} m/s", (10, 240), font, font_scale, (255, 0, 0), thickness)

        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            cv2.putText(frame, "No marker detected", (10, 60), font, font_scale, (0, 0, 255), thickness)

        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if ids is not None and drop_zone_id in ids:
            print(f"Pose: x={x_cam:.3f}, y={y_cam:.3f}, z={z_cam:.3f}\n"
                  f"Setpoints → vel_east: {vel_east:.4f}, vel_north: {vel_north:.4f}")
        else:
            print("No marker detected. Setpoints all zero.")

        time.sleep(0.05)

except KeyboardInterrupt:
    pass

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Tracking stopped.")

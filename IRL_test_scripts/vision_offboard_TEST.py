import time
import cv2
import numpy as np
from picamera2 import Picamera2

# Helper functions
def compute_velocity(pos):
    if abs(pos) < 0.01:
        return 0
    elif abs(pos) < 0.3:
        return np.sign(pos)  # +1 or -1
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
    main={"format": 'RGB888', "size": (1280, 960)}  # <-- larger main frame
)
picam2.configure(config)
picam2.start()
time.sleep(2)

cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)  # <-- allow resizing
cv2.resizeWindow("Preview", 1280, 960)          # <-- set window size
print("Starting marker tracking. Press 'q' to exit.")

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=parameters)
        vx = vy = vz = 0.0

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
                #tvec -= xyadjust
                x_cam, y_cam, z_cam = tvec

                vx = compute_velocity(x_cam)
                vy = compute_velocity(y_cam)
                break

            # Display pose and velocities separately
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2

            cv2.putText(frame, f"x = {x_cam:.4f} m", (10, 40), font, font_scale, (0, 255, 0), thickness)
            cv2.putText(frame, f"y = {y_cam:.4f} m", (10, 90), font, font_scale, (0, 255, 0), thickness)
            cv2.putText(frame, f"z = {z_cam:.4f} m", (10, 140), font, font_scale, (0, 255, 0), thickness)
            cv2.putText(frame, f"vx = {vx:.4f} m/s", (10, 190), font, font_scale, (255, 0, 0), thickness)
            cv2.putText(frame, f"vy = {vy:.4f} m/s", (10, 240), font, font_scale, (255, 0, 0), thickness)
            cv2.putText(frame, f"vz = {vz:.4f} m/s", (10, 290), font, font_scale, (255, 0, 0), thickness)

        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            cv2.putText(frame, "No marker detected", (10, 60), font, font_scale, (0, 0, 255), thickness)

        # show preview
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # terminal log
        if ids is not None and drop_zone_id in ids:
            print(f"Pose: x={x_cam:.3f}, y={y_cam:.3f}, z={z_cam:.3f}\n"
                  f"Setpoints → vx: {vx:.4f}, vy: {vy:.4f}, vz: {vz:.4f}")
        else:
            print("No marker detected. Setpoints all zero.")

        time.sleep(0.05)

except KeyboardInterrupt:
    pass

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Tracking stopped.")

import time
import cv2
import numpy as np
from picamera2 import Picamera2

# Helper functions
def compute_velocity(pos):
    if 0.01 < pos < 1:
        return 1
    elif pos > 0.01:
        return -pos
    else:
        return 0

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
    main={"format": 'RGB888', "size": (640, 480)}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

cv2.namedWindow("Preview", cv2.WINDOW_AUTOSIZE)
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

                # draw detected marker
                cv2.aruco.drawDetectedMarkers(frame, [corners[idx]])

                # pose estimation
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[idx]], marker_size, camera_matrix, dist_coeffs
                )
                xyadjust = np.array([-0.06204336, -0.02906156, 0])
                tvec = np.array(tvecs[0][0])
                tvec -= xyadjust
                x_cam, y_cam, z_cam = tvec
                
                vx = compute_velocity(x_cam)
                vy = compute_velocity(y_cam)
                break

            # overlay pose and setpoints on frame
            text = f"x={x_cam:.4f}m y={y_cam:.4f}m  →  vy={vy:.4f}, vz={vz:.4f}"
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No marker detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # show preview
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # terminal log
        if ids is not None and drop_zone_id in ids:
            print(f"Pose: x={x_cam:.3f}, y={y_cam:.3f}, z={z_cam:.3f} m | "
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

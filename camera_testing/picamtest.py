import numpy as np
import cv2
from picamera2 import Picamera2
import time
import logging

logging.basicConfig(level=logging.DEBUG)

# Draw Axis on Markers
def draw_axis(img, rvec, tvec, camera_matrix, dist_coeffs, length):
    axis_points = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]]).reshape(-1, 3)
    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    img_points = np.round(img_points).astype(int)

    img = cv2.line(img, tuple(img_points[0].ravel()), tuple(img_points[1].ravel()), (0, 0, 255), 2)  # x-axis
    img = cv2.line(img, tuple(img_points[0].ravel()), tuple(img_points[2].ravel()), (0, 255, 0), 2)  # y-axis
    img = cv2.line(img, tuple(img_points[0].ravel()), tuple(img_points[3].ravel()), (255, 0, 0), 2)  # z-axis
    return img

# Display ArUco Markers
def aruco_display(corners, ids, rejected, image):
    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            topRight = tuple(map(int, topRight))
            bottomRight = tuple(map(int, bottomRight))
            bottomLeft = tuple(map(int, bottomLeft))
            topLeft = tuple(map(int, topLeft))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            cX, cY = int((topLeft[0] + bottomRight[0]) / 2), int((topLeft[1] + bottomRight[1]) / 2)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            logging.info(f"[Inference] ArUco marker ID: {markerID}")
    return image

# Pose Estimation
def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    frame = aruco_display(corners, ids, rejected_img_points, frame)

    if ids is not None:
        for marker_index in range(len(ids)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[marker_index], 0.254, matrix_coefficients, distortion_coefficients)
            draw_axis(frame, rvec, tvec, matrix_coefficients, distortion_coefficients, 0.1)
    return frame

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(raw={"size": (1640, 1232)}, main={"format": 'RGB888', "size": (640, 480)}))
picam2.start()
time.sleep(2)

# Camera calibration parameters
intrinsic_camera = np.array([[933.15867, 0, 657.59], [0, 933.1586, 400.36993], [0, 0, 1]])
distortion = np.array([-0.43948, 0.18514, 0, 0])

# ArUco dictionary type
ARUCO_DICT = {
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250
}
aruco_type = "DICT_6X6_250"

try:
    while True:
        img = picam2.capture_array()
        output = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion)

        # Always display GUI
        cv2.imshow("Estimated Pose", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    picam2.close()
    cv2.destroyAllWindows()

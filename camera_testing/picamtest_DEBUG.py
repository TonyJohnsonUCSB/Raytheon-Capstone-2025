#!/usr/bin/env python3
"""
Pose estimation using Picamera2 and ArUco markers.
This version loads the camera calibration coefficients from a YAML file.
Usage:
    python3 pose_estimation_from_yaml.py
"""

import numpy as np
import cv2
from picamera2 import Picamera2
import time
import logging
import yaml  # Ensure you have PyYAML installed: pip install pyyaml

logging.basicConfig(level=logging.DEBUG)

# Function to load calibration coefficients from a YAML file.
def load_calibration(filename):
    try:
        with open(filename, "r") as f:
            calib_data = yaml.safe_load(f)
        camera_matrix = np.array(calib_data["camera_matrix"])
        # The YAML file may have the distortion coefficients under a key like "distortion_coefficients"
        # Adjust the key name if necessary.
        dist_coeffs = np.array(calib_data["distortion_coefficients"])
        logging.info(f"Loaded camera matrix: {camera_matrix}")
        logging.info(f"Loaded distortion coefficients: {dist_coeffs}")
        return camera_matrix, dist_coeffs
    except Exception as e:
        logging.error(f"Failed to load calibration file {filename}: {e}")
        raise

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
    if len(corners) > 0 and ids is not None:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners_reshaped = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners_reshaped
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

            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            logging.debug(f"[Inference] ArUco marker ID: {markerID}")
    return image

# Pose Estimation
def pose_estimation(frame, aruco_dict_type, camera_matrix, dist_coeffs):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    frame = aruco_display(corners, ids, rejected_img_points, frame)

    if ids is not None:
        for marker_index in range(len(ids)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[marker_index],
                                                                 0.175, camera_matrix, dist_coeffs)
            frame = draw_axis(frame, rvec, tvec, camera_matrix, dist_coeffs, 0.1)
    return frame

# Main script
def main():
    # Load calibration coefficients from YAML.
    calibration_file = "calibration_output.yaml"
    camera_matrix, dist_coeffs = load_calibration(calibration_file)

    # ArUco dictionary type selection.
    ARUCO_DICT = {
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250
    }
    aruco_type = "DICT_6X6_250"

    # Initialize Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(raw={"size": (1640, 1232)},
                                                 main={"format": 'RGB888', "size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    try:
        while True:
            img = picam2.capture_array()
            output = pose_estimation(img, ARUCO_DICT[aruco_type], camera_matrix, dist_coeffs)
            cv2.imshow("Estimated Pose", output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

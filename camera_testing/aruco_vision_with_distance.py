#!/usr/bin/env python3
"""
Pose estimation using Picamera2 and ArUco markers.
This version loads calibration coefficients from a YAML file and displays distances in inches.
Usage:
    python3 pose_estimation_from_yaml.py
"""

import numpy as np
import cv2
import sys
import time
from picamera2 import Picamera2
import yaml  # Requires: pip install pyyaml
import logging

logging.basicConfig(level=logging.DEBUG)

def load_calibration(filename):
    """
    Load the calibration coefficients from a YAML file.
    Expects the YAML file to have keys:
      - "camera_matrix": a 3x3 matrix
      - "distortion_coefficients": a list/array of distortion coefficients
    """
    try:
        with open(filename, "r") as f:
            calib_data = yaml.safe_load(f)
        camera_matrix = np.array(calib_data["camera_matrix"])
        dist_coeffs = np.array(calib_data["distortion_coefficients"])
        logging.info(f"Loaded camera matrix:\n{camera_matrix}")
        logging.info(f"Loaded distortion coefficients:\n{dist_coeffs}")
        return camera_matrix, dist_coeffs
    except Exception as e:
        logging.error(f"Failed to load calibration file {filename}: {e}")
        sys.exit(1)

def draw_axis(img, rvec, tvec, camera_matrix, dist_coeffs, length):
    """
    Draw 3D axes on the marker for visualization.
    """
    # Define the axis points in 3D space
    axis_points = np.float32([[0, 0, 0], [length, 0, 0],
                               [0, length, 0], [0, 0, length]]).reshape(-1, 3)
    # Project the 3D axis points to 2D image points
    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    # Convert img_points to integers and reshape
    img_points = np.int32(img_points).reshape(-1, 2)
    # Draw the axes on the image
    img = cv2.line(img, tuple(img_points[0]), tuple(img_points[1]), (0, 0, 255), 2)  # x-axis (red)
    img = cv2.line(img, tuple(img_points[0]), tuple(img_points[2]), (0, 255, 0), 2)  # y-axis (green)
    img = cv2.line(img, tuple(img_points[0]), tuple(img_points[3]), (255, 0, 0), 2)  # z-axis (blue)
    return img

def aruco_display(corners, ids, rejected, image, drop_zoneID):  
    if len(corners) > 0 and ids is not None:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            marker_corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = marker_corners
            
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # Draw the bounding box of the ArUco marker
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            
            # Calculate and draw the center of the ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            
            # Check if the marker ID equals the drop-zone ID
            if markerID == drop_zoneID:
                status = "Drop-Off"
                color = (0, 0, 255)  # Red for drop-off
            else:
                status = "Non-Drop-Off"
                color = (255, 0, 0)  # Blue for non-drop-off
            
            # Display the marker ID and status
            label = f"ID: {markerID} - {status}"
            cv2.putText(image, label, (topLeft[0], topLeft[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            print(f"[Inference] ArUco marker {label}")
            
    return image

def pose_estimation(frame, aruco_dict_type, camera_matrix, dist_coeffs, drop_zoneID, marker_size):
    drop_zone_found = False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    frame = aruco_display(corners, ids, rejected_img_points, frame, drop_zoneID)
    
    (height, width) = frame.shape[:2]
    image_center = (int(width / 2), int(height / 2))
    cv2.circle(frame, image_center, radius=5, color=(255, 0, 0), thickness=-1)
    
    if ids is not None:
        for marker_index, marker_id in enumerate(ids):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[marker_index],
                                                                 marker_size, camera_matrix, dist_coeffs)
            frame = draw_axis(frame, rvec, tvec, camera_matrix, dist_coeffs, 0.1)
            
            if marker_id == drop_zoneID:
                drop_zone_found = True
                tvec_flat = np.squeeze(tvec)
                # tvec_flat[2] is the distance in meters; convert to inches (1 m = 39.37 inches)
                distance_m = np.linalg.norm(tvec)
                distance_in = distance_m * 39.37
                angle_x = np.degrees(np.arctan(tvec_flat[0] / tvec_flat[2]))
                
                cv2.putText(frame, f"Distance: {distance_in:.2f} in", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Angle: {angle_x:.2f} deg", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                drop_zone_found = False
    return frame, drop_zone_found

def main():
    # Define ArUco dictionary types and choose one.
    ARUCO_DICT = {
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250  
    }
    aruco_type = "DICT_6X6_250"

    # Load calibration coefficients from YAML instead of hardcoding them.
    calibration_file = "calibration_output.yaml"
    intrinsic_camera, distortion = load_calibration(calibration_file)

    # Initialize Picamera2
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(raw={"size": (1640, 1232)},
                                                         main={"format": 'RGB888', "size": (640, 480)}))
    picam2.start()
    time.sleep(2)

    drop_zoneID = 1
    marker_size = 0.254  # Size of the physical marker in meters

    try:
        while True:
            img = picam2.capture_array()
            output, found = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera,
                                            distortion, drop_zoneID, marker_size)
            cv2.imshow("Estimated Pose", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

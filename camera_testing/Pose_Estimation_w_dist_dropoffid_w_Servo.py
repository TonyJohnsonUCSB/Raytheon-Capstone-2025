#!/usr/bin/env python3 

import numpy as np
import cv2
import sys
import time
from picamera2 import Picamera2
from adafruit_pca9685 import PCA9685
from board import SCL, SDA
import busio

# Initialize PCA9685
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 50  # Set frequency to 50Hz for servos

# Servo configuration
SERVO_CHANNEL = 0  # Channel on PCA9685 where the servo is connected
SERVO_MIN = 150    # Minimum pulse length for the servo (adjust as needed)
SERVO_MAX = 600    # Maximum pulse length for the servo (adjust as needed)

def set_servo_angle(channel, angle):
    """
    Set the servo angle using PCA9685.
    :param channel: PCA9685 channel where the servo is connected.
    :param angle: Desired angle in degrees (-90 to 90).
    """
    pulse = int(SERVO_MIN + (angle + 90) * (SERVO_MAX - SERVO_MIN) / 180)
    pca.channels[channel].duty_cycle = pulse

def draw_axis(img, rvec, tvec, camera_matrix, dist_coeffs, length):
    """
    Draw 3D axes on the marker for visualization.
    """
    axis_points = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]]).reshape(-1, 3)
    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    img_points = np.int32(img_points).reshape(-1, 2)
    img = cv2.line(img, tuple(img_points[0]), tuple(img_points[1]), (0, 0, 255), 2)  # x-axis (red)
    img = cv2.line(img, tuple(img_points[0]), tuple(img_points[2]), (0, 255, 0), 2)  # y-axis (green)
    img = cv2.line(img, tuple(img_points[0]), tuple(img_points[3]), (255, 0, 0), 2)  # z-axis (blue)
    return img

def aruco_display(corners, ids, rejected, image, drop_zoneID):  
    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            if markerID == drop_zoneID:
                status = "Drop-Off"
                color = (0, 0, 255)
            else:
                status = "Non-Drop-Off"
                color = (255, 0, 0)
            label = f"ID: {markerID} - {status}"
            cv2.putText(image, label, (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            print(f"[Inference] ArUco marker {label}")
    return image

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, drop_zoneID, marker_size):
    drop_zone_found = False
    angle_y = None
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
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[marker_index], marker_size, matrix_coefficients, distortion_coefficients)
            draw_axis(frame, rvec, tvec, matrix_coefficients, distortion_coefficients, 0.1)
            if marker_id == drop_zoneID:
                drop_zone_found = True
                distance = np.linalg.norm(tvec)
                cv2.putText(frame, f"Distance to Drop-Zone: {distance:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                tvec_flat = np.squeeze(tvec)
                angle_x = np.degrees(np.arctan(tvec_flat[0] / tvec_flat[2]))
                angle_y = np.degrees(np.arctan(tvec_flat[1] / tvec_flat[2]))
                cv2.putText(frame, f"Angle: {angle_x:.2f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                drop_zone_found = False
    return frame, drop_zone_found, angle_y

def move_camera(angle_y):
    """
    Will center camera over marker using the servo
    """
    global detect_time
    global last_angle
    if 'detect_time' not in globals():
        detect_time = start_time
    if angle_y is not None:
        set_servo_angle(SERVO_CHANNEL, angle_y + last_angle)
        last_angle = angle_y + last_angle
        detect_time = time.time()
    elif angle_y is None and time.time() - detect_time < 5:
        set_servo_angle(SERVO_CHANNEL, last_angle)
    else:
        set_servo_angle(SERVO_CHANNEL, default_pos)

ARUCO_DICT = {
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250  
}
aruco_type = "DICT_6X6_250"

intrinsic_camera = np.array(((933.15867, 0, 657.59),(0,933.1586, 400.36993),(0,0,1)))
distortion = np.array((-0.43948,0.18514,0,0))

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(raw={"size": (1640, 1232)}, main={"format": 'RGB888', "size": (640, 480)}))
picam2.start()
time.sleep(2)

drop_zoneID = 1
marker_size = 0.06611
angle_y = None
default_pos = 0
last_angle = 0
start_time = time.time()

try:
    while True:
        img = picam2.capture_array()
        output, _, angle_y = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion, drop_zoneID, marker_size)
        cv2.imshow("Estimated Pose", output)
        move_camera(angle_y)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.stop()
    picam2.close()
    cv2.destroyAllWindows()
    pca.deinit()
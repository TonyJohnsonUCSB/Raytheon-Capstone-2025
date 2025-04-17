#!/usr/bin/env python3 

import numpy as np
import cv2
import sys
from time import sleep
import time
from picamera2 import Picamera2
from adafruit_pca9685.pca9685 import PCA9685

from board import SCL, SDA
import busio

# Initialize PCA9685
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 50  # Set frequency to 50Hz for servos

# Servo configuration
SERVO_CHANNEL = 0  # Channel on PCA9685 where the servo is connected
SERVO_MIN = 600    # Minimum pulse length for the servo (adjust as needed)
SERVO_MAX = 2400    # Maximum pulse length for the servo (adjust as needed)

def set_servo_angle(channel, angle):
    """
    Set the servo angle using PCA9685.
    :param channel: PCA9685 channel where the servo is connected.
    :param angle: Desired angle in degrees (-90 to 90).
    """
    pulse = int(SERVO_MIN + (angle + 90) * (SERVO_MAX - SERVO_MIN) / 180)
    duty = int(pulse / 20000 * 65535)
    pca.channels[channel].duty_cycle = duty

# Servo sweep function
def servo_sweep(channel):
    """
    Sweep the servo from its minimum to maximum position and back to the default position.
    :param channel: PCA9685 channel where the servo is connected.
    """
    print("Performing servo sweep...")
    # Sweep from minimum to maximum
    for angle in range(-90, 91, 5):  # Increment by 10 degrees
        set_servo_angle(channel, angle)
        time.sleep(0.05)  # Small delay for smooth movement

    # Sweep back from maximum to minimum
    for angle in range(90, -91, -5):  # Decrement by 10 degrees
        set_servo_angle(channel, angle)
        time.sleep(0.05)

    # Return to default position
    set_servo_angle(channel, default_pos)
    print("Servo sweep complete.")

def draw_axis(img, rvec, tvec, camera_matrix, dist_coeffs, length):
    """
    Draw 3D axes on the marker for visualization.
    """
    # Define the axis points in 3D space
    axis_points = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]]).reshape(-1, 3)

    # Project the 3D axis points to 2D image points
    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)

    # Convert img_points to integers and reshape
    img_points = np.int32(img_points).reshape(-1, 2)

    # Draw the axes on the image
    img = cv2.line(img, tuple(img_points[0]), tuple(img_points[1]), (0, 0, 255), 2)  # x-axis (red)
    img = cv2.line(img, tuple(img_points[0]), tuple(img_points[2]), (0, 255, 0), 2)  # y-axis (green)
    img = cv2.line(img, tuple(img_points[0]), tuple(img_points[3]), (255, 0, 0), 2)  # z-axis (blue)

    return img

# Function to display markers in images
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

            # Draw the bounding box of the ArUco marker
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            
            # Calculate and draw the center of the ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            
            # Check if the marker ID is 1 (drop-off)
            if markerID == drop_zoneID:
                status = "Drop-Off"
                color = (0, 0, 255)  # Red color for drop-off
            else:
                status = "Non-Drop-Off"
                color = (255,0, 0)  # Green color for non-drop-off
            
            # Display the marker ID and status
            label = f"ID: {markerID} - {status}"
            cv2.putText(image, label, (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
            print(f"[Inference] ArUco marker {label}")
            
    return image

# In reality we just need pose estimation the other functions are to visualize what the code is doing
# All we would really need is identification of the correct marker
def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, drop_zoneID, marker_size):
    drop_zone_found = False
    angle_y = None  # Initialize angle_y to None ###############################################################
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Process image to black and white
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)  # Specify ArUco library
    parameters = cv2.aruco.DetectorParameters_create()
    
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    # Overlay the detected markers on the frame (assuming aruco_display is defined)
    frame = aruco_display(corners, ids, rejected_img_points, frame, drop_zoneID)
    
    # Get the image center (frame dimensions: height, width)
    (height, width) = frame.shape[:2]
    image_center = (int(width / 2), int(height / 2))
    
    # Draw a small circle at the center of the frame
    cv2.circle(frame, image_center, radius=5, color=(255, 0, 0), thickness=-1)
    
    if ids is not None:
        for marker_index, marker_id in enumerate(ids):
            # Estimate pose for the marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[marker_index], marker_size, matrix_coefficients, distortion_coefficients)
            
            # Draw the axes for the marker (assuming draw_axis is defined)
            draw_axis(frame, rvec, tvec, matrix_coefficients, distortion_coefficients, 0.1)
            
            if marker_id == drop_zoneID:
                drop_zone_found = True
                # Compute the distance from the camera to the marker
                distance = np.linalg.norm(tvec)
                # Display the distance on the frame
                cv2.putText(frame, f"Distance to Drop-Zone: {distance:.2f} m", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # Squeeze tvec to remove extra dimensions
                tvec_flat = np.squeeze(tvec)
                angle_x = np.degrees(np.arctan(tvec_flat[0] / tvec_flat[2]))
                angle_y = np.degrees(np.arctan(tvec_flat[1] / tvec_flat[2]))
                
                # Display the calculated angle on the frame
                cv2.putText(frame, f"Angle: {angle_x:.2f} deg", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                drop_zone_found = False
    return frame, drop_zone_found, angle_y

# To move the camera we need to extract the angle of the marker with rexpect to the center of the camera
def move_camera(angle_y):
    """
    Will center camera over marker using the servo
    """
    global detect_time  # Declare detect_time as global to persist its value across calls
    global last_angle  # Declare last_angle as global to persist its value across calls

    if 'detect_time' not in globals():
        detect_time = start_time  # Initialize detect_time with start_time if not already defined

    # Three cases that must be accounted for 
    # 1) Marker is not in Frame -> Do nothing stay at default position
    # 2) Marker is Found -> move camera accordingly 
    # 3) Marker is in frame but flickering -> We can use a timer to account for
    #    flickering meaning if marker is lost for a split second the servo won't trip up 
    #    by having a timer to actually make sure marker is not lost. and is just flickering
    
    # Angle Y is not None only when Marker is found 
    if angle_y is not None:  # If marker is found
        anle = angle_y +last_angle
        if angle > 180:
            angle = 180
        elif angle < 0:
            angle = 0
        set_servo_angle(SERVO_CHANNEL, angle_y + last_angle)  # Move servo x degrees from last position
        last_angle = angle_y + last_angle  # Save servo's last position
        detect_time = time.time()  # Update detect_time to the current time
    # elif angle_y is None and time.time() - detect_time < 5:  # If marker is flickering
        # set_servo_angle(SERVO_CHANNEL, last_angle)  # Keep servo at the last recorded angle before marker was lost
    # else:
        # set_servo_angle(SERVO_CHANNEL, default_pos)  # If marker is not found, stay at default position
                
            
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

# Variables
drop_zoneID = 1
marker_size = 0.254 #Size of physical marker in meters (10in)

#We will have a default Servo Position
angle_y = None
default_pos = 0
last_angle = 0
start_time = time.time()

# Perform servo sweep at the start
servo_sweep(SERVO_CHANNEL)

try:
    while True:
        img = picam2.capture_array()
        output,_, angle_y = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion, drop_zoneID, marker_size)

        # Always display GUI
        cv2.imshow("Estimated Pose", output)
        move_camera(angle_y)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    picam2.close()
    cv2.destroyAllWindows()
    pca.deinit()

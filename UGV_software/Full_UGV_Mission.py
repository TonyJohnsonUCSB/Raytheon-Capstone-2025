import numpy as np
import cv2
import sys
import time
from PIL import Image, ImageDraw
from IPython.display import clear_output
import math
import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from time import sleep
from picamera2 import Picamera2
from adafruit_pca9685.pca9685 import PCA9685
import yaml  # Requires: pip install pyyaml
from board import SCL, SDA
import busio

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

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, drop_zoneID, marker_size):
   
    distance = None
    angle_y = None
    angle_x = None
    ##### This Part of the code will be the one we need mostly for Raytheon This will give marker ids ########
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Processes image to black and white
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type) # Specifies ArUco library were using
    parameters = cv2.aruco.DetectorParameters()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    # This returns processed image with Arucos and their ids overlaid
    frame = aruco_display(corners, ids, rejected_img_points, frame, drop_zoneID)
    

    if ids is not None:
        for marker_index, marker_id in enumerate(ids):
            # Estimate pose for the marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[marker_index], marker_size, matrix_coefficients, distortion_coefficients)

            # Draw the axes for the marker
            draw_axis(frame, rvec, tvec, matrix_coefficients, distortion_coefficients, marker_size)
            
            if marker_id == drop_zoneID:
                drop_zone_found = True
                # Compute the distance from the camera to the marker
                distance = np.linalg.norm(tvec)
                flat_tvec = np.squeeze(tvec)
                x = flat_tvec[0]
                y = flat_tvec[1]
                z = flat_tvec[2]
                angle_y = np.degrees(np.arctan(y/z)) # important to move camera up and down
                angle_x =np.degrees(np.arctan(x / z)) #
                # Display the distance on the frame
                cv2.putText(frame, f"Distance to Drop-Zone: {distance:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Y Angle to Drop-Zone: {angle_y:.2f} degrees", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"X Angle to Drop-Zone: {angle_x:.2f} degrees", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
    return frame, distance, angle_y, angle_x

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
        
# PID controller class
class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

    def update(self, measurement):
        current_time = time.time()
        dt = current_time - self.prev_time if current_time - self.prev_time > 0 else 1e-16
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        u = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.prev_error = error
        self.prev_time = current_time
        return u

def set_servo_angle(channel, angle):
    """
    Set the servo angle using PCA9685.
    :param channel: PCA9685 channel where the servo is connected.
    :param angle: Desired angle in degrees (-90 to 90).
    """
    # Clamp angle to prevent out-of-range duty values
    angle = max(-90, min(90, angle))

    pulse = int(SERVO_MIN + (angle + 90) * (SERVO_MAX - SERVO_MIN) / 180)
    duty = int(pulse / 20000 * 65535)
    pca.channels[channel].duty_cycle = duty

ARUCO_DICT = {
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250  
}
aruco_type = "DICT_6X6_250"

# Camera Parameters
calibration_file = "calibration_output.yaml"
intrinsic_camera, distortion = load_calibration(calibration_file)
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(raw={"size": (1640, 1232)}, main={"format": 'RGB888', "size": (640, 480)}))
picam2.start()

# Marker Parameters
drop_zoneID = 1
marker_size = 0.254  # Size of physical marker in meters
drop_zone_found = False

# PID Setup for car
desired_distance = 1.0 # desired distance from the marker in meters
desired_lateral = 0.0  # marker centered horizontally
forward_pid = PIDController(kp=8.0, ki=0., kd=0.0, setpoint=desired_distance)
lateral_pid = PIDController(kp=8.0, ki=0, kd=0.0, setpoint=desired_lateral)
max_speed_forward = 2.24 # [m/s]
max_speed_angle = 100 #[deg/s]
forward_velocity = 0
lateral_velocity = 0

# Servo/Camera setup 
camera_desired_angle = 0 
camera_PID = PIDController(Kp = 0.75, Ki = 0.0, kd = 0.0, setpoint = camera_desired_angle)
SERVO_CHANNEL = 0  # Channel on PCA9685 where the servo is connected
SERVO_MIN = 600    # Minimum pulse length for the servo (adjust as needed)
SERVO_MAX = 2400    # Maximum pulse length for the servo (adjust as needed)
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 50  # Set frequency to 50Hz for servos
current_angle = 0
set_servo_angle(SERVO_CHANNEL,current_angle) #set servo to an initial angle

# Mission setup
TARGET_LATITUDE = 34.414893950 # degrees
TARGET_LONGITUDE = -119.843535639 # degrees
TARGET_ALTITUDE = 0                  # for rovers, altitude is usually set to ground level
TARGET_YAW = 0                       # desired heading in degrees
loop_marker = 0


async def main():
    # Connecting to Rover via USB (adjust port/baud as needed)
    print("Waiting for rover to connect via USB...")
    rover = System()
    await rover.connect(system_address=="serial:///dev/ttyAMA0:57600") # address for the raspberry pi using UART port

    print("Waiting for rover to connect...")
    async for state in rover.core.connection_state():
        if state.is_connected:
            print("Rover connected!")
            break

    # Waypoint Mission 
    # print("Waiting for global position estimate...")
    # async for health in rover.telemetry.health():
    #     # For a rover we only need the global position to be ok
    #     if health.is_global_position_ok:
    #         print("-- Global position estimate OK")
    #         break

    # print(f"-- Driving rover to waypoint: lat={TARGET_LATITUDE}, lon={TARGET_LONGITUDE}")
    # await rover.action.goto_location(TARGET_LATITUDE,
    #                                  TARGET_LONGITUDE,
    #                                  TARGET_ALTITUDE,
    #                                  TARGET_YAW)
  
    try:
        # Main computer vision loop
        while True:
            img = picam2.capture_array() #capture an image

            output, distance, angle_y, angle_x = pose_estimation(
                img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion, drop_zoneID, marker_size
            )

            if distance is not None and loop_marker == 0: #If the Aruco is found activate offboard mode and never enter this loop again
                loop_marker = 1 
                #    Send an initial setpoint (all zeros) to satisfy the PX4 requirement
                initial_velocity = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                await rover.offboard.set_velocity_body(initial_velocity)

                print("Starting offboard mode...")
                try:
                    await rover.offboard.start()
                except OffboardError as error:
                    print(f"Offboard start failed: {error._result.result}")
                    return

                print("Offboard Mode On")

            # Loop for tracking marker
            if distance is not None: #If ArUco is found
                # Update servo angle
                u = camera_PID.update(angle_y)    # we get u update from controller
                new_angle = current_angle + u     # update angle 
                
                #Bottleneck new angle to servo bounds
                if new_angle > 90:
                    new_angle = 90
                elif new_angle <-90:
                    new_angle = -90
            
                # Set the new servo angle
                set_servo_angle(SERVO_CHANNEL,new_angle)# Set the servo to updated angle
                current_angle = new_angle               # Update the current angle

                # Update forward velocity
                forward_velocity = forward_velocity - 1*forward_pid.update(distance)

                # Bottleneck forward velocity
                if forward_velocity > max_speed_forward:
                    forward_velocity = max_speed_forward
                elif forward_velocity < -max_speed_forward:
                    forward_velocity = -max_speed_forward
                
                # Update Lateral velocity 
                lateral_velocity = lateral_velocity -1*lateral_pid.update(angle_x)

                # Bottleneck Lateral Velocity
                if lateral_velocity > max_speed_angle:
                    lateral_velocity = max_speed_angle
                elif lateral_velocity < -max_speed_angle:
                    lateral_velocity = -max_speed_angle
                
                # Overlay velocities on the output image
                cv2.putText(output, f"forward velocity: {forward_velocity:.2f} m/s", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(output, f"lateral velocity: {lateral_velocity:.2f} deg/s", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                #print(f"Forward Velocity: {forward_velocity:.2f}, Lateral Velocity: {lateral_velocity:.2f}")
                cv2.putText(output, f"Distance to Drop-Zone: {distance:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(output, f"Y Angle to Drop-Zone: {angle_y:.2f} degrees", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(output, f"X Angle to Drop-Zone: {angle_x:.2f} degrees", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Send velocity commands via offboard command:
                velocity_command = VelocityBodyYawspeed(forward_velocity, 0, 0.0, lateral_velocity)
                await rover.offboard.set_velocity_body(velocity_command)

            else: #If marker is not found stop car
                initial_velocity = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                await rover.offboard.set_velocity_body(initial_velocity)
            
            cv2.imshow('Estimated Pose', output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()
        pca.deinit()

# Call the main coroutine using await (preferred in Jupyter)
await main()

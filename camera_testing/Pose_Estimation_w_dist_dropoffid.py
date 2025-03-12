import os
import time
import threading
import numpy as np
import cv2
import matplotlib.pyplot as plt
from picamera2 import Picamera2

# Ensure OpenCV uses the correct display backend
os.environ["QT_QPA_PLATFORM"] = "xcb"

def show_image():
    """ Continuously updates the OpenCV window with new frames. """
    cv2.namedWindow("Estimated Pose", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Estimated Pose", 100, 100)  # Ensures window is visible

    while True:
        if hasattr(show_image, "frame") and show_image.frame is not None:
            print("Displaying frame in OpenCV window...")  # Debugging
            cv2.imshow("Estimated Pose", show_image.frame)
        else:
            print("Waiting for first frame...")  # Debugging

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting OpenCV window thread...")
            break

    cv2.destroyAllWindows()

def show_fallback_image(image):
    """ Backup method to display image using Matplotlib if OpenCV fails. """
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
    print("Matplotlib display successful")

# In reality we just need pose estimation the other functions are to visualize what the code is doing
# All we would really need is identification of the correct marker
def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, drop_zoneID, marker_size):

    """ Detect and estimate pose of ArUco markers. """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)

    parameters = cv2.aruco.DetectorParameters()
    
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        ids = ids.flatten()
        for marker_index, marker_id in enumerate(ids):
            print(f"Processing marker ID {marker_id}")
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[marker_index], marker_size, matrix_coefficients, distortion_coefficients
            )
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.1)

            if marker_id == drop_zoneID:
                distance = np.linalg.norm(tvec)
                cv2.putText(frame, f"Distance to Drop-Zone: {distance:.2f} m", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# Define ArUco dictionary
ARUCO_DICT = {
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250  
}
aruco_type = "DICT_6X6_250"

# Camera intrinsic parameters
intrinsic_camera = np.array([[933.15867, 0, 657.59], 
                             [0, 933.1586, 400.36993], 
                             [0, 0, 1]], dtype=np.float32)

# Corrected distortion matrix
distortion = np.array([-0.43948, 0.18514, 0, 0, 0], dtype=np.float32).reshape(1, 5)

# Initialize Picamera2
picam2 = Picamera2()
print("Camera has been configured")
picam2.start()
time.sleep(2)

drop_zoneID = 1
marker_size = 0.254  # Size of the physical marker in meters

# Initialize and start OpenCV display thread
show_image.frame = None  # Initialize frame
thread = threading.Thread(target=show_image, daemon=True)
thread.start()

# Main loop for computer vision
print("Starting capture loop")
try:
    while True:
        print("Capturing image")
        img = picam2.capture_array()
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        print(f"Captured image shape: {img.shape}, dtype: {img.dtype}")

        output = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion, drop_zoneID, marker_size)

        # Resize the image before updating the display thread
        output = cv2.resize(output, (640, 480))

        # ✅ Force the frame update!
        show_image.frame = output  # Send frame to OpenCV display thread

        # If OpenCV is not showing images, use Matplotlib as a fallback
        if not hasattr(show_image, "frame") or show_image.frame is None:
            print("OpenCV window not updating, using Matplotlib fallback.")
            show_fallback_image(output)

        time.sleep(0.05)  # Allow time for the display to update

finally:
    picam2.stop()
    picam2.close()
    cv2.destroyAllWindows()

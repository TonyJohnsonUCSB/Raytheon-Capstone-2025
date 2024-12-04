import numpy as np
import cv2
from picamera2 import Picamera2, QtPreview

# Function to draw axes on markers
def draw_axis(img, rvec, tvec, camera_matrix, dist_coeffs, length):
    axis_points = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]]).reshape(-1, 3)
    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    img_points = np.round(img_points).astype(int)

    img = cv2.line(img, tuple(img_points[0].ravel()), tuple(img_points[1].ravel()), (0, 0, 255), 2)  # X-axis (red)
    img = cv2.line(img, tuple(img_points[0].ravel()), tuple(img_points[2].ravel()), (0, 255, 0), 2)  # Y-axis (green)
    img = cv2.line(img, tuple(img_points[0].ravel()), tuple(img_points[3].ravel()), (255, 0, 0), 2)  # Z-axis (blue)
    return img

# Function to display ArUco markers
def aruco_display(corners, ids, rejected, image):
    if len(corners) > 0:
        ids = ids.flatten()
        for (marker_corner, marker_id) in zip(corners, ids):
            corners = marker_corner.reshape((4, 2))
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
            cv2.putText(image, str(marker_id), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("[Inference] ArUco marker ID:", marker_id)
    return image

# Function for pose estimation
def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    frame = aruco_display(corners, ids, rejected_img_points, frame)

    if ids is not None:
        for marker_index, marker_id in enumerate(ids):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[marker_index], 0.175, matrix_coefficients, distortion_coefficients)
            draw_axis(frame, rvec, tvec, matrix_coefficients, distortion_coefficients, 0.1)
    return frame

# Camera intrinsic parameters
intrinsic_camera = np.array([[933.15867, 0, 657.59], [0, 933.1586, 400.36993], [0, 0, 1]])
distortion = np.array([-0.43948, 0.18514, 0, 0])

# ArUco dictionary
ARUCO_DICT = {"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL}
aruco_type = "DICT_ARUCO_ORIGINAL"

# Initialize Picamera2 with QtPreview
picam2 = Picamera2()
picam2.start_preview(QtPreview())

camera_config = picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
picam2.configure(camera_config)
picam2.start()

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV compatibility
        output = pose_estimation(frame, ARUCO_DICT[aruco_type], intrinsic_camera, distortion)
        cv2.imshow('Estimated Pose', output)

        # Exit loop on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()

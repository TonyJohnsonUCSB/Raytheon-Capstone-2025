import numpy as np
import cv2
import sys
import time
# Draws Axis on Markers
def draw_axis(img, rvec, tvec, camera_matrix, dist_coeffs, length):
    axis_points = np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)

    # Project axis points to the image plane
    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)

    # Convert image points to integers
    img_points = np.round(img_points).astype(int)

    # Draw lines
    img = cv2.line(img, tuple(img_points[0].ravel()), tuple(img_points[1].ravel()), (0,0,255), 2)  # x-axis (red)
    img = cv2.line(img, tuple(img_points[0].ravel()), tuple(img_points[2].ravel()), (0,255,0), 2)  # y-axis (green)
    img = cv2.line(img, tuple(img_points[0].ravel()), tuple(img_points[3].ravel()), (255,0,0), 2)  # z-axis (blue)

    return img

# Function to display markers in images
def aruco_display(corners, ids, rejected, image):  
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
			
			cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
			print("[Inference] ArUco marker ID: {}".format(markerID))
			
	return image
def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    ##### This Part of the code will be the one we need mostly for Raytheon This will give marker ids ########
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Processes image to black and white
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type) # Specifies ArUco library were using
    parameters = cv2.aruco.DetectorParameters()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    #######################
    
    # This returns processed image with Arucos and their ids overlaid
    frame = aruco_display(corners, ids, rejected_img_points, frame)
    
### This Part of the code was relevant to the project I did, unecessary for what we're doing ####
### We might need this to try to center drone over ArUco however you can get position of Aruco in frame #####
        #     distance_vector_nested_list = None
        #     x_rel = None
        #     y_rel = None
        #     rot_aboutZ = None
        #     center_x = None  # Default value
        #     center_y = None  # Default value
        #     marker_corners = None  # Initialize marker_corners outside of the loop

    if ids is not None:
        for marker_index, marker_id in enumerate(ids):
        #             if marker_id == 150:
        #                 marker_corners = corners[marker_index][0]
        #                 center_x = int(np.mean(marker_corners[:, 0]))
        #                 center_y = int(np.mean(marker_corners[:, 1]))
        #                 print("Pixel Coordinates of Marker 0 (Center):", (center_x, center_y))

        #                 # Calculate pixel length for the marker
        #                 pixel_length = np.linalg.norm(marker_corners[0] - marker_corners[1])  # distance between two adjacent corners
        #                 pixels_per_meter = pixel_length / 0.175
        #                 print("Pixel Length of Marker {}: {}".format(marker_id, pixel_length))
        #                 print("Pixels per Meter:", pixels_per_meter)

            # Estimate pose for the marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[marker_index], 0.175, matrix_coefficients, distortion_coefficients)

            # Draw the axes for the marker
            draw_axis(frame, rvec, tvec, matrix_coefficients, distortion_coefficients, 0.1)

        # This part was used to Compute distances between markers which I dont think we'll need
        #     if len(corners) > 1:
        #         print("There are two or more markers")
        #         # Estimate pose for the first marker
        #         rvec1, tvec1, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], 0.175, matrix_coefficients, distortion_coefficients)

        #         # Estimate pose for the second marker
        #         rvec2, tvec2, _ = cv2.aruco.estimatePoseSingleMarkers(corners[1], 0.175, matrix_coefficients, distortion_coefficients)

        #         # Calculating Relative Position Vectors
        #         distance_vector_nested_list = (tvec1 - tvec2)
        #         array = np.array(distance_vector_nested_list)
        #         flattened_array = array.flatten()
        #         for marker_index, marker_id in enumerate(ids):
        #             marker_corners = corners[marker_index][0]
        #             center_x = int(np.mean(marker_corners[:, 0]))
        #             center_y = int(np.mean(marker_corners[:, 1]))
        #         pixel_length = np.linalg.norm(marker_corners[0] - marker_corners[1])  # distance between two adjacent corners
        #         pixels_per_meter = pixel_length / 0.175
        #         x_rel = flattened_array[0] * pixels_per_meter  # Relative distance in x-direction of markers in Pixels
        #         y_rel = flattened_array[1] * pixels_per_meter
        #         print("Distance vector between markers:", array)

        #         # Calculating Relative Rotations
        #         Theta = angle_between_markers(rvec1.squeeze(), rvec2.squeeze())
        #         rot_aboutZ = -Theta[2]
        #         print(f"Relative Rotation Angle around Z-axis (degrees): {rot_aboutZ }")

        #         for i in range(len(corners)):
        #             rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.175, matrix_coefficients, distortion_coefficients)
        #             draw_axis(frame, rvec, tvec, matrix_coefficients, distortion_coefficients, 0.1)

        # #     return frame, (x_rel, y_rel), rot_aboutZ, (center_x, center_y)
    return frame
from PIL import Image, ImageDraw

ARUCO_DICT = {"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,}

aruco_type = "DICT_ARUCO_ORIGINAL"

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

arucoParams = cv2.aruco.DetectorParameters()


intrinsic_camera = np.array(((933.15867, 0, 657.59),(0,933.1586, 400.36993),(0,0,1)))
distortion = np.array((-0.43948,0.18514,0,0))


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# While loop for computer vision 
while cap.isOpened():
     
    ret, img = cap.read()
    
    # Calculates Position of Arucos and outputs the image with the axises and ids overlaid
    output = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion)
    
    # Shows Image 
    cv2.imshow('Estimated Pose',  output)
    
    # Code to exit image
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


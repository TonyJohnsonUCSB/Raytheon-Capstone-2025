'''This script is for generating data
1. Provide desired path to store images.
2. Press 'c' to capture image and display it.
3. Press any button to continue.
4. Press 'q' to quit.
'''

import cv2
import os

# Initialize the camera
camera = cv2.VideoCapture(0)
ret, img = camera.read()

# Ensure the path exists
path = "/Users/pauldiarte/Documents/GitHub/ME153/CalibrationImages(Paul)"
os.makedirs(path, exist_ok=True)
count = 0

while True:
    name = os.path.join(path, f"image_{count}.jpg")
    ret, img = camera.read()
    cv2.imshow("img", img)

    # Capture image if 'c' is pressed
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite(name, img)
        count += 1
        print(f"Image {name} saved.")

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()

from pathlib import Path

# Define the path as a string
path_string = "CalibrationImages(Paul)"

# Convert the path string to a Path object
root = Path(path_string).parent.absolute()

# Output the absolute path
print("Absolute root directory:", root)

# Newest Cell
import cv2
from cv2 import aruco
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

# Define the path as a string
# path_string = "/Users/pauldiarte/Documents/GitHub/ME153/CalibrationImages"

# # Convert the path string to a Path object
# root = Path(path_string).parent.absolute()

root_path = Path("/Users/pauldiarte/Documents/GitHub/ME153/CalibrationImages(Paul)")
root = root_path
# Output the absolute path
print("Absolute root directory:", root_path)

# Set this flsg True for calibrating camera and False for validating results real time
calibrate_camera = True

# Set path to the images
calib_imgs_path = root_path / "aruco_data"

# For validating results, show aruco board to camera.
aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_1000 )

#Provide length of the marker's side
markerLength = 3.634  # Here, measurement unit is centimetre.

# Provide separation between markers
markerSeparation = 0.47  # Here, measurement unit is centimetre.

# create arUco board
board = cv2.aruco.GridBoard((4,5), markerLength, markerSeparation, aruco_dict)

'''uncomment following block to draw and show the board'''
# img = board.generateImage((864,1080))
# cv2.imshow("aruco", img)

arucoParams = aruco.DetectorParameters()

img_list = []
if calibrate_camera == True:
    for idx in range(0, 50):
        fn = f"image_{idx}.jpg"
        print(idx, '', end='')
        file_path = os.path.join(root_path, fn)
        img = cv2.imread(file_path)
        if img is not None:
            img_list.append(img)
            h, w, c = img.shape
        else:
            print(f"\nError reading {fn}")

    print('Calibration images')
    print('Calibration images')
    counter, corners_list, id_list = [], [], []
    first = True
    for img in tqdm(img_list):
        img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
        if first == True:
            corners_list = corners
            id_list = ids
            first = False
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list,ids))
        counter.append(len(ids))
    print('Found {} unique markers'.format(np.unique(ids)))

    counter = np.array(counter)
    print ("Calibrating camera .... Please wait...")
    #mat = np.zeros((3,3), float)
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, img_gray.shape, None, None )

    print("Camera matrix is \n", mtx, "\n And is stored in calibration.yaml file along with distortion coefficients : \n", dist)
    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
    with open("calibration.yaml", "w") as f:
        yaml.dump(data, f)

else:
    camera = cv2.VideoCapture(0)
    ret, img = camera.read()

    with open('calibration.yaml') as f:
        loadeddict = yaml.load(f)
    mtx = loadeddict.get('camera_matrix')
    dist = loadeddict.get('dist_coeff')
    mtx = np.array(mtx)
    dist = np.array(dist)

    ret, img = camera.read()
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    h,  w = img_gray.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    pose_r, pose_t = [], []
    while True:
        ret, img = camera.read()
        img_aruco = img
        im_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        h,  w = im_gray.shape[:2]
        dst = cv2.undistort(im_gray, mtx, dist, None, newcameramtx)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, aruco_dict, parameters=arucoParams)
        #cv2.imshow("original", img_gray)
        if corners == None:
            print ("pass")
        else:

            ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, newcameramtx, dist) # For a board
            print ("Rotation ", rvec, "Translation", tvec)
            if ret != 0:
                img_aruco = aruco.drawDetectedMarkers(img, corners, ids, (0,255,0))
                img_aruco = aruco.drawAxis(img_aruco, newcameramtx, dist, rvec, tvec, 10)    # axis length 100 can be changed according to your requirement

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break;
        cv2.imshow("World co-ordinate frame axes", img_aruco)

cv2.destroyAllWindows()

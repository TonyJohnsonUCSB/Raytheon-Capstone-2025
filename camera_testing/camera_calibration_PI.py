#!/usr/bin/env python3
"""
Raspberry Pi Camera Calibration and Pose Estimation Script using Picamera2

Usage examples:
  - Capture calibration images:
      python3 script.py --mode capture
  - Calibrate the camera using captured images:
      python3 script.py --mode calibrate --images 50
  - Validate calibration (real-time pose estimation):
      python3 script.py --mode validate
"""

import numpy as np
import cv2
from picamera2 import Picamera2
import time
import os
import yaml
import logging
from pathlib import Path
from tqdm import tqdm

# logging.basicConfig(level=logging.DEBUG)

# Default image directory
IMAGE_DIR = "/home/rtxcapstone/Raytheon-Capstone-2025/camera_testing/calibration_images"

# ----------------- Mode Functions ----------------- #
def capture_images(save_dir=IMAGE_DIR):
    """
    Capture mode: Uses Picamera2 to display a live preview and save images when 'c' is pressed.
    """
    os.makedirs(save_dir, exist_ok=True)
    count = 0

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(raw={"size": (1640, 1232)},
                                                   main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Allow time for camera warm-up
    logging.info("Capture mode: Press 'c' to capture an image, 'q' to quit.")

    try:
        while True:
            frame = picam2.capture_array()
            cv2.imshow("Capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                filename = os.path.join(save_dir, f"image_{count}.jpg")
                cv2.imwrite(filename, frame)
                logging.info(f"Captured image saved as {filename}")
                count += 1
            elif key == ord('q') or count == 50:
                break
    finally:
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()
    logging.info("Image capture ended.")

def calibrate_camera_from_images(image_dir=IMAGE_DIR, num_images=50):
    """
    Calibration mode: Reads saved images, detects ArUco markers, performs camera calibration,
    and saves the calibration parameters to 'calibration.yaml'.
    """
    image_dir = Path(image_dir)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    markerLength = 3.726
    markerSeparation = 0.49
    board = cv2.aruco.GridBoard((4, 5), markerLength, markerSeparation, aruco_dict)
    arucoParams = cv2.aruco.DetectorParameters_create()

    img_list = []
    for idx in range(num_images):
        image_path = image_dir / f"image_{idx}.jpg"
        if image_path.exists():
            img = cv2.imread(str(image_path))
            if img is not None:
                img_list.append(img)
            else:
                logging.warning(f"Failed to load {image_path}")
        else:
            logging.warning(f"{image_path} does not exist.")

    if not img_list:
        logging.error("No images found for calibration.")
        return

    corners_list = []
    ids_list = []
    marker_counts = []
    for img in tqdm(img_list, desc="Detecting markers"):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=arucoParams)
        if ids is not None:
            corners_list.extend(corners)
            ids_list.extend(ids)
            marker_counts.append(len(ids))
        else:
            marker_counts.append(0)
    marker_counts = np.array(marker_counts)

    if len(marker_counts) == 0 or np.all(marker_counts == 0):
        logging.error("No markers detected in any image. Calibration failed.")
        return

    logging.info("Calibrating camera... Please wait.")
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
        corners_list, np.array(ids_list), marker_counts, board, gray.shape, None, None)

    if ret:
        logging.info("Calibration successful!")
        logging.info(f"Camera matrix:\n{mtx}")
        logging.info(f"Distortion coefficients:\n{dist}")
        calib_data = {'camera_matrix': mtx.tolist(), 'dist_coeff': dist.tolist()}
        with open("calibration.yaml", "w") as f:
            yaml.dump(calib_data, f)
        logging.info("Calibration data saved to calibration.yaml")
    else:
        logging.error("Calibration failed.")

def validate_pose(calib_file="calibration.yaml", marker_length=0.175):
    """
    Validation mode: Loads calibration data and runs a real-time pose estimation using Picamera2.
    """
    try:
        with open(calib_file, "r") as f:
            calib_data = yaml.safe_load(f)
        camera_matrix = np.array(calib_data['camera_matrix'])
        dist_coeffs = np.array(calib_data['dist_coeff'])
    except Exception as e:
        logging.error(f"Error loading calibration data: {e}")
        return

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(raw={"size": (1640, 1232)},
                                                   main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    logging.info("Validation mode: Press 'q' to quit.")

    try:
        while True:
            frame = picam2.capture_array()
            cv2.imshow("Validation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()
    logging.info("Validation ended.")

# ----------------- Main Entry Point ----------------- #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Raspberry Pi Camera Calibration and Pose Estimation using Picamera2")
    parser.add_argument('--mode', type=str, choices=['capture', 'calibrate', 'validate'], default='capture',
                        help="Mode: 'capture' to collect images, 'calibrate' to compute calibration, 'validate' for real-time pose estimation.")
    parser.add_argument('--images', type=int, default=50,
                        help="Number of images to use for calibration.")
    args = parser.parse_args()

    if args.mode == "capture":
        capture_images()
    elif args.mode == "calibrate":
        calibrate_camera_from_images(num_images=args.images)
    elif args.mode == "validate":
        validate_pose()

#!/usr/bin/env python3
"""
Debug script for camera calibration using pre-captured images with flattened detections.
This version:
  - Loads images from the specified directory.
  - Detects ArUco markers in each image.
  - Filters out images with fewer than a specified minimum number of markers.
  - Flattens detections into cumulative lists.
  - Converts marker counts and IDs to NumPy arrays with proper dtypes.
  - Sets calibration termination criteria.
  - Attempts camera calibration, prints debug info, and writes the outputs to a file.
Usage:
    python3 debug_calibration_flat_debug.py --dir /path/to/calibration_images --images 50 --min_markers 5
"""

import numpy as np
import cv2
import logging
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml  # Ensure you have installed PyYAML: pip install pyyaml

def debug_calibration(image_dir, num_images, min_markers):
    # Set up the ArUco dictionary and board parameters.
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    markerLength = 3.726    # Adjust units as needed.
    markerSeparation = 0.49
    board = cv2.aruco.GridBoard((4, 5), markerLength, markerSeparation, aruco_dict)
    
    # Use DetectorParameters_create() if available, otherwise use the constructor.
    if hasattr(cv2.aruco, 'DetectorParameters_create'):
        arucoParams = cv2.aruco.DetectorParameters_create()
    else:
        arucoParams = cv2.aruco.DetectorParameters()

    # Lists to store flattened detections.
    corners_all = []
    ids_all = []
    marker_counts = []
    valid_images = []

    last_gray = None

    for idx in tqdm(range(num_images), desc="Processing images"):
        image_path = Path(image_dir) / f"image_{idx}.jpg"
        if not image_path.exists():
            logging.warning(f"Image {image_path} does not exist. Skipping.")
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            logging.warning(f"Failed to load image {image_path}. Skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        last_gray = gray  # For image size.
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=arucoParams)
        detected = 0 if ids is None else len(ids)
        if ids is not None and detected >= min_markers:
            # For each marker, reshape its corner array to (4,2) and add to the cumulative list.
            for c in corners:
                c = np.asarray(c, dtype=np.float32).reshape((4, 2))
                corners_all.append(c)
            # Flatten and extend the IDs list.
            ids_all.extend(ids.flatten().tolist())
            marker_counts.append(detected)
            valid_images.append(idx)
            logging.info(f"Image {idx}: Detected {detected} markers (accepted).")
        else:
            logging.info(f"Image {idx}: Detected {detected} markers (rejected, insufficient markers).")

    if last_gray is None or len(marker_counts) == 0:
        logging.error("No valid images processed. Exiting.")
        return

    logging.info(f"Valid image indices: {valid_images}")
    logging.info(f"Marker counts per valid image: {marker_counts}")
    logging.info(f"Total valid images used for calibration: {len(marker_counts)}")
    logging.info(f"Total markers (flattened) used for calibration: {len(ids_all)}")
    
    # Use image size in (width, height) order.
    image_size = (last_gray.shape[1], last_gray.shape[0])
    logging.info(f"Using image size (width, height): {image_size}")

    # Convert data to expected types.
    counter = np.array(marker_counts, dtype=np.int32)
    ids_all_arr = np.array(ids_all, dtype=np.int32)
    # Note: corners_all is already a list of (4,2) arrays of type float32.

    # Set termination criteria for calibration.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    
    try:
        logging.info("Attempting calibration...")
        ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
            corners_all, ids_all_arr, counter, board, image_size, None, None,
            flags=0, criteria=criteria)
        logging.info(f"Calibration result: {ret}")
        logging.info(f"Camera matrix:\n{mtx}")
        logging.info(f"Distortion coefficients:\n{dist}")

        # Prepare output data.
        calibration_data = {
            "ret": ret,
            "camera_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist(),
            "rotation_vectors": [r.tolist() for r in rvecs],
            "translation_vectors": [t.tolist() for t in tvecs]
        }
        output_file = "calibration_output.yaml"
        with open(output_file, "w") as f:
            yaml.dump(calibration_data, f)
        logging.info(f"Calibration outputs written to {output_file}")
    except Exception as e:
        logging.error(f"Calibration encountered an error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Debug camera calibration using pre-captured images (flattened detections, with explicit data type conversions)."
    )
    parser.add_argument('--dir', type=str,
                        default="/home/rtxcapstone/Raytheon-Capstone-2025/camera_testing/calibration_images",
                        help="Directory containing calibration images.")
    parser.add_argument('--images', type=int, default=50,
                        help="Number of images to process for calibration.")
    parser.add_argument('--min_markers', type=int, default=5,
                        help="Minimum number of markers required to accept an image for calibration.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    debug_calibration(args.dir, args.images, args.min_markers)

if __name__ == "__main__":
    main()

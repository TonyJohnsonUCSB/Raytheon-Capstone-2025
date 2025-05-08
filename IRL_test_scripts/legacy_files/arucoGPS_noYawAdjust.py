import numpy as np
import cv2
import asyncio
import time
import traceback
from mavsdk import System
from picamera2 import Picamera2

# --- Constants ---
ARUCO_DICT = {
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250
}
ARUCO_TYPE = "DICT_6X6_250"
MARKER_SIZE = 0.06611  # meters
DROP_ZONE_ID = 1

CAMERA_MATRIX = np.array([[933.15867, 0, 657.59], [0, 933.1586, 400.36993], [0, 0, 1]])
DIST_COEFFS = np.array([-0.43948, 0.18514, 0, 0])

# --- GPS Calculation ---
def add_offset_to_gps(lat, lon, dx, dy):
    d_lat = (dy / 6378137.0) * (180.0 / np.pi)
    d_lon = (dx / (6378137.0 * np.cos(np.pi * lat / 180.0))) * (180.0 / np.pi)
    return lat + d_lat, lon + d_lon

# --- ArUco Pose Estimation ---
def pose_estimation(frame):
    if frame is None or frame.size == 0:
        print("-- Empty frame received")
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[ARUCO_TYPE])
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    results = []
    if ids is not None:
        ids = ids.flatten()
        for marker_index, marker_id in enumerate(ids):
            if marker_index >= len(corners):
                continue
            marker_corners = corners[marker_index]
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [marker_corners], MARKER_SIZE, CAMERA_MATRIX, DIST_COEFFS
            )
            if tvecs is not None and len(tvecs) > 0:
                tvec = tvecs[0][0]
                results.append((marker_id, tvec))
    return results

# --- Main Async Loop ---
async def main():
    # Connect to drone
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    print("-- Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Drone connected")
            break

    print("-- Waiting for valid GPS fix...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok:
            break
        await asyncio.sleep(0.5)

    # Setup camera
    print("-- Initializing camera...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        raw={"size": (1640, 1232)},
        main={"format": 'RGB888', "size": (640, 480)}
    ))
    picam2.start()
    time.sleep(2)
    print("-- Camera started")

    try:
        async for position in drone.telemetry.position():
            frame = picam2.capture_array()
            detections = pose_estimation(frame)

            if detections:
                for marker_id, tvec in detections:
                    x, y, z = tvec
                    print(f"[Marker ID: {marker_id}] X: {x:.3f} m | Y: {y:.3f} m | Z: {z:.3f} m")

                    if marker_id == DROP_ZONE_ID:
                        new_lat, new_lon = add_offset_to_gps(position.latitude_deg, position.longitude_deg, x, z)
                        print(f"  ↳ Estimated Marker GPS: Lat {new_lat:.7f}, Lon {new_lon:.7f}")
            await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        print("-- Interrupted by user.")
    finally:
        print("-- Cleaning up...")
        picam2.stop()
        picam2.close()

if __name__ == "__main__":
    asyncio.run(main())

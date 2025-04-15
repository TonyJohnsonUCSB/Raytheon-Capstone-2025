import numpy as np
import cv2
import time
import asyncio
from picamera2 import Picamera2
from mavsdk import System
from math import cos, radians

# Camera and ArUco marker configuration
ARUCO_DICT = {
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250
}
ARUCO_TYPE = "DICT_6X6_250"
INTRINSIC_CAMERA = np.array([[933.15867, 0, 657.59], [0, 933.1586, 400.36993], [0, 0, 1]])
DISTORTION = np.array([-0.43948, 0.18514, 0, 0])
DROP_ZONE_ID = 1
MARKER_SIZE = 0.06611  # in meters

# Convert local x/z offset to lat/lon
def local_offset_to_gps(lat_cam, lon_cam, x_offset, z_offset):
    delta_lat = z_offset / 111111.0
    delta_lon = x_offset / (111111.0 * cos(radians(lat_cam)))
    return lat_cam + delta_lat, lon_cam + delta_lon

# Process a frame to detect markers and compute GPS estimates
def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, drop_zone_id, marker_size, lat_cam, lon_cam):
    if frame is None or frame.size == 0:
        print("-- Empty frame received")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    print(f"-- Detected markers: {0 if ids is None else len(ids)}")

    if ids is not None:
        ids = ids.flatten()
        for marker_index, marker_id in enumerate(ids):
            if marker_index >= len(corners):
                continue

            marker_corners = corners[marker_index]
            if marker_corners is None or len(marker_corners) == 0:
                continue

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [marker_corners], marker_size, matrix_coefficients, distortion_coefficients
            )

            if tvecs is not None and len(tvecs) > 0:
                tvec = tvecs[0][0]
                x, y, z = tvec
                print(f"[Marker ID: {int(marker_id)}] X: {x:.3f} m | Y: {y:.3f} m | Z: {z:.3f} m")

                lat_est, lon_est = local_offset_to_gps(lat_cam, lon_cam, x, z)
                print(f"  -- My Current GPS → Latitude: {FAKE_LAT:.6f}, Longitude: {FAKE_LON:.6f}")
                print(f"  → Estimated Marker GPS: Lat: {lat_est:.7f}, Lon: {lon_est:.7f}")

                if marker_id == drop_zone_id:
                    distance = np.linalg.norm(tvec)
                    angle_x = np.degrees(np.arctan2(x, z))
                    print(f"  ↳ Drop-Zone → Distance: {distance:.2f} m | Angle X: {angle_x:.2f}°")

# Main async loop to run camera and GPS capture
async def main():
    print("-- Connecting to drone...")
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    print("-- Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Drone connected")
            break

    print("-- Waiting for GPS lock...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- GPS position OK")
            break

    print("-- Starting camera...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        raw={"size": (1640, 1232)},
        main={"format": 'RGB888', "size": (640, 480)}
    ))
    picam2.start()
    time.sleep(2)

    print("-- Running main loop")
    try:
        async for position in drone.telemetry.position():
            frame = picam2.capture_array()
            lat = position.latitude_deg
            lon = position.longitude_deg
            pose_estimation(frame, ARUCO_DICT[ARUCO_TYPE], INTRINSIC_CAMERA, DISTORTION, DROP_ZONE_ID, MARKER_SIZE, lat, lon)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("-- Interrupted by user")
    finally:
        print("-- Cleaning up")
        picam2.stop()
        picam2.close()

if __name__ == "__main__":
    asyncio.run(main())

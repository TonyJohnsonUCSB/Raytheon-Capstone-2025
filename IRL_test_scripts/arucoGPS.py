import numpy as np
import cv2
import time
import asyncio
from mavsdk import System
from mavsdk.telemetry import Position, EulerAngle
from picamera2 import Picamera2

# Camera intrinsics and distortion coefficients
intrinsic_camera = np.array([[933.15867, 0, 657.59],
                             [0, 933.1586, 400.36993],
                             [0, 0, 1]])
distortion = np.array([-0.43948, 0.18514, 0, 0])

ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_SIZE_METERS = 0.06611
DROP_ZONE_ID = 1

# Earth radius (in meters)
EARTH_RADIUS = 6378137.0

def offset_to_gps(lat, lon, d_north, d_east):
    d_lat = d_north / EARTH_RADIUS
    d_lon = d_east / (EARTH_RADIUS * np.cos(np.radians(lat)))
    new_lat = lat + np.degrees(d_lat)
    new_lon = lon + np.degrees(d_lon)
    return new_lat, new_lon

def pose_estimation(frame, matrix_coefficients, distortion_coefficients, marker_size, yaw_deg):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    detections = []

    if ids is not None:
        ids = ids.flatten()
        for i, marker_id in enumerate(ids):
            marker_corners = corners[i]
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [marker_corners], marker_size, matrix_coefficients, distortion_coefficients
            )
            if tvecs is not None and len(tvecs) > 0:
                tvec = tvecs[0][0]
                x, y, z = tvec  # x (right), y (down), z (forward)

                # Rotate x/z offset into NED frame using yaw
                yaw_rad = np.radians(yaw_deg)
                d_north = z * np.cos(yaw_rad) - x * np.sin(yaw_rad)
                d_east = z * np.sin(yaw_rad) + x * np.cos(yaw_rad)

                detections.append({
                    "id": marker_id,
                    "offset": (x, y, z),
                    "delta_north": d_north,
                    "delta_east": d_east
                })

    return detections

async def main():
    # Initialize drone connection
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    print("-- Waiting for drone connection")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone")
            break

    print("-- Waiting for global position fix")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok:
            print("-- GPS fix acquired")
            break

    # Initialize camera
    print("-- Initializing camera")
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": 'RGB888', "size": (640, 480)}
    ))
    picam2.start()
    time.sleep(2)
    print("-- Camera ready")

    try:
        async for position, attitude in zip(drone.telemetry.position(), drone.telemetry.attitude_euler()):
            frame = picam2.capture_array()
            print("-- Frame captured")

            detections = pose_estimation(
                frame,
                intrinsic_camera,
                distortion,
                MARKER_SIZE_METERS,
                attitude.yaw_deg
            )

            for d in detections:
                marker_id = d["id"]
                x, y, z = d["offset"]
                d_north = d["delta_north"]
                d_east = d["delta_east"]

                marker_lat, marker_lon = offset_to_gps(position.latitude_deg, position.longitude_deg, d_north, d_east)

                print(f"[Marker ID: {marker_id}] X: {x:.2f} m | Y: {y:.2f} m | Z: {z:.2f} m")
                print(f"  → GPS: {marker_lat:.6f}, {marker_lon:.6f}")

            await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        print("-- Interrupted by user")
    finally:
        picam2.stop()
        picam2.close()
        print("-- Camera shut down")

if __name__ == "__main__":
    asyncio.run(main())

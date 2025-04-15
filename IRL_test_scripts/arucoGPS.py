import asyncio
import numpy as np
import cv2
import time
import math
import traceback
from mavsdk import System
from picamera2 import Picamera2

# --- Settings ---
ARUCO_DICT = {
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250
}
ARUCO_TYPE = "DICT_6X6_250"
MARKER_SIZE = 0.06611  # meters
DROP_ZONE_ID = 1

INTRINSIC_CAMERA = np.array([
    [933.15867, 0, 657.59],
    [0, 933.1586, 400.36993],
    [0, 0, 1]
])
DISTORTION = np.array([-0.43948, 0.18514, 0, 0])

METERS_PER_DEG_LAT = 111111

async def main():
    print("-- Waiting for drone connection")
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone")
            break

    print("-- Waiting for global position fix")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- GPS fix acquired")
            break

    print("-- Initializing camera")
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": 'RGB888', "size": (640, 480)},
        raw={"size": (1536, 864)}
    ))
    picam2.start()
    time.sleep(2)
    print("-- Camera started")

    try:
        position_stream = drone.telemetry.position()
        while True:
            position = await position_stream.__anext__()
            frame = picam2.capture_array()

            if frame is None or frame.size == 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[ARUCO_TYPE])
            parameters = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

            if ids is not None:
                ids = ids.flatten()
                for marker_index, marker_id in enumerate(ids):
                    if marker_index >= len(corners):
                        continue

                    marker_corners = corners[marker_index]
                    if marker_corners is None or len(marker_corners) == 0:
                        continue

                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [marker_corners], MARKER_SIZE, INTRINSIC_CAMERA, DISTORTION
                    )

                    if tvecs is not None and len(tvecs) > 0:
                        x, y, z = tvecs[0][0]
                        print(f"-- [Marker ID: {int(marker_id)}] X: {x:.3f} m | Y: {y:.3f} m | Z: {z:.3f} m")

                        if marker_id == DROP_ZONE_ID:
                            current_lat = position.latitude_deg
                            current_lon = position.longitude_deg
                            meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(current_lat))

                            d_north = -y  # ArUco coordinate: forward is -Y
                            d_east = x    # ArUco coordinate: right is +X

                            delta_lat = d_north / METERS_PER_DEG_LAT
                            delta_lon = d_east / meters_per_deg_lon

                            marker_lat = current_lat + delta_lat
                            marker_lon = current_lon + delta_lon

                            print(f"--   ↳ Estimated Marker GPS: ({marker_lat:.6f}, {marker_lon:.6f})")

            # Show preview
            cv2.imshow("Camera Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("-- Interrupted by user")

    except Exception:
        print("-- Error occurred:")
        traceback.print_exc()

    finally:
        print("-- Camera shut down")
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())

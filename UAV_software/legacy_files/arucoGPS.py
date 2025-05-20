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
MARKER_SIZE = 0.254  # 10 in = 0.254 m
DROP_ZONE_ID = 1

INTRINSIC_CAMERA = np.array([
    [933.15867, 0, 657.59],
    [0, 933.1586, 400.36993],
    [0, 0, 1]
])
DISTORTION = np.array([-0.43948, 0.18514, 0, 0])  # k1, k2, p1, p2

METERS_PER_DEG_LAT = 111111  # approximate

async def main():
    print("-- Waiting for drone connection")
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone")
            break

    #print("-- Waiting for global position fix")
    #async for health in drone.telemetry.health():
        #if health.is_global_position_ok and health.is_home_position_ok:
         #   print("-- GPS fix acquired")
          #  break

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
        async for position in position_stream:
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
                    # extract this marker's corners
                    marker_corners = corners[marker_index]

                    # estimate pose
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [marker_corners], MARKER_SIZE, INTRINSIC_CAMERA, DISTORTION
                    )

                    # camera‐frame translation (meters)
                    x_cam, y_cam, z_cam = tvecs[0][0]
                    print(f"-- [Marker ID: {int(marker_id)}] "
                          f"X_cam: {x_cam:.3f} m | Y_cam: {y_cam:.3f} m | Z_cam: {z_cam:.3f} m")

                    if marker_id == DROP_ZONE_ID:
                        # camera axes: +X → East, +Y → North, +Z → Down
                        d_east  = x_cam
                        d_north = y_cam
                        d_down  = z_cam

                        # convert to lat/lon offsets
                        #current_lat = position.latitude_deg
                        #current_lon = position.longitude_deg
                        current_lat = 0
                        current_lon = 0
                        meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(current_lat))
                        delta_lat = d_north / METERS_PER_DEG_LAT
                        delta_lon = d_east  / meters_per_deg_lon

                        marker_lat = current_lat + delta_lat
                        marker_lon = current_lon + delta_lon

                        print(f"--   ↳ Estimated Marker GPS: ({marker_lat:.6f}, {marker_lon:.6f})")

            # show camera preview
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

import asyncio
import threading
import time
import math
import numpy as np
import cv2
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.geofence import Point, Polygon, FenceType, GeofenceData

# — Waypoints and geofence —
coordinates = [
    (34.4189167, -119.8553056),
    (34.4189722, -119.8553056),
    (34.4189722, -119.8551667),
    (34.4189167, -119.8551667)
]
geofence_points = [
    Point(34.418606, -119.855929),
    Point(34.418600, -119.855196),
    Point(34.419221, -119.855198),
    Point(34.419228, -119.855931)
]

# — Flight & vision params —
ALTITUDE       = 10.0            # meters
RESOLUTION     = (640, 480)
MARKER_SIZE    = 0.06611         # meters
DROP_ZONE_ID   = 1
INTRINSIC      = np.array([
    [653.1070007239106, 0.0,               339.2952147845755],
    [0.0,               650.7753992788821, 258.1165494889447],
    [0.0,               0.0,               1.0]
], dtype=np.float32)
DIST_COEFFS    = np.array([
    -0.03887864427953473,
     0.6888798469690414,
     0.00815702400928161,
     0.010438854120041072,
    -1.713270699000528
], dtype=np.float32)
R_EARTH        = 6378137.0        # meters

def enu_to_geodetic(lat0, lon0, alt0, east, north, up):
    dlat = north / R_EARTH
    dlon = east  / (R_EARTH * math.cos(math.radians(lat0)))
    lat = lat0 + math.degrees(dlat)
    lon = lon0 + math.degrees(dlon)
    alt = alt0 + up
    return lat, lon, alt

async def search_marker(drone, camera, timeout=10.0):
    start = time.time()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    while time.time() - start < timeout:
        frame = camera.capture_array()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
        if ids is not None:
            for idx, mid in enumerate(ids.flatten()):
                if mid == DROP_ZONE_ID:
                    _, _, tvecs = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS
                    )
                    if len(tvecs) > 0:
                        x_east, y_north, z_down = tvecs[0][0]
                        z_up = -z_down
                        async for pos in drone.telemetry.position():
                            lat0 = pos.latitude_deg
                            lon0 = pos.longitude_deg
                            alt0 = pos.absolute_altitude_m
                            break
                        return enu_to_geodetic(lat0, lon0, alt0,
                                               x_east, y_north, z_up)
        await asyncio.sleep(0.1)
    return None

async def run():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    print("Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected")
            break

    print("Waiting for global position...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position OK")
            break

    print("-- Uploading geofence")
    polygon = Polygon(geofence_points, FenceType.INCLUSION)
    await drone.geofence.upload_geofence(GeofenceData(polygons=[polygon], circles=[]))

    print("-- Arming")
    await drone.action.arm()

    print("-- Taking off")
    await drone.action.set_takeoff_altitude(3.0)
    await drone.action.takeoff()
    await asyncio.sleep(15)

    # initialize camera
    camera = Picamera2()
    cfg = camera.create_preview_configuration(main={"format":"RGB888", "size":RESOLUTION})
    camera.configure(cfg)
    camera.start()
    time.sleep(2)

    # start video recording
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter('/home/rtxcapstone/Desktop/testVideo.avi', fourcc, 30.0, RESOLUTION)
    if not writer.isOpened():
        raise RuntimeError("Failed to open video writer")
    recording = True

    def record_loop():
        while recording:
            frame = camera.capture_array()
            writer.write(frame)
            # small sleep to avoid CPU hog
            time.sleep(0.01)

    record_thread = threading.Thread(target=record_loop, daemon=True)
    record_thread.start()

    # fly square and search
    for i, (lat, lon) in enumerate(coordinates, start=1):
        print(f"-- Waypoint {i}: flying to {lat:.6f}, {lon:.6f}")
        await drone.action.goto_location(lat, lon, ALTITUDE, 0.0)
        await asyncio.sleep(15)

        print("-- Searching for marker at this waypoint")
        found = await search_marker(drone, camera, timeout=10.0)
        if found:
            mlat, mlon, malt = found
            print(f"-- Marker found at {mlat:.6f}, {mlon:.6f}, alt {malt:.1f}m")
            print("-- Flying to marker and landing")
            await drone.action.goto_location(mlat, mlon, malt, 0.0)
            await asyncio.sleep(15)
            await drone.action.land()
            # cleanup recording
            recording = False
            record_thread.join()
            writer.release()
            return

    print("-- Marker not found; returning to launch")
    await drone.action.return_to_launch()
    # cleanup recording
    recording = False
    record_thread.join()
    writer.release()

if __name__ == "__main__":
    asyncio.run(run())

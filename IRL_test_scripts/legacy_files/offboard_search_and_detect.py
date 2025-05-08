import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from mavsdk.geofence import Point, Polygon, FenceType, GeofenceData

# --- params ---
WAYPOINTS = [
    (34.4189167, -119.8553056),
    (34.4189722, -119.8553056),
    (34.4189722, -119.8551667),
    (34.4189167, -119.8551667),
]
ALTITUDE      = 3.0
DROP_ZONE_ID  = 1
MARKER_SIZE   = 0.06611

# geofence polygon points
GEOFENCE_POLYGON = [
    Point(34.419215, -119.854763),
    Point(34.418600, -119.854763),
    Point(34.418602, -119.855852),
    Point(34.419218, -119.855850),
]

ARUCO_DICT    = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
DETECT_PARAMS = cv2.aruco.DetectorParameters_create()
CAM_MAT       = np.array([[933.15867, 0, 657.59],
                          [0, 933.1586, 400.36993],
                          [0,       0,      1]])
DIST_COEFFS   = np.array([-0.43948, 0.18514, 0, 0])

# --- camera init ---
picam2 = Picamera2()
cfg   = picam2.create_preview_configuration(
    raw={"size": (1640,1232)},
    main={"format":'RGB888',"size":(640,480)}
)
picam2.configure(cfg)
picam2.start()
time.sleep(2)
cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Preview",640,480)
print("Camera initialized and preview window opened")

def compute_vel(pos, axis):
    if abs(pos) < 0.01:
        return 0.0
    if abs(pos) < 0.3:
        return (-1, -1)[axis] * np.sign(pos)
    return (-1, 1)[axis] * pos

async def connect_and_arm():
    drone = System()
    print("-> Connecting to drone (serial:///dev/ttyAMA0:57600)...")
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    print("-> Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to drone, components:", state.component_ids)
            break

    print("-> Waiting for GPS fix and home position...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("Global position OK and home position set")
            break

    # upload geofence
    print("-> Uploading geofence polygon...")
    fence = Polygon(GEOFENCE_POLYGON, FenceType.INCLUSION)
    geofence_data = GeofenceData([fence], [])
    result = await drone.geofence.upload_geofence(geofence_data)
    print("Geofence upload result:", result)

    print("-> Arming motors...")
    await drone.action.arm()
    print(f"-> Taking off to {ALTITUDE} m altitude...")
    await drone.action.set_takeoff_altitude(ALTITUDE)
    await drone.action.takeoff()
    await asyncio.sleep(6)
    print("Takeoff complete")
    return drone

async def scan_for_marker(timeout=3.0):
    print(f"-> Scanning for marker ID {DROP_ZONE_ID} for up to {timeout} seconds...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        frame = await asyncio.to_thread(picam2.capture_array)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, ARUCO_DICT, parameters=DETECT_PARAMS
        )
        if ids is not None and DROP_ZONE_ID in ids.flatten():
            print("Marker detected during scan")
            return True
    print("Marker not detected during scan")
    return False

async def offboard_center_and_land(drone):
    print("-> Entering offboard mode for precision approach...")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0,0,0,0))
    try:
        await drone.offboard.start()
        print("Offboard mode started")
    except OffboardError as e:
        print("Failed to start offboard mode:", e._result.result)
        return False

    while True:
        frame = await asyncio.to_thread(picam2.capture_array)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, ARUCO_DICT, parameters=DETECT_PARAMS
        )

        if ids is None or DROP_ZONE_ID not in ids.flatten():
            print("Marker lost, hovering...")
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0,0,0,0)
            )
        else:
            idx = list(ids.flatten()).index(DROP_ZONE_ID)
            _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]], MARKER_SIZE, CAM_MAT, DIST_COEFFS
            )
            x, y, _ = tvecs[0][0]
            vx = compute_vel(x, axis=0)
            vy = compute_vel(y, axis=1)
            print(f"Marker pose x={x:.3f}, y={y:.3f}; commanding vx={vx:.3f}, vy={vy:.3f}")
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(vx, vy, 0, 0)
            )
            if abs(x) < 0.02 and abs(y) < 0.02:
                print("Centered over marker")
                break

        await asyncio.sleep(0.05)

    print("-> Stopping offboard mode and landing...")
    await drone.offboard.stop()
    await drone.action.land()
    await asyncio.sleep(10)
    print("Landed on marker")
    return True

async def main():
    drone = await connect_and_arm()
    found = False

    for idx, (lat, lon) in enumerate(WAYPOINTS, start=1):
        print(f"-> Heading to waypoint {idx}: ({lat}, {lon}) at {ALTITUDE} m")
        await drone.action.goto_location(lat, lon, ALTITUDE, 0.0)
        print("Arrived at waypoint")
        await asyncio.sleep(1)
        if await scan_for_marker():
            found = await offboard_center_and_land(drone)
            break

    if not found:
        print("Marker never found, landing now")
        await drone.action.land()
        await asyncio.sleep(10)
        print("Landed without marker")

    picam2.stop()
    cv2.destroyAllWindows()
    print("Mission complete. Marker found?", found)

if __name__ == "__main__":
    asyncio.run(main())

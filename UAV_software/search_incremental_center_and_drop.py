#!/usr/bin/env python3

import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw
from mavsdk.geofence import Point, Polygon, FenceType, GeofenceData

# ----------------------------
# Camera Globals
# ----------------------------
picam2 = Picamera2()
write_width, write_height = 640, 480

# ----------------------------
# Calibration and Distortion
# ----------------------------
INTRINSIC = np.array([
    [653.1070007239106, 0.0, 339.2952147845755],
    [0.0, 650.7753992788821, 258.1165494889447],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

DIST_COEFFS = np.array([
    -0.03887864427953473,
     0.6888798469690414,
     0.00815702400928161,
     0.010438854120041072,
    -1.713270699000528
], dtype=np.float32)

# ----------------------------
# ArUco Detection Parameters
# ----------------------------
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
DETECT_PARAMS = cv2.aruco.DetectorParameters_create()
DETECT_PARAMS.adaptiveThreshConstant = 7
DETECT_PARAMS.minMarkerPerimeterRate = 0.03
MARKER_SIZE = 0.06611  # meters
TARGET_ID = 1

# ----------------------------
# Flight Parameters
# ----------------------------
ALTITUDE = 5       # takeoff and waypoint altitude in meters
AMSL_ALTITUDE = ALTITUDE + 9
TOLERANCE = 0.01  # N/E position tolerance for landing in meters

# ----------------------------
# Waypoints and Geofence
# ----------------------------
# Updated coordinates to plot
coordinates = [
    (34.4189,  -119.85533),
    (34.4189,  -119.85530),
    (34.4189,  -119.85528),
    (34.4189,  -119.85526),
    (34.4189,  -119.85524),
    (34.4189,  -119.85522),
    (34.4189,  -119.85520),
]

# Geofence shifted 20 m east (pre‑computed)
GEOFENCE_POINTS = [
    (34.4186,  -119.85600),
    (34.4186,  -119.85475),
    (34.4192,  -119.85475),
    (34.4192,  -119.85600),
    (34.4186,  -119.85600),
]

# ----------------------------
# Initialize Camera
# ----------------------------
cam_cfg = picam2.create_preview_configuration(
    raw={"size": (1640, 1232)},
    main={"format": "RGB888", "size": (write_width, write_height)}
)
picam2.configure(cam_cfg)
picam2.start()
print("[DEBUG] Camera started: RGB888 preview at {}x{}".format(write_width, write_height))
time.sleep(2)  # allow auto-exposure to stabilize


async def connect_and_arm():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    print("-- Arming")
    await drone.action.arm()

    print("-- Taking off")
    await drone.action.set_takeoff_altitude(5)
    await drone.action.takeoff()

    await asyncio.sleep(10)

    return drone

async def search_marker(timeout=5.0):
    print(f"[DEBUG] Searching for marker (timeout: {timeout} seconds)")
    t0 = time.time()
    prev_gray = None

    while (time.time() - t0) < timeout:
        frame = await asyncio.to_thread(picam2.capture_array)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100,
                                          qualityLevel=0.01, minDistance=20)
            if pts is not None:
                curr, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                valid_count = np.count_nonzero(st.reshape(-1) == 1)
                #print(f"[DEBUG] Optical flow valid points: {valid_count}")
                if valid_count >= 6:
                    M, _ = cv2.estimateAffinePartial2D(pts[st.reshape(-1) == 1],
                                                     curr[st.reshape(-1) == 1])
                    if M is not None:
                        frame = cv2.warpAffine(frame, M, frame.shape[1::-1])
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        #print("[DEBUG] Applied stabilization warp")
        prev_gray = gray

        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT,
                                                 parameters=DETECT_PARAMS)
        if ids is not None and TARGET_ID in ids.flatten():
            print(f"[DEBUG] Marker ID {TARGET_ID} detected")
            print("Sending drone GPS location")

            idx = list(ids.flatten()).index(TARGET_ID)
            _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS
            )
            offset = tvecs[0][0]
            print(f"[DEBUG] Pose offset: {offset}")
            return offset

        print("[DEBUG] Marker not found in current frame")

    print("[DEBUG] Marker search timed out")
    return None

async def approach_and_land(drone):
    print('[DEBUG] Starting velocity-based continuous approach')
    # get initial yaw
    async for att in drone.telemetry.attitude_euler():
        yaw = att.yaw_deg
        print(f'[DEBUG] Initial yaw: {yaw:.1f}')
        break

    # set zero velocity and start offboard
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, yaw))
    try:
        print('[DEBUG] Enabling offboard')
        await drone.offboard.start()
        print('[DEBUG] Offboard started')
    except OffboardError as e:
        print(f'[ERROR] Offboard start failed: {e}')
        return

    while True:
        # get current NED
        async for od in drone.telemetry.position_velocity_ned():
            north = od.position.north_m
            east  = od.position.east_m
            down  = od.position.down_m
            print(f'[DEBUG] Current NED: N={north:.2f}, E={east:.2f}, D={down:.2f}')
            break

        # search for marker
        print('[DEBUG] Updating offset via camera')
        offset = await search_marker(timeout=2.0)
        if offset is None:
            print('[DEBUG] No offset, retrying')
            continue

        dx_e = offset[0]
        dx_n = offset[1]
        dist = math.hypot(dx_n, dx_e)
        print(f'[DEBUG] Distance to marker: {dist:.3f}m')

        if dist < TOLERANCE:
            print('[DEBUG] Within tolerance, landing')
            await drone.offboard.stop()
            await drone.action.land()
            print('[DEBUG] Land command sent')
            return

        # compute velocity toward marker
        vn = - (dx_n / dist) * VELOCITY * 0.5
        ve = (dx_e / dist) * VELOCITY * 0.5
        print(f'[DEBUG] Commanding velocity N={vn:.2f}m/s, E={ve:.2f}m/s')
        await drone.offboard.set_velocity_ned(VelocityNedYaw(vn, ve, 0.0, yaw))

        # wait before next update
        await asyncio.sleep(1.0)
     


async def run():
    drone = None
    try:
        drone = await connect_and_arm()
        for lat, lon in coordinates:
            print(f"[DEBUG] Heading to waypoint ({lat}, {lon}) at {ALTITUDE} meters")
            await drone.action.goto_location(lat, lon, AMSL_ALTITUDE, 0.0)
            await asyncio.sleep(7)
            print(f"[DEBUG] Attempting marker search at ({lat}, {lon})")
            tvec = await search_marker(10.0)
            if tvec is not None:
                await approach_and_land(drone)
                return

        print("[DEBUG] No marker found; returning to launch")
        await drone.action.return_to_launch()
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
    finally:
        if drone:
            try:
                await drone.offboard.stop()
            except:
                pass
            await drone.action.land()
        print("[DEBUG] Mission complete or aborted")


if __name__ == '__main__':
    print("[DEBUG] Script start")
    asyncio.run(run())
    print("[DEBUG] Script exit")

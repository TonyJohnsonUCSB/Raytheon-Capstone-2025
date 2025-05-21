#!/usr/bin/env python3

import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

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
TOLERANCE = 0.01   # N/E position tolerance for landing in meters

# ----------------------------
# Waypoints and Geofence
# ----------------------------
coordinates = [
    (34.4189,  -119.85533),
    (34.4189,  -119.85530),
    (34.4189,  -119.85528),
    (34.4189,  -119.85526),
    (34.4189,  -119.85524),
    (34.4189,  -119.85522),
    (34.4189,  -119.85520),
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
time.sleep(2)

async def connect_and_arm():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            break
    await drone.action.arm()
    await drone.action.set_takeoff_altitude(ALTITUDE)
    await drone.action.takeoff()
    await asyncio.sleep(10)
    return drone

async def search_marker(timeout=5.0):
    t0 = time.time()
    prev_gray = None
    while time.time() - t0 < timeout:
        frame = await asyncio.to_thread(picam2.capture_array)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100,
                                          qualityLevel=0.01, minDistance=20)
            if pts is not None:
                curr, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                if np.count_nonzero(st) >= 6:
                    M, _ = cv2.estimateAffinePartial2D(
                        pts[st.flatten()==1], curr[st.flatten()==1]
                    )
                    if M is not None:
                        frame = cv2.warpAffine(frame, M, frame.shape[1::-1])
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        prev_gray = gray

        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT,
                                                 parameters=DETECT_PARAMS)
        if ids is not None and TARGET_ID in ids.flatten():
            idx = list(ids.flatten()).index(TARGET_ID)
            _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS
            )
            return tvecs[0][0]
    return None

async def approach_and_land(drone):
    # start offboard
    async for od in drone.telemetry.position_velocity_ned():
        break
    async for att in drone.telemetry.attitude_euler():
        break
    try:
        await drone.offboard.set_position_ned(
            PositionNedYaw(od.position.north_m,
                            od.position.east_m,
                            od.position.down_m,
                            att.yaw_deg)
        )
        await drone.offboard.start()
    except OffboardError:
        return

    while True:
        # get fresh pose
        async for od in drone.telemetry.position_velocity_ned():
            north0 = od.position.north_m
            east0  = od.position.east_m
            down0  = od.position.down_m
            break
        async for att in drone.telemetry.attitude_euler():
            yaw = att.yaw_deg
            break

        offset = await search_marker(timeout=2.0)
        if offset is None:
            continue

        target_n = north0 + offset[1]
        target_e = east0  + offset[0]

        await drone.offboard.set_position_ned(
            PositionNedYaw(target_n, target_e, down0, yaw)
        )

        await asyncio.sleep(0.5)
        async for od in drone.telemetry.position_velocity_ned():
            err_n = abs(od.position.north_m - target_n)
            err_e = abs(od.position.east_m  - target_e)
            break

        if err_n < TOLERANCE and err_e < TOLERANCE:
            await drone.offboard.stop()
            await drone.action.land()
            return

async def run():
    drone = await connect_and_arm()
    try:
        for lat, lon in coordinates:
            await drone.action.goto_location(lat, lon, AMSL_ALTITUDE, 0.0)
            await asyncio.sleep(7)
            await approach_and_land(drone)
            return
        await drone.action.return_to_launch()
    finally:
        try:
            await drone.offboard.stop()
        except:
            pass
        await drone.action.land()

if __name__ == "__main__":
    asyncio.run(run())
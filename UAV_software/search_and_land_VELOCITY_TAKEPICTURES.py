#!/usr/bin/env python3

import asyncio
import time
import math
import os
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw

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
ALTITUDE = 5               # takeoff altitude above ground, in meters
AMSL_ALTITUDE = ALTITUDE + 9
TOLERANCE = 0.05           # N/E tolerance for landing, in meters
VELOCITY = 0.5             # approach speed, m/s

if VELOCITY <= 0:
    raise ValueError('VELOCITY must be positive and non-zero')

# ----------------------------
# Waypoints
# ----------------------------
coordinates = [
    (34.4188664, -119.8559220),
    (34.41886,   -119.8559220)
]

# ----------------------------
# Init Camera
# ----------------------------
cam_cfg = picamera2.create_preview_configuration(
    raw={'size': (1640, 1232)},
    main={'format': 'RGB888', 'size': (write_width, write_height)}
)
picam2.configure(cam_cfg)
picam2.start()
print('[DEBUG] Camera started')
time.sleep(2)
print('[DEBUG] Camera exposure stabilized')

# Ensure directory exists for saving images
os.makedirs("test_photos", exist_ok=True)

async def connect_and_arm():
    print('[DEBUG] Connecting to drone...')
    drone = System()
    await drone.connect(system_address='serial:///dev/ttyAMA0:57600')

    print('[DEBUG] Waiting for connection...')
    async for state in drone.core.connection_state():
        if state.is_connected:
            print('[DEBUG] Connected')
            break

    print('[DEBUG] Waiting for global position...')
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print('[DEBUG] Global position OK')
            break

    print('[DEBUG] Arming')
    await drone.action.arm()

    print(f'[DEBUG] Setting takeoff altitude to {ALTITUDE}m')
    await drone.action.set_takeoff_altitude(float(ALTITUDE))

    print('[DEBUG] Taking off')
    await drone.action.takeoff()
    await asyncio.sleep(10)
    print('[DEBUG] Takeoff complete')

    frame = await asyncio.to_thread(picam2.capture_array)
    cv2.imwrite("test_photos/takeoff_complete.jpg", frame)
    print('[DEBUG] Saved test_photos/takeoff_complete.jpg')

    return drone

async def search_marker(timeout=3.0):
    print(f'[DEBUG] search_marker: timeout={timeout}s')
    t0 = time.time()
    prev_gray = None
    frame_cnt = 0

    while time.time() - t0 < timeout:
        frame = await asyncio.to_thread(picam2.capture_array)
        frame_cnt += 1
        print(f'[DEBUG] Frame {frame_cnt} captured')
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100,
                                          qualityLevel=0.01, minDistance=20)
            print(f'[DEBUG] Features to track: {0 if pts is None else len(pts)}')
            if pts is not None:
                curr, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                valid = np.count_nonzero(st.reshape(-1) == 1)
                print(f'[DEBUG] Optical flow valid: {valid}')
                if valid >= 6:
                    M, _ = cv2.estimateAffinePartial2D(
                        pts[st.reshape(-1)==1], curr[st.reshape(-1)==1]
                    )
                    if M is not None:
                        frame = cv2.warpAffine(frame, M, frame.shape[1::-1])
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        print('[DEBUG] Frame stabilized')

        prev_gray = gray

        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=DETECT_PARAMS)
        print(f'[DEBUG] detectMarkers ids: {None if ids is None else ids.flatten().tolist()}')
        if ids is not None and TARGET_ID in ids.flatten():
            idx = list(ids.flatten()).index(TARGET_ID)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS
            )
            offset = tvecs[0][0]
            print(f'[DEBUG] Marker offset: x={offset[0]:.3f}, y={offset[1]:.3f}, z={offset[2]:.3f}')

            # overlay pose on frame
            cv2.aruco.drawDetectedMarkers(frame, [corners[idx]])
            cv2.drawFrameAxes(
                frame,
                INTRINSIC,
                DIST_COEFFS,
                rvecs[0][0],
                tvecs[0][0],
                0.03
            )
            cv2.imwrite("test_photos/marker_detected.jpg", frame)
            print('[DEBUG] Saved test_photos/marker_detected.jpg')

            return offset

        print('[DEBUG] Marker not found')

    print('[DEBUG] search_marker timed out')
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
            print('[DEBUG] Within tolerance, preparing to land')

            frame_pre_land = await asyncio.to_thread(picam2.capture_array)
            cv2.imwrite("test_photos/pre_land.jpg", frame_pre_land)
            print('[DEBUG] Saved test_photos/pre_land.jpg')

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
    print('[DEBUG] run() start')
    drone = await connect_and_arm()
    try:
        for idx, (lat, lon) in enumerate(coordinates):
            print(f'[DEBUG] Waypoint {idx+1}: goto ({lat}, {lon})')
            await drone.action.goto_location(lat, lon, AMSL_ALTITUDE, 0.0)
            await asyncio.sleep(7)

            print('[DEBUG] Arrived, starting approach')
            await approach_and_land(drone)
            print('[DEBUG] approach_and_land complete, exiting')
            return

        print('[DEBUG] No marker found, RTL')
        await drone.action.return_to_launch()

    except Exception as e:
        print(f'[ERROR] Exception: {e}')
    finally:
        try:
            await drone.offboard.stop()
            print('[DEBUG] Offboard stopped in cleanup')
        except:
            pass
        await drone.action.land()
        print('[DEBUG] Cleanup land command sent')

if __name__ == '__main__':
    print('[DEBUG] Script start]')
    asyncio.run(run())
    print('[DEBUG] Script exit')

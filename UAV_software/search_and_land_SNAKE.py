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
ALTITUDE = 5         # takeoff and waypoint altitude in meters
AMSL_ALTITUDE = ALTITUDE + 9
TOLERANCE = 0.1      # N/E position tolerance for landing in meters

# ----------------------------
# Waypoints and Geofence
# ----------------------------
LAT1 = 34.41900
LAT2 = 34.418925
LAT3 = 34.41885
LAT4 = 34.418775
LON1 = -119.855400
LON2 = -119.855250
LON3 = -119.855100
LON4 = -119.854950

coordinates = [
    (LAT1, LON1), (LAT1, LON2), (LAT1, LON3), (LAT1, LON4),
    (LAT2, LON4), (LAT2, LON3), (LAT2, LON2), (LAT2, LON1),
    (LAT3, LON1), (LAT3, LON2), (LAT3, LON3), (LAT3, LON4),
    (LAT4, LON4), (LAT4, LON3), (LAT4, LON2), (LAT4, LON1),
]

GEOFENCE_POINTS = [
    (34.4186, -119.85600),
    (34.4186, -119.85475),
    (34.4192, -119.85475),
    (34.4192, -119.85600),
    (34.4186, -119.85600),
]

# ----------------------------
# Initialize Camera
# ----------------------------
cam_cfg = picam2.create_preview_configuration(
    raw={'size': (1640, 1232)},
    main={'format': 'RGB888', 'size': (write_width, write_height)}
)
picam2.configure(cam_cfg)
picam2.start()
print('[DEBUG] Camera started: {}x{} RGB888'.format(write_width, write_height))
time.sleep(2)
print('[DEBUG] Camera auto-exposure stabilized')

async def connect_and_arm():
    print('[DEBUG] Connecting to drone...')
    drone = System()
    await drone.connect(system_address='serial:///dev/ttyAMA0:57600')

    print('[DEBUG] Waiting for connection...')
    async for state in drone.core.connection_state():
        if state.is_connected:
            print('[DEBUG] Drone connected')
            break

    print('[DEBUG] Waiting for global position estimate...')
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print('[DEBUG] Global position estimate OK')
            break

    print('[DEBUG] Arming...')
    await drone.action.arm()

    print('[DEBUG] Setting takeoff altitude to {}m'.format(ALTITUDE))
    await drone.action.set_takeoff_altitude(ALTITUDE)

    print('[DEBUG] Taking off...')
    await drone.action.takeoff()
    await asyncio.sleep(10)
    print('[DEBUG] Takeoff complete')

    return drone

async def search_marker(timeout=10.0):
    print('[DEBUG] search_marker timeout={}s'.format(timeout))
    t0 = time.time()
    prev_gray = None
    frame_count = 0

    while time.time() - t0 < timeout:
        frame = await asyncio.to_thread(picam2.capture_array)
        frame_count += 1
        print('[DEBUG] Frame {} captured'.format(frame_count))

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=20
            )
            print('[DEBUG] goodFeaturesToTrack returned {}'.format(
                0 if pts is None else len(pts)))
            if pts is not None:
                curr, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                valid = np.count_nonzero(st.reshape(-1) == 1)
                print('[DEBUG] Optical flow valid points: {}'.format(valid))
                if valid >= 6:
                    M, _ = cv2.estimateAffinePartial2D(
                        pts[st.reshape(-1) == 1], curr[st.reshape(-1) == 1]
                    )
                    if M is not None:
                        frame = cv2.warpAffine(frame, M, frame.shape[1::-1])
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        print('[DEBUG] Applied stabilization warp')

        prev_gray = gray

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, ARUCO_DICT, parameters=DETECT_PARAMS
        )
        print('[DEBUG] detectMarkers ids: {}'.format(
            None if ids is None else ids.flatten().tolist()))

        if ids is not None and TARGET_ID in ids.flatten():
            print('[DEBUG] Marker {} detected'.format(TARGET_ID))
            idx = list(ids.flatten()).index(TARGET_ID)
            _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS
            )
            offset = tvecs[0][0]
            print('[DEBUG] Pose offset: {}'.format(offset))
            return offset

        print('[DEBUG] Marker not found in this frame')

    print('[DEBUG] Marker search timed out')
    return None

async def approach_and_land(drone, initial_offset):
    print('[DEBUG] Starting continuous approach_and_land')
    # get initial NED and yaw
    async for od in drone.telemetry.position_velocity_ned():
        north0 = od.position.north_m
        east0  = od.position.east_m
        down0  = od.position.down_m
        print('[DEBUG] Initial NED: N={:.2f}, E={:.2f}, D={:.2f}'.format(
            north0, east0, down0))
        break
    async for att in drone.telemetry.attitude_euler():
        yaw = att.yaw_deg
        print('[DEBUG] Initial yaw: {:.1f} deg'.format(yaw))
        break

    # start offboard at current position
    await drone.offboard.set_position_ned(
        PositionNedYaw(north0, east0, down0, yaw)
    )
    try:
        print('[DEBUG] Enabling offboard')
        await drone.offboard.start()
        print('[DEBUG] Offboard started')
    except OffboardError as e:
        print('[ERROR] Offboard start failed:', e)
        return

    # use the initial offset for first command
    offsets = [initial_offset]
    iteration = 0

    while True:
        iteration += 1
        print('[DEBUG] approach iteration {}'.format(iteration))

        # current state
        async for od in drone.telemetry.position_velocity_ned():
            north = od.position.north_m
            east  = od.position.east_m
            down  = od.position.down_m
            print('[DEBUG] Current NED: N={:.2f}, E={:.2f}, D={:.2f}'.format(
                north, east, down))
            break
        async for att in drone.telemetry.attitude_euler():
            yaw = att.yaw_deg
            print('[DEBUG] Current yaw: {:.1f} deg'.format(yaw))
            break

        # get new offset if not first iteration
        if iteration > 1:
            print('[DEBUG] Searching for updated offset')
            offset = await search_marker(timeout=5.0)
            if offset is None:
                print('[DEBUG] No update offset, retrying')
                continue
        else:
            offset = initial_offset

        dx_e, dx_n, _ = offset
        target_n = north + dx_n
        target_e = east  + dx_e
        print('[DEBUG] Computed target NED: N={:.2f}, E={:.2f}'.format(
            target_n, target_e))

        # send setpoint
        print('[DEBUG] Sending position setpoint')
        await drone.offboard.set_position_ned(
            PositionNedYaw(target_n, target_e, down0, yaw)
        )

        # wait a bit for move
        await asyncio.sleep(1.0)

        # check error
        async for od in drone.telemetry.position_velocity_ned():
            err_n = abs(od.position.north_m - target_n)
            err_e = abs(od.position.east_m  - target_e)
            print('[DEBUG] Position error: N_err={:.3f}, E_err={:.3f}'.format(
                err_n, err_e))
            break

        if err_n < TOLERANCE and err_e < TOLERANCE:
            print('[DEBUG] Within tolerance, landing now')
            await drone.offboard.stop()
            await drone.action.land()
            print('[DEBUG] Land command sent')
            return
        else:
            print('[DEBUG] Not within tolerance, continue looping')

async def run():
    drone = None
    try:
        drone = await connect_and_arm()
        for lat, lon in coordinates:
            print('[DEBUG] Heading to waypoint ({}, {})'.format(lat, lon))
            await drone.action.goto_location(lat, lon, AMSL_ALTITUDE, 0.0)
            await asyncio.sleep(7)

            print('[DEBUG] Attempting initial marker search')
            tvec = await search_marker(timeout=10.0)
            if tvec is not None:
                await approach_and_land(drone, tvec)
                return

        print('[DEBUG] No marker found on any waypoint, returning to launch')
        await drone.action.return_to_launch()

    except Exception as e:
        print('[ERROR] Exception in run():', e)
    finally:
        if drone:
            try:
                await drone.offboard.stop()
                print('[DEBUG] Offboard stopped in cleanup')
            except:
                pass
            await drone.action.land()
            print('[DEBUG] Cleanup land command sent')

if __name__ == '__main__':
    print('[DEBUG] Script start')
    asyncio.run(run())
    print('[DEBUG] Script exit')
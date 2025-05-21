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
# Waypoints
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
# Init Camera
# ----------------------------
cam_cfg = picam2.create_preview_configuration(
    raw={"size": (1640, 1232)},
    main={"format": "RGB888", "size": (write_width, write_height)}
)
picam2.configure(cam_cfg)
picam2.start()
print("[DEBUG] Camera configured and started")
time.sleep(2)
print("[DEBUG] Camera auto-exposure stabilized")

async def connect_and_arm():
    print("[DEBUG] Connecting to drone...")
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    print("[DEBUG] Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[DEBUG] Drone connected!")
            break

    print("[DEBUG] Waiting for global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("[DEBUG] Global position and home position OK")
            break

    print("[DEBUG] Arming drone...")
    await drone.action.arm()

    print(f"[DEBUG] Setting takeoff altitude to {ALTITUDE}m")
    await drone.action.set_takeoff_altitude(ALTITUDE)

    print("[DEBUG] Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(10)
    print("[DEBUG] Takeoff complete")

    return drone

async def search_marker(timeout=5.0):
    print(f"[DEBUG] search_marker: timeout={timeout}s")
    t0 = time.time()
    prev_gray = None
    frame_count = 0

    while time.time() - t0 < timeout:
        frame = await asyncio.to_thread(picam2.capture_array)
        frame_count += 1
        print(f"[DEBUG] Frame {frame_count}: captured")

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100,
                                          qualityLevel=0.01, minDistance=20)
            print(f"[DEBUG] goodFeaturesToTrack pts: {None if pts is None else len(pts)}")
            if pts is not None:
                curr, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                valid = np.count_nonzero(st.reshape(-1) == 1)
                print(f"[DEBUG] Optical flow valid pts: {valid}")
                if valid >= 6:
                    M, _ = cv2.estimateAffinePartial2D(
                        pts[st.reshape(-1) == 1], curr[st.reshape(-1) == 1]
                    )
                    if M is not None:
                        frame = cv2.warpAffine(frame, M, frame.shape[1::-1])
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        print("[DEBUG] Frame stabilized with warp")

        prev_gray = gray

        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=DETECT_PARAMS)
        print(f"[DEBUG] detectMarkers found ids: {None if ids is None else ids.flatten().tolist()}")

        if ids is not None and TARGET_ID in ids.flatten():
            print(f"[DEBUG] Marker {TARGET_ID} detected")
            idx = list(ids.flatten()).index(TARGET_ID)
            _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS
            )
            offset = tvecs[0][0]
            print(f"[DEBUG] Marker tvec offset: {offset}")
            return offset

        print("[DEBUG] Marker not found in this frame")

    print("[DEBUG] search_marker timed out")
    return None

async def approach_and_land(drone):
    print("[DEBUG] Starting approach_and_land loop")
    # start offboard with current pos/yaw
    async for od in drone.telemetry.position_velocity_ned():
        north0 = od.position.north_m
        east0  = od.position.east_m
        down0  = od.position.down_m
        print(f"[DEBUG] Initial NED: N={north0:.2f}, E={east0:.2f}, D={down0:.2f}")
        break
    async for att in drone.telemetry.attitude_euler():
        yaw = att.yaw_deg
        print(f"[DEBUG] Initial yaw: {yaw:.1f} deg")
        break

    try:
        print("[DEBUG] Setting initial offboard setpoint")
        await drone.offboard.set_position_ned(PositionNedYaw(north0, east0, down0, yaw))
        print("[DEBUG] Starting offboard mode")
        await drone.offboard.start()
        print("[DEBUG] Offboard started")
    except OffboardError as e:
        print(f"[ERROR] Failed to start offboard: {e}")
        return

    while True:
        print("[DEBUG] approach_and_land: new iteration")
        # get fresh state
        async for od in drone.telemetry.position_velocity_ned():
            north = od.position.north_m
            east  = od.position.east_m
            down  = od.position.down_m
            print(f"[DEBUG] Current NED: N={north:.2f}, E={east:.2f}, D={down:.2f}")
            break
        async for att in drone.telemetry.attitude_euler():
            yaw = att.yaw_deg
            print(f"[DEBUG] Current yaw: {yaw:.1f}")
            break

        print("[DEBUG] Searching for marker to update offset")
        offset = await search_marker(timeout=2.0)
        if offset is None:
            print("[DEBUG] No offset this iteration, retrying")
            continue

        print(f"[DEBUG] Received offset: x={offset[0]:.3f}, y={offset[1]:.3f}, z={offset[2]:.3f}")
        target_n = north + offset[1]
        target_e = east  + offset[0]
        print(f"[DEBUG] New target NED: N={target_n:.2f}, E={target_e:.2f}, D={down:.2f}")

        print("[DEBUG] Sending new offboard setpoint")
        await drone.offboard.set_position_ned(PositionNedYaw(target_n, target_e, down, yaw))
        print("[DEBUG] Setpoint sent, waiting 0.5s")
        await asyncio.sleep(0.5)

        async for od in drone.telemetry.position_velocity_ned():
            err_n = abs(od.position.north_m - target_n)
            err_e = abs(od.position.east_m  - target_e)
            print(f"[DEBUG] Position error: err_n={err_n:.3f}, err_e={err_e:.3f}")
            break

        if err_n < TOLERANCE and err_e < TOLERANCE:
            print("[DEBUG] Within tolerance, landing now")
            await drone.offboard.stop()
            await drone.action.land()
            print("[DEBUG] Land command sent")
            return
        else:
            print("[DEBUG] Not within tolerance, looping again")

async def run():
    print("[DEBUG] Script run() start")
    drone = await connect_and_arm()
    try:
        for idx, (lat, lon) in enumerate(coordinates):
            print(f"[DEBUG] Waypoint {idx+1}/{len(coordinates)}: goto ({lat}, {lon})")
            await drone.action.goto_location(lat, lon, AMSL_ALTITUDE, 0.0)
            await asyncio.sleep(7)
            print(f"[DEBUG] Arrived at waypoint, entering approach_and_land")
            await approach_and_land(drone)
            print("[DEBUG] approach_and_land returned, exiting run()")
            return
        print("[DEBUG] No marker found on any waypoint, returning to launch")
        await drone.action.return_to_launch()
    except Exception as e:
        print(f"[ERROR] Exception in run(): {e}")
    finally:
        try:
            await drone.offboard.stop()
            print("[DEBUG] Offboard stopped in cleanup")
        except:
            pass
        await drone.action.land()
        print("[DEBUG] Land command sent in cleanup")

if __name__ == "__main__":
    print("[DEBUG] Script start")
    asyncio.run(run())
    print("[DEBUG] Script exit")
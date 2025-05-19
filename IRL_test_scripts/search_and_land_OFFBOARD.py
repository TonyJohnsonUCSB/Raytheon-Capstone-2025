#!/usr/bin/env python3

import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw
from mavsdk.geofence import Point, Polygon, FenceType, GeofenceData

# -- Calibration (hard‑coded) --
INTRINSIC = np.array([
    [653.1070007239106,   0.0,               339.2952147845755],
    [0.0,                 650.7753992788821, 258.1165494889447],
    [0.0,                 0.0,               1.0]
], dtype=np.float32)
DIST_COEFFS = np.array([
    -0.03887864427953473,
     0.6888798469690414,
     0.00815702400928161,
     0.010438854120041072,
    -1.713270699000528
], dtype=np.float32)

# -- ArUco params --
ARUCO_DICT   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
DETECT_PARAMS = cv2.aruco.DetectorParameters_create()
DETECT_PARAMS.adaptiveThreshConstant = 7
DETECT_PARAMS.minMarkerPerimeterRate  = 0.03
MARKER_SIZE   = 0.06611  # meters
TARGET_ID     = 1

# -- Flight & tolerance params --
ALTITUDE      = 5    # meters for takeoff and waypoints
TOLERANCE     = 0.10  # meters tolerance in N/E before landing

# -- Waypoints & geofence --
coordinates = [
    (34.418953, -119.855332),
    (34.418948, -119.855281),
    (34.418945, -119.855245),
    (34.418942, -119.855215)
]
geofence_points = [
    Point(34.418606, -119.855929),
    Point(34.418600, -119.855196),
    Point(34.419221, -119.855198),
    Point(34.419228, -119.855931),
    Point(34.418606, -119.855929)
]

# -- Camera setup --
picam2 = Picamera2()
cam_cfg = picam2.create_preview_configuration(
    raw  = {"size": (1640, 1232)},
    main = {"format": "RGB888", "size": (640, 480)}
)
picam2.configure(cam_cfg)
picam2.start()
print("[DEBUG] Camera started with RGB888 preview at 640x480")
# allow auto‑exposure to settle
time.sleep(2)

# -- Recording globals --
writer = None
record_enabled = False
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = '/home/rtxcapstone/Desktop/searchAndLandTest2.avi'

async def connect_and_arm():
    print("[DEBUG] Connecting to drone...")
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[DEBUG] Drone connected")
            break
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("[DEBUG] Drone healthy: global position and home position OK")
            break
    print("[DEBUG] Uploading geofence...")
    poly = Polygon(geofence_points, FenceType.INCLUSION)
    gf = GeofenceData(polygons=[poly], circles=[])
    await drone.geofence.upload_geofence(gf)
    print("[DEBUG] Geofence uploaded")
    await drone.action.arm()
    print("[DEBUG] Drone armed")
    await drone.action.set_takeoff_altitude(ALTITUDE)
    print(f"[DEBUG] Takeoff altitude set to {ALTITUDE} meters")

    # start recording at takeoff
    global writer, record_enabled
    writer = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
    record_enabled = True
    print("[DEBUG] Recording started")

    print("[DEBUG] Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(10)
    print("[DEBUG] Takeoff complete, hovering at altitude")
    return drone

async def search_marker(timeout=10.0):
    print(f"[DEBUG] Starting marker search for up to {timeout} seconds")
    t0 = time.time()
    prev_gray = None
    while time.time() - t0 < timeout:
        frame = await asyncio.to_thread(picam2.capture_array)
        if record_enabled:
            writer.write(frame)
            print("[DEBUG] Recorded frame during search")

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100,
                                          qualityLevel=0.01, minDistance=20)
            if pts is not None:
                curr, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                valid = st.reshape(-1) == 1
                print(f"[DEBUG] Optical flow valid points: {np.count_nonzero(valid)}")
                if np.count_nonzero(valid) >= 6:
                    M, _ = cv2.estimateAffinePartial2D(pts[valid], curr[valid])
                    if M is not None:
                        frame = cv2.warpAffine(frame, M, frame.shape[1::-1])
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        print("[DEBUG] Applied stabilization warp to frame")
        prev_gray = gray

        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=DETECT_PARAMS)
        if ids is not None and TARGET_ID in ids:
            print(f"[DEBUG] Detected target ArUco ID {TARGET_ID}")
            idx = list(ids.flatten()).index(TARGET_ID)
            _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS
            )
            offset = tvecs[0][0]
            print(f"[DEBUG] Pose estimated, offset = {offset}")
            return offset
        else:
            print("[DEBUG] Target marker not found in this frame")
    print("[DEBUG] Marker search timed out without detection")
    return None

async def approach_and_land(drone, offset):
    print("[DEBUG] Beginning approach and landing sequence")
    async for odom in drone.telemetry.position_velocity_ned():
        north0 = odom.position.north_m
        east0  = odom.position.east_m
        down0  = odom.position.down_m
        print(f"[DEBUG] Current NED: north={north0}, east={east0}, down={down0}")
        break
    async for att in drone.telemetry.attitude_euler():
        yaw = att.yaw_deg
        print(f"[DEBUG] Current yaw: {yaw} deg")
        break

    print("[DEBUG] Sending initial offboard hold setpoint")
    await drone.offboard.set_position_ned(PositionNedYaw(north0, east0, down0, yaw))
    try:
        print("[DEBUG] Starting offboard mode")
        await drone.offboard.start()
    except OffboardError:
        print("[ERROR] Offboard start failed, aborting landing approach")
        return

    target_n = north0 + offset[1]
    target_e = east0  + offset[0]
    print(f"[DEBUG] Computed target NED: north={target_n}, east={target_e}, down={down0}")

    print("[DEBUG] Sending offboard target setpoint")
    await drone.offboard.set_position_ned(
        PositionNedYaw(target_n, target_e, down0, yaw)
    )

    while True:
        await asyncio.sleep(0.5)
        async for od in drone.telemetry.position_velocity_ned():
            err_n = abs(od.position.north_m - target_n)
            err_e = abs(od.position.east_m  - target_e)
            print(f"[DEBUG] Position error: north_err={err_n}, east_err={err_e}")
            break
        if err_n < TOLERANCE and err_e < TOLERANCE:
            print("[DEBUG] Within tolerance, stopping offboard and landing")
            try:
                await drone.offboard.stop()
            except OffboardError:
                print("[WARN] Error stopping offboard")
            await drone.action.land()
            print("[DEBUG] Land command sent")

            # stop recording at landing
            global writer, record_enabled
            if record_enabled:
                writer.release()
                record_enabled = False
                print("[DEBUG] Recording stopped")
            return

async def run():
    drone = None
    try:
        drone = await connect_and_arm()
        for lat, lon in coordinates:
            print(f"[DEBUG] Navigating to waypoint ({lat}, {lon}) at {ALTITUDE}m")
            await drone.action.goto_location(lat, lon, ALTITUDE, 0.0)
            await asyncio.sleep(15)
            print(f"[DEBUG] Searching for marker at {lat},{lon}")
            tvec = await search_marker(10.0)
            if tvec is not None:
                print("[DEBUG] Marker found, proceeding to approach_and_land")
                await approach_and_land(drone, tvec)
                return
        print("[DEBUG] No marker found at any waypoint, returning to launch")
        await drone.action.return_to_launch()
    except Exception as e:
        print(f"[ERROR] Encountered exception: {e}")
    finally:
        # fallback: stop recording if still active
        global writer, record_enabled
        if record_enabled and writer is not None:
            writer.release()
            record_enabled = False
            print("[DEBUG] Recording stopped (fallback)")
        if drone is not None:
            try:
                print("[DEBUG] Cleaning up: stopping offboard mode")
                await drone.offboard.stop()
            except:
                pass
            try:
                print("[DEBUG] Cleaning up: sending land command")
                await drone.action.land()
            except:
                pass
        print("[DEBUG] Run complete or aborted")

if __name__ == "__main__":
    try:
        print("[DEBUG] Script started")
        asyncio.run(run())
    finally:
        print("[DEBUG] Exiting script")

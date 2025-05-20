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
# Camera and Recording Globals
# ----------------------------
picam2 = Picamera2()
write_width, write_height = 640, 480
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = '/home/rtxcapstone/Desktop/520searchandland1.avi'
writer = None
record_enabled = False

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
TOLERANCE = 0.01  # N/E position tolerance for landing in meters

# ----------------------------
# Waypoints and Geofence
# ----------------------------
coordinates = [
    (34.418953, -119.855332),
    (34.418948, -119.855281),
    (34.418945, -119.855245),
    (34.418942, -119.855215)
]

GEOFENCE_POINTS = [
    Point(34.418606, -119.855929),
    Point(34.418600, -119.855196),
    Point(34.419221, -119.855198),
    Point(34.419228, -119.855931),
    Point(34.418606, -119.855929)
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


def start_recording():
    """Begin video recording."""
    global writer, record_enabled
    fourcc    = cv2.VideoWriter_fourcc(*"XVID")
    writer       = cv2.VideoWriter("/home/rtxcapstone/Desktop/520serachandland.avi",
                            fourcc, 20.0, (640, 480))
    record_enabled = True
    print(f"[DEBUG] Recording started: {output_path}")


def stop_recording():
    """Stop video recording and release resources."""
    global writer, record_enabled
    if record_enabled and writer is not None:
        writer.release()
        record_enabled = False
        print("[DEBUG] Recording stopped")


async def connect_and_arm():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")  # Connect to the drone via serial port


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
    await drone.action.set_takeoff_altitude(3.0)  # Set altitude to 5 meters
    await drone.action.takeoff()

    await asyncio.sleep(10)

    return drone


async def search_marker(timeout=10.0):
    """Capture frames and look for the target ArUco marker."""
    print(f"[DEBUG] Searching for marker (timeout: {timeout} seconds)")
    t0 = time.time()
    prev_gray = None

    while (time.time() - t0) < timeout:
        frame = await asyncio.to_thread(picam2.capture_array)
        if record_enabled:
            writer.write(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100,
                                          qualityLevel=0.01, minDistance=20)
            if pts is not None:
                curr, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                valid_count = np.count_nonzero(st.reshape(-1) == 1)
                print(f"[DEBUG] Optical flow valid points: {valid_count}")
                if valid_count >= 6:
                    M, _ = cv2.estimateAffinePartial2D(pts[st.reshape(-1) == 1],
                                                     curr[st.reshape(-1) == 1])
                    if M is not None:
                        frame = cv2.warpAffine(frame, M, frame.shape[1::-1])
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        print("[DEBUG] Applied stabilization warp")
        prev_gray = gray

        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT,
                                                 parameters=DETECT_PARAMS)
        if ids is not None and TARGET_ID in ids.flatten():
            print(f"[DEBUG] Marker ID {TARGET_ID} detected")
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


async def approach_and_land(drone, offset):
    """Fly to the marker offset in NED frame and land."""
    
    print("[DEBUG] Starting approach and landing sequence")

    async for od in drone.telemetry.position_velocity_ned():
        north0, east0, down0 = od.position.north_m, od.position.east_m, od.position.down_m
        print(f"[DEBUG] Current NED -> N: {north0:.2f}, E: {east0:.2f}, D: {down0:.2f}")
        break

    async for att in drone.telemetry.attitude_euler():
        yaw = att.yaw_deg
        print(f"[DEBUG] Current yaw: {yaw:.1f} deg")
        break

    await drone.offboard.set_position_ned(PositionNedYaw(north0, east0, down0, yaw))
    try:
        print("[DEBUG] Enabling offboard mode")
        await drone.offboard.start()
    except OffboardError:
        print("[ERROR] Offboard start failed")
        return

    target_n = north0 + offset[1] 
    target_e = east0 + offset[0] 
    print(f"[DEBUG] Commanding move to N: {target_n:.2f}, E: {target_e:.2f}")

    await drone.offboard.set_position_ned(
        PositionNedYaw(target_n, target_e, down0, yaw)
    )

    while True:
        await asyncio.sleep(0.5)
        async for od in drone.telemetry.position_velocity_ned():
            err_n = abs(od.position.north_m - target_n)
            err_e = abs(od.position.east_m - target_e)
            print(f"[DEBUG] Err -> N: {err_n:.2f}, E: {err_e:.2f}")
            break
        if err_n < TOLERANCE and err_e < TOLERANCE:
            print("[DEBUG] Within tolerance, initiating landing")
            await drone.offboard.stop()
            await drone.action.land()
            stop_recording()
            return


async def run():
    drone = None
    try:
        drone = await connect_and_arm()
        for lat, lon in coordinates:
            print(f"[DEBUG] Heading to waypoint ({lat}, {lon}) at {ALTITUDE} meters")
            await drone.action.goto_location(lat, lon, 12, 0.0)
            await asyncio.sleep(15)
            print(f"[DEBUG] Attempting marker search at ({lat}, {lon})")
            tvec = await search_marker(10.0)
            if tvec is not None:
                await approach_and_land(drone, tvec)
                return

        print("[DEBUG] No marker found; returning to launch")
        await drone.action.return_to_launch()
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
    finally:
        stop_recording()
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

#!/usr/bin/env python3

import asyncio
import time
import cv2
import numpy as np
import math
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw
import serial

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
ARUCO_DICT    = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
DETECT_PARAMS = cv2.aruco.DetectorParameters_create()
DETECT_PARAMS.adaptiveThreshConstant    = 7
DETECT_PARAMS.minMarkerPerimeterRate     = 0.03
MARKER_SIZE = 0.06611  # meters
TARGET_ID   = 2

# ----------------------------
# Flight Parameters
# ----------------------------
ALTITUDE       = 5        # takeoff and waypoint altitude in meters
AMSL_ALTITUDE  = ALTITUDE + 9
TOLERANCE      = 0.01     # N/E position tolerance for landing in meters
VELOCITY       = 0.5      # m/s

# ----------------------------
# Waypoints and Offset Landing Spot
# ----------------------------
coordinates = [
    (34.4189,   -119.85533),
    (34.4189,   -119.85530),
    (34.4189,   -119.85528),
    (34.4189,   -119.85526),
    (34.4189,   -119.85524),
    (34.4189,   -119.85522),
    (34.4189,   -119.85520),
]

# Fixed landing spot: 10 yards (9.144 m) SE of first waypoint
OFFSET_LAT, OFFSET_LON = 34.41884192, -119.85525959

# ----------------------------
# Initialize Camera
# ----------------------------
cam_cfg = picam2.create_preview_configuration(
    raw   = {"size": (1640, 1232)},
    main  = {"format": "RGB888", "size": (write_width, write_height)},
)
picam2.configure(cam_cfg)
picam2.start()
print("[DEBUG] Camera started: preview at {}x{}".format(write_width, write_height))
time.sleep(2)

ser = serial.Serial(port='/dev/ttyUSB0',baudrate=57600)

async def get_gps_coordinates_from_drone(drone):
    async for pos in drone.telemetry.position():
        return round(pos.latitude_deg, 6), round(pos.longitude_deg, 6)


async def connect_and_arm():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected")
            break

    print("Waiting for global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Position OK")
            break

    print("-- Arming")
    await drone.action.arm()
    print("-- Taking off")
    await drone.action.set_takeoff_altitude(ALTITUDE)
    await drone.action.takeoff()
    await asyncio.sleep(10)

    return drone

async def search_marker(timeout=5.0):
    print(f"[DEBUG] Searching for marker (timeout={timeout}s)")
    t0 = time.time()
    prev_gray = None

    while (time.time() - t0) < timeout:
        frame = await asyncio.to_thread(picam2.capture_array)
        gray  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100,
                                          qualityLevel=0.01, minDistance=20)
            if pts is not None:
                curr, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                valid = np.count_nonzero(st.reshape(-1))
                if valid >= 6:
                    M, _ = cv2.estimateAffinePartial2D(
                        pts[st.reshape(-1) == 1],
                        curr[st.reshape(-1) == 1]
                    )
                    if M is not None:
                        frame = cv2.warpAffine(frame, M, frame.shape[1::-1])
                        gray  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        prev_gray = gray
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT,
                                                 parameters=DETECT_PARAMS)
        if ids is not None and TARGET_ID in ids.flatten():
            idx = list(ids.flatten()).index(TARGET_ID)
            _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS
            )
            offset = tvecs[0][0]
            print(f"[DEBUG] Marker {TARGET_ID} found, offset={offset}")
            return offset

        print("[DEBUG] Marker not in frame")
    print("[DEBUG] Marker search timed out")
    return None

async def approach_and_land(drone):
    print("[DEBUG] Starting continuous approach")
    async for att in drone.telemetry.attitude_euler():
        yaw = att.yaw_deg
        break

    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, yaw))
    try:
        await drone.offboard.start()
        print("[DEBUG] Offboard enabled")
    except OffboardError as e:
        print(f"[ERROR] Offboard start failed: {e}")
        return

    while True:
        async for od in drone.telemetry.position_velocity_ned():
            north, east, down = od.position.north_m, od.position.east_m, od.position.down_m
            break

        print("[DEBUG] Searching marker for offset")
        offset = await search_marker(timeout=2.0)
        if offset is None:
            continue

        dx_e, dx_n = offset[0], offset[1]
        dist = math.hypot(dx_n, dx_e)
        print(f"[DEBUG] Distance to marker: {dist:.3f}m")

        if dist < TOLERANCE:
        #print("[DEBUG] Within tolerance → landing")
            print("Drone within N and E tolerance. Sending GPS location.")
            latitude, longitude = await get_gps_coordinates_from_drone(drone)
            coordinates = f"{latitude},{longitude}\n".encode('utf-8')
            loop=0
            while loop<500:
                print(f"Sending GPS location: {coordinates.decode().strip()}.")
                ser.write(coordinates)
                loop += 1
            await drone.offboard.stop()
            await drone.action.land()
            return

        vn = -(dx_n/dist) * VELOCITY * 0.5
        ve =  (dx_e/dist) * VELOCITY * 0.5
        print(f"[DEBUG] Vel cmd N={vn:.2f}, E={ve:.2f}")
        await drone.offboard.set_velocity_ned(VelocityNedYaw(vn, ve, 0, yaw))
        await asyncio.sleep(1.0)

async def run():
    drone = None
    try:
        drone = await connect_and_arm()

        for lat, lon in coordinates:
            print(f"[DEBUG] → Waypoint ({lat}, {lon}, {AMSL_ALTITUDE}")
            await drone.action.goto_location(lat, lon, AMSL_ALTITUDE, 0.0)
            await asyncio.sleep(7)

            tvec = await search_marker(10.0)
            if tvec is not None:
                await approach_and_land(drone)
                return

        # no marker found → go to precomputed SE offset landing spot
        print(f"[DEBUG] No marker → heading to fixed SE offset ({OFFSET_LAT}, {OFFSET_LON})")
        await drone.action.goto_location(OFFSET_LAT, OFFSET_LON, AMSL_ALTITUDE, 0.0)
        await asyncio.sleep(10)

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

if __name__ == "__main__":
    print("[DEBUG] Starting mission")
    asyncio.run(run())
    print("[DEBUG] Mission script exit")

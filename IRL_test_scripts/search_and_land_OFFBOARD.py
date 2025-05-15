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
ALTITUDE      = 10.0     # meters for takeoff and waypoints
TOLERANCE     = 0.10     # meters tolerance in N/E before landing

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
    Point(34.419228, -119.855931)
]

# -- Camera setup --
picam2 = Picamera2()
cam_cfg = picam2.create_preview_configuration(
    raw  = {"size": (1640, 1232)},
    main = {"format": "RGB888", "size": (640, 480)}
)
picam2.configure(cam_cfg)
picam2.start()
time.sleep(2)

# -- Video writer to record preview --
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(
    '/home/rtxcapstone/Desktop/testVideo.avi',
    fourcc,
    20.0,
    (640, 480)
)

async def connect_and_arm():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            break
    # upload geofence
    poly = Polygon(geofence_points, FenceType.INCLUSION)
    gf = GeofenceData(polygons=[poly], circles=[])
    await drone.geofence.upload_geofence(gf)
    await drone.action.arm()
    await drone.action.set_takeoff_altitude(ALTITUDE)
    await drone.action.takeoff()
    await asyncio.sleep(10)
    return drone

async def search_marker(timeout=10.0):
    t0 = time.time()
    prev_gray = None
    while time.time() - t0 < timeout:
        frame = await asyncio.to_thread(picam2.capture_array)
        writer.write(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100,
                                          qualityLevel=0.01, minDistance=20)
            if pts is not None:
                curr, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                valid = st.reshape(-1) == 1
                if np.count_nonzero(valid) >= 6:
                    M, _ = cv2.estimateAffinePartial2D(pts[valid], curr[valid])
                    if M is not None:
                        frame = cv2.warpAffine(frame, M, frame.shape[1::-1])
        prev_gray = gray

        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray2, ARUCO_DICT, parameters=DETECT_PARAMS)
        if ids is not None and TARGET_ID in ids:
            idx = list(ids.flatten()).index(TARGET_ID)
            _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS
            )
            return tvecs[0][0]
    return None

async def approach_and_land(drone, offset):
    # Start offboard to move in N/E only, hold altitude
    await drone.offboard.set_position_ned(PositionNedYaw(0, 0, 0, 0))
    try:
        await drone.offboard.start()
    except OffboardError:
        return
    # get current NED
    async for odom in drone.telemetry.position_velocity_ned():
        north0 = odom.position.north_m
        east0  = odom.position.east_m
        down0  = odom.position.down_m
        break

    target_n = north0 + offset[1]
    target_e = east0  + offset[0]
    # command move to target N/E, keep down0
    await drone.offboard.set_position_ned(
        PositionNedYaw(target_n, target_e, down0, 0.0)
    )

    # wait until within tolerance
    while True:
        await asyncio.sleep(0.5)
        async for od in drone.telemetry.position_velocity_ned():
            err_n = abs(od.position.north_m - target_n)
            err_e = abs(od.position.east_m  - target_e)
            break
        if err_n < TOLERANCE and err_e < TOLERANCE:
            print("Reached within tolerance, landing...")
            try:
                await drone.offboard.stop()
            except OffboardError:
                pass
            await drone.action.land()
            return

async def run():
    drone = await connect_and_arm()
    for lat, lon in coordinates:
        await drone.action.goto_location(lat, lon, ALTITUDE, 0.0)
        await asyncio.sleep(15)
        print(f"Searching for marker at {lat},{lon}")
        tvec = await search_marker(10.0)
        if tvec is not None:
            print("Marker found, approaching and landing...")
            await approach_and_land(drone, tvec)
            return
    await drone.action.return_to_launch()
    await asyncio.sleep(15)
    await drone.action.land()

if __name__ == "__main__":
    asyncio.run(run())
    writer.release()

import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw
import math
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
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
DETECT_PARAMS = cv2.aruco.DetectorParameters_create()
DETECT_PARAMS.adaptiveThreshConstant = 7
DETECT_PARAMS.minMarkerPerimeterRate = 0.03
MARKER_SIZE = 0.06611  # meters
TARGET_ID = 2

# ----------------------------
# Flight Parameters
# ----------------------------
ALTITUDE = 3       # takeoff and waypoint altitude (AGL)
TOLERANCE = 0.10   # 10 cm centering tolerance when approaching marker (m)
VELOCITY_MS = 0.2  # m/s horizontal speed during sweep
SERIAL_PORT = '/dev/ttyUSB0'
BAUDRATE = 57600
ser = serial.Serial(port=SERIAL_PORT, baudrate=BAUDRATE)

# ----------------------------
# Initial GPS Waypoint
# ----------------------------
FIRST_WP_LAT = 34.41870255
FIRST_WP_LON = -119.85509000

# ----------------------------
# Initialize Camera
# ----------------------------
print("[DEBUG] Configuring camera...")
cam_cfg = picam2.create_preview_configuration(
    raw={"size": (1640, 1232)},
    main={"format": "RGB888", "size": (write_width, write_height)}
)
picam2.configure(cam_cfg)
picam2.start()
print(f"[DEBUG] Camera started: RGB888 preview at {write_width}x{write_height}")
time.sleep(2)
print("[DEBUG] Camera auto-exposure should be stable now")


async def fetch_current_gps_coordinates(drone):
    async for pos in drone.telemetry.position():
        lat = round(pos.latitude_deg, 10)
        lon = round(pos.longitude_deg, 10)
        return lat, lon


async def initialize_drone_and_takeoff():
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
    await asyncio.sleep(5)
    return drone


async def detect_aruco_marker(timeout=2.0):
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
                valid = st.reshape(-1) == 1
                if np.count_nonzero(valid) >= 6:
                    M, _ = cv2.estimateAffinePartial2D(pts[valid], curr[valid])
                    if M is not None:
                        frame = cv2.warpAffine(frame, M, frame.shape[1::-1])
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        prev_gray = gray
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=DETECT_PARAMS)
        if ids is not None and TARGET_ID in ids.flatten():
            idx = list(ids.flatten()).index(TARGET_ID)
            _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS
            )
            return tvecs[0][0]  # [x, y, z]
        await asyncio.sleep(0.05)

    return None


async def approach_and_land(drone, initial_offset):
    # Fetch initial NED & yaw
    async for od in drone.telemetry.position_velocity_ned():
        north0, east0, down0 = od.position.north_m, od.position.east_m, od.position.down_m
        break
    async for att in drone.telemetry.attitude_euler():
        yaw = att.yaw_deg
        break

    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, yaw))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        return False

    # Approach marker until within tolerance
    target_n = north0 + initial_offset[1]
    target_e = east0 + initial_offset[0]
    while True:
        async for od in drone.telemetry.position_velocity_ned():
            cur_n, cur_e = od.position.north_m, od.position.east_m
            break
        dx = target_n - cur_n
        dy = target_e - cur_e
        dist = math.hypot(dx, dy)
        if dist <= TOLERANCE:
            break

        vx = (dx / dist) * VELOCITY_MS
        vy = (dy / dist) * VELOCITY_MS
        await drone.offboard.set_velocity_ned(VelocityNedYaw(vx, vy, 0.0, yaw))

        new_offset = await detect_aruco_marker(timeout=0.05)
        if new_offset is not None:
            # refine target if needed
            target_n = cur_n + new_offset[1]
            target_e = cur_e + new_offset[0]
        await asyncio.sleep(0.1)

    # Send GPS over serial 100 times
    latitude, longitude = await fetch_current_gps_coordinates(drone)
    coord_bytes = f"{latitude},{longitude}\n".encode("utf-8")
    for _ in range(100):
        ser.write(coord_bytes)
        await asyncio.sleep(0.05)

    # Move 5 m south (negative north) before landing
    async for od in drone.telemetry.position_velocity_ned():
        cur_n, cur_e, cur_d = od.position.north_m, od.position.east_m, od.position.down_m
        break
    south_target_n = cur_n - 5.0
    while True:
        async for od in drone.telemetry.position_velocity_ned():
            cur_n, cur_e = od.position.north_m, od.position.east_m
            break
        dx = south_target_n - cur_n
        dist = abs(dx)
        if dist <= TOLERANCE:
            break
        vx = -(VELOCITY_MS if dx < 0 else -VELOCITY_MS)
        await drone.offboard.set_velocity_ned(VelocityNedYaw(vx, 0.0, 0.0, yaw))
        await asyncio.sleep(0.1)

    try:
        await drone.offboard.stop()
    except OffboardError:
        pass
    await drone.action.land()
    return True


async def execute_mission():
    drone = None
    try:
        drone = await initialize_drone_and_takeoff()

        # Fly to first GPS waypoint
        async for hp in drone.telemetry.home():
            home_abs = hp.absolute_altitude_m
            break
        target_amsl = home_abs + ALTITUDE
        await drone.action.goto_location(FIRST_WP_LAT, FIRST_WP_LON, target_amsl, 0.0)
        await asyncio.sleep(7)

        # Fetch NED origin & yaw
        async for od in drone.telemetry.position_velocity_ned():
            north0, east0, down0 = od.position.north_m, od.position.east_m, od.position.down_m
            break
        async for att in drone.telemetry.attitude_euler():
            yaw = att.yaw_deg
            break

        # Start offboard in velocity mode
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, yaw))
        try:
            await drone.offboard.start()
        except OffboardError:
            await drone.action.return_to_launch()
            return

        # Square sweep: 5 m east/west, 1 m south steps
        current_n = north0
        current_e = east0
        leg_tolerance = 0.50  # 50 cm

        for _ in range(5):
            # Leg 1: 5 m east
            target_n = current_n
            target_e = current_e + 5.0
            while True:
                async for od in drone.telemetry.position_velocity_ned():
                    cur_n, cur_e = od.position.north_m, od.position.east_m
                    break
                dx = target_n - cur_n
                dy = target_e - cur_e
                dist = math.hypot(dx, dy)
                if dist <= leg_tolerance:
                    break
                vx = (dx / dist) * VELOCITY_MS
                vy = (dy / dist) * VELOCITY_MS
                await drone.offboard.set_velocity_ned(VelocityNedYaw(vx, vy, 0.0, yaw))

                offset = await detect_aruco_marker(timeout=0.05)
                if offset is not None:
                    await drone.offboard.stop()
                    await approach_and_land(drone, offset)
                    return
                await asyncio.sleep(0.1)

            # Leg 2: 1 m south (negative north)
            target_n = current_n - 1.0
            target_e = current_e + 5.0
            while True:
                async for od in drone.telemetry.position_velocity_ned():
                    cur_n, cur_e = od.position.north_m, od.position.east_m
                    break
                dx = target_n - cur_n
                dy = target_e - cur_e
                dist = math.hypot(dx, dy)
                if dist <= leg_tolerance:
                    break
                vx = (dx / dist) * VELOCITY_MS
                vy = (dy / dist) * VELOCITY_MS
                await drone.offboard.set_velocity_ned(VelocityNedYaw(vx, vy, 0.0, yaw))

                offset = await detect_aruco_marker(timeout=0.05)
                if offset is not None:
                    await drone.offboard.stop()
                    await approach_and_land(drone, offset)
                    return
                await asyncio.sleep(0.1)

            # Leg 3: 5 m west (back to current_e)
            target_n = current_n - 1.0
            target_e = current_e
            while True:
                async for od in drone.telemetry.position_velocity_ned():
                    cur_n, cur_e = od.position.north_m, od.position.east_m
                    break
                dx = target_n - cur_n
                dy = target_e - cur_e
                dist = math.hypot(dx, dy)
                if dist <= leg_tolerance:
                    break
                vx = (dx / dist) * VELOCITY_MS
                vy = (dy / dist) * VELOCITY_MS
                await drone.offboard.set_velocity_ned(VelocityNedYaw(vx, vy, 0.0, yaw))

                offset = await detect_aruco_marker(timeout=0.05)
                if offset is not None:
                    await drone.offboard.stop()
                    await approach_and_land(drone, offset)
                    return
                await asyncio.sleep(0.1)

            # Prepare next row
            current_n = current_n - 1.0
            # current_e stays the same

        await drone.offboard.stop()
        await drone.action.return_to_launch()

    except Exception:
        if drone:
            try:
                await drone.offboard.stop()
            except:
                pass
            await drone.action.land()
    finally:
        if drone:
            try:
                await drone.offboard.stop()
            except:
                pass
            await drone.action.land()


if __name__ == '__main__':
    asyncio.run(execute_mission())

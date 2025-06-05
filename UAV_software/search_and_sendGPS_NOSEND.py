import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw
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
    print("[DEBUG] fetch_current_gps_coordinates: waiting for one GPS fix")
    async for pos in drone.telemetry.position():
        lat = round(pos.latitude_deg, 10)
        lon = round(pos.longitude_deg, 10)
        print(f"[DEBUG] fetch_current_gps_coordinates: got lat={lat}, lon={lon}")
        return lat, lon


async def initialize_drone_and_takeoff():
    print("[DEBUG] initialize_drone_and_takeoff: connecting to drone over serial")
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[DEBUG] initialize_drone_and_takeoff: drone connected")
            break

    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("[DEBUG] initialize_drone_and_takeoff: GPS and home position OK")
            break

    print("[DEBUG] initialize_drone_and_takeoff: arming")
    await drone.action.arm()
    print(f"[DEBUG] initialize_drone_and_takeoff: setting takeoff altitude to {ALTITUDE} m")
    await drone.action.set_takeoff_altitude(ALTITUDE)
    print("[DEBUG] initialize_drone_and_takeoff: taking off")
    await drone.action.takeoff()
    await asyncio.sleep(5)

    return drone


async def detect_aruco_marker(timeout=2.0):
    print(f"[DEBUG] detect_aruco_marker: starting detection with timeout={timeout}s")
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
                        print("[DEBUG] detect_aruco_marker: applied frame stabilization")

        prev_gray = gray
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=DETECT_PARAMS)
        if ids is not None and TARGET_ID in ids.flatten():
            idx = list(ids.flatten()).index(TARGET_ID)
            print(f"[DEBUG] detect_aruco_marker: found target ID {TARGET_ID} in frame")
            _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS
            )
            offset = tvecs[0][0]  # [x, y, z] in meters
            print(f"[DEBUG] detect_aruco_marker: estimated offset = {offset}")
            return offset
        await asyncio.sleep(0.05)

    print("[DEBUG] detect_aruco_marker: timeout reached, no marker found")
    return None


async def approach_and_land(drone, initial_offset):
    print(f"[DEBUG] approach_and_land: starting with initial_offset = {initial_offset}")

    # Fetch initial NED and yaw
    async for od in drone.telemetry.position_velocity_ned():
        north0, east0, down0 = od.position.north_m, od.position.east_m, od.position.down_m
        print(f"[DEBUG] approach_and_land: current NED = ({north0:.2f}, {east0:.2f}, {down0:.2f})")
        break
    async for att in drone.telemetry.attitude_euler():
        yaw = att.yaw_deg
        print(f"[DEBUG] approach_and_land: current yaw = {yaw:.2f}°")
        break

    print("[DEBUG] approach_and_land: setting initial offboard position hold")
    await drone.offboard.set_position_ned(PositionNedYaw(north0, east0, down0, yaw))
    try:
        await drone.offboard.start()
        print("[DEBUG] approach_and_land: offboard started")
    except OffboardError as e:
        print(f"[ERROR] approach_and_land: Failed to start offboard: {e}")
        return False

    # Move toward marker until within tolerance
    target_n = north0 + initial_offset[1]
    target_e = east0 + initial_offset[0]
    print(f"[DEBUG] approach_and_land: initial NED target = ({target_n:.2f}, {target_e:.2f}, {down0:.2f})")

    while True:
        await drone.offboard.set_position_ned(PositionNedYaw(target_n, target_e, down0, yaw))
        await asyncio.sleep(0.5)

        new_offset = await detect_aruco_marker(timeout=2.0)
        if new_offset is None:
            print("[DEBUG] approach_and_land: no new offset, retrying")
            continue

        dx, dy = new_offset[0], new_offset[1]
        dist = math.hypot(dy, dx)
        print(f"[DEBUG] approach_and_land: detected offset = (dx={dx:.3f}, dy={dy:.3f}), dist = {dist:.3f} m")

        if dist <= TOLERANCE:
            print(f"[DEBUG] approach_and_land: within tolerance ({dist:.3f} ≤ {TOLERANCE}), preparing to send GPS")
            break

        target_n += dy
        target_e += dx
        print(f"[DEBUG] approach_and_land: updating target to ({target_n:.2f}, {target_e:.2f})")

    # Fetch final GPS
    latitude, longitude = await fetch_current_gps_coordinates(drone)
    coord_bytes = f"{latitude},{longitude}\n".encode("utf-8")

    # Send GPS over serial 100 times
    for i in range(100):
        print(f"[DEBUG] approach_and_land: Sending GPS location ({i+1}/100): {coord_bytes.decode().strip()}")
        ser.write(coord_bytes)
        await asyncio.sleep(0.05)

    # Fly 5 meters south (negative north in NED)
    async for od in drone.telemetry.position_velocity_ned():
        cur_north, cur_east, cur_down = od.position.north_m, od.position.east_m, od.position.down_m
        break
    south_target_n = cur_north - 5.0
    print(f"[DEBUG] approach_and_land: moving 5 m south to ({south_target_n:.2f}, {cur_east:.2f}, {cur_down:.2f})")
    await drone.offboard.set_position_ned(PositionNedYaw(south_target_n, cur_east, cur_down, yaw))
    await asyncio.sleep(5)

    # Stop offboard and land
    try:
        await drone.offboard.stop()
        print("[DEBUG] approach_and_land: offboard stopped")
    except OffboardError:
        print("[ERROR] approach_and_land: offboard stop failed")
    await drone.action.land()
    print("[DEBUG] approach_and_land: landing sequence initiated")
    return True


def gps_to_ned_meters(lat_ref, lon_ref, lat, lon):
    dlat = lat - lat_ref
    dlon = lon - lon_ref
    meters_per_deg_lat = 111139.0
    meters_per_deg_lon = 111139.0 * math.cos(math.radians(lat_ref))
    north = dlat * meters_per_deg_lat
    east = dlon * meters_per_deg_lon
    return north, east


async def execute_mission():
    print("[DEBUG] execute_mission: starting")
    drone = None
    try:
        drone = await initialize_drone_and_takeoff()

        print("[DEBUG] execute_mission: flying to first GPS waypoint")
        async for hp in drone.telemetry.home():
            home_abs = hp.absolute_altitude_m
            print(f"[DEBUG] execute_mission: home AMSL altitude = {home_abs:.2f} m")
            break

        target_amsl = home_abs + ALTITUDE
        print(f"[DEBUG] execute_mission: goto_location(lat={FIRST_WP_LAT}, lon={FIRST_WP_LON}, alt={target_amsl:.2f})")
        await drone.action.goto_location(FIRST_WP_LAT, FIRST_WP_LON, target_amsl, 0.0)
        await asyncio.sleep(7)

        print("[DEBUG] execute_mission: fetching NED origin & yaw at first waypoint")
        async for od in drone.telemetry.position_velocity_ned():
            north0, east0, down0 = od.position.north_m, od.position.east_m, od.position.down_m
            print(f"[DEBUG] execute_mission: NED origin = ({north0:.2f}, {east0:.2f}, {down0:.2f})")
            break
        async for att in drone.telemetry.attitude_euler():
            yaw = att.yaw_deg
            print(f"[DEBUG] execute_mission: yaw at first waypoint = {yaw:.2f}°")
            break

        print("[DEBUG] execute_mission: setting initial offboard to hold position")
        await drone.offboard.set_position_ned(PositionNedYaw(north0, east0, down0, yaw))
        try:
            await drone.offboard.start()
            print("[DEBUG] execute_mission: offboard started")
        except OffboardError as e:
            print(f"[ERROR] execute_mission: offboard start failed: {e}")
            await drone.action.return_to_launch()
            return

        print("[DEBUG] execute_mission: beginning offboard sweep (5 m east/west, 1 m south steps)")
        current_n = north0
        current_e = east0
        leg_tolerance = 0.50  # 50 cm to consider "reached" for each leg

        for _ in range(5):
            # --- Leg 1: Go 5 m east ---
            target_n = current_n
            target_e = current_e + 5.0
            print(f"[DEBUG] execute_mission: Leg 1 → target NED = ({target_n:.2f}, {target_e:.2f}, {down0:.2f})")
            while True:
                # Fetch current position
                async for od in drone.telemetry.position_velocity_ned():
                    cur_n, cur_e = od.position.north_m, od.position.east_m
                    break
                dx = target_n - cur_n
                dy = target_e - cur_e
                dist = math.hypot(dx, dy)
                if dist <= leg_tolerance:
                    print(f"[DEBUG] execute_mission: Reached Leg 1 target (dist={dist:.2f} m)")
                    break

                await drone.offboard.set_position_ned(PositionNedYaw(target_n, target_e, down0, yaw))
                offset = await detect_aruco_marker(timeout=0.05)
                if offset is not None:
                    print("[DEBUG] execute_mission: Marker found on Leg 1, switching to approach_and_land")
                    await drone.offboard.stop()
                    await approach_and_land(drone, offset)
                    return
                await asyncio.sleep(0.1)

            # --- Leg 2: Go 1 m south (negative north) ---
            target_n = current_n - 1.0
            target_e = current_e + 5.0
            print(f"[DEBUG] execute_mission: Leg 2 → target NED = ({target_n:.2f}, {target_e:.2f}, {down0:.2f})")
            while True:
                async for od in drone.telemetry.position_velocity_ned():
                    cur_n, cur_e = od.position.north_m, od.position.east_m
                    break
                dx = target_n - cur_n
                dy = target_e - cur_e
                dist = math.hypot(dx, dy)
                if dist <= leg_tolerance:
                    print(f"[DEBUG] execute_mission: Reached Leg 2 target (dist={dist:.2f} m)")
                    break

                await drone.offboard.set_position_ned(PositionNedYaw(target_n, target_e, down0, yaw))
                offset = await detect_aruco_marker(timeout=0.05)
                if offset is not None:
                    print("[DEBUG] execute_mission: Marker found on Leg 2, switching to approach_and_land")
                    await drone.offboard.stop()
                    await approach_and_land(drone, offset)
                    return
                await asyncio.sleep(0.1)

            # --- Leg 3: Go 5 m west (back to original east) ---
            target_n = current_n - 1.0
            target_e = current_e
            print(f"[DEBUG] execute_mission: Leg 3 → target NED = ({target_n:.2f}, {target_e:.2f}, {down0:.2f})")
            while True:
                async for od in drone.telemetry.position_velocity_ned():
                    cur_n, cur_e = od.position.north_m, od.position.east_m
                    break
                dx = target_n - cur_n
                dy = target_e - cur_e
                dist = math.hypot(dx, dy)
                if dist <= leg_tolerance:
                    print(f"[DEBUG] execute_mission: Reached Leg 3 target (dist={dist:.2f} m)")
                    break

                await drone.offboard.set_position_ned(PositionNedYaw(target_n, target_e, down0, yaw))
                offset = await detect_aruco_marker(timeout=0.05)
                if offset is not None:
                    print("[DEBUG] execute_mission: Marker found on Leg 3, switching to approach_and_land")
                    await drone.offboard.stop()
                    await approach_and_land(drone, offset)
                    return
                await asyncio.sleep(0.1)

            # Prepare for next row
            current_n = current_n - 1.0
            # current_e remains the same

        print("[DEBUG] execute_mission: sweep complete without finding marker")
        try:
            await drone.offboard.stop()
            print("[DEBUG] execute_mission: offboard stopped")
        except OffboardError:
            pass
        print("[DEBUG] execute_mission: commanding Return to Launch (RTL)")
        await drone.action.return_to_launch()

    except Exception as e:
        print(f"[ERROR] execute_mission: exception occurred: {e}")
        if drone:
            try:
                await drone.offboard.stop()
                print("[DEBUG] execute_mission: offboard stopped in exception handler")
            except:
                pass
            await drone.action.land()
            print("[DEBUG] execute_mission: landing in exception handler")
    finally:
        if drone:
            try:
                await drone.offboard.stop()
            except:
                pass
            await drone.action.land()
            print("[DEBUG] execute_mission: cleanup complete, drone disarmed/landed (if necessary)")


if __name__ == '__main__':
    asyncio.run(execute_mission())

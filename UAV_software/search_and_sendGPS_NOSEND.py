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
print("[DEBUG] Initializing camera globals")
picam2 = Picamera2()
write_width, write_height = 640, 480

# ----------------------------
# Calibration and Distortion
# ----------------------------
print("[DEBUG] Setting up intrinsic and distortion coefficients")
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
print("[DEBUG] Configuring ArUco detection parameters")
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
DETECT_PARAMS = cv2.aruco.DetectorParameters_create()
DETECT_PARAMS.adaptiveThreshConstant = 7
DETECT_PARAMS.minMarkerPerimeterRate = 0.03
MARKER_SIZE = 0.06611  # meters
TARGET_ID = 2

# ----------------------------
# Flight Parameters
# ----------------------------
print("[DEBUG] Defining flight parameters")
ALTITUDE = 3       # takeoff and waypoint altitude (AGL)
TOLERANCE = 0.10   # 10 cm centering tolerance when approaching marker (m)
VELOCITY_MS = 0.2  # m/s horizontal speed during sweep
SERIAL_PORT = '/dev/ttyUSB0'
BAUDRATE = 57600
print(f"[DEBUG] Opening serial port {SERIAL_PORT} at baud {BAUDRATE}")
ser = serial.Serial(port=SERIAL_PORT, baudrate=BAUDRATE)

# ----------------------------
# Initial GPS Waypoint
# ----------------------------
print("[DEBUG] Setting first GPS waypoint")
FIRST_WP_LAT = 34.41870255
FIRST_WP_LON = -119.85509000

# ----------------------------
# Initialize Camera
# ----------------------------
print("[DEBUG] Configuring camera preview and main streams")
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
    print("[DEBUG] fetch_current_gps_coordinates: awaiting one GPS fix")
    async for pos in drone.telemetry.position():
        lat = round(pos.latitude_deg, 10)
        lon = round(pos.longitude_deg, 10)
        print(f"[DEBUG] fetch_current_gps_coordinates: latitude={lat}, longitude={lon}")
        return lat, lon


async def initialize_drone_and_takeoff():
    print("[DEBUG] initialize_drone_and_takeoff: creating System object and connecting")
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")
    print("[DEBUG] Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[DEBUG] initialize_drone_and_takeoff: drone connected")
            break

    print("[DEBUG] Waiting for health checks (GPS & home position)...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("[DEBUG] initialize_drone_and_takeoff: GPS and home position OK")
            break

    print("[DEBUG] initialize_drone_and_takeoff: commanding arm")
    await drone.action.arm()
    print(f"[DEBUG] initialize_drone_and_takeoff: setting takeoff altitude to {ALTITUDE} m")
    await drone.action.set_takeoff_altitude(ALTITUDE)
    print("[DEBUG] initialize_drone_and_takeoff: commanding takeoff")
    await drone.action.takeoff()
    await asyncio.sleep(5)
    print("[DEBUG] initialize_drone_and_takeoff: takeoff complete")
    return drone


async def detect_aruco_marker(timeout=2.0):
    print(f"[DEBUG] detect_aruco_marker: start detection with timeout={timeout}s")
    t0 = time.time()
    prev_gray = None

    while (time.time() - t0) < timeout:
        frame = await asyncio.to_thread(picam2.capture_array)
        print("[DEBUG] detect_aruco_marker: captured frame")
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        print("[DEBUG] detect_aruco_marker: converted to grayscale")

        if prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100,
                                          qualityLevel=0.01, minDistance=20)
            if pts is not None:
                print(f"[DEBUG] detect_aruco_marker: found {len(pts)} good features to track")
                curr, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                valid = st.reshape(-1) == 1
                print(f"[DEBUG] detect_aruco_marker: {np.count_nonzero(valid)} valid optical flow points")
                if np.count_nonzero(valid) >= 6:
                    M, _ = cv2.estimateAffinePartial2D(pts[valid], curr[valid])
                    if M is not None:
                        print("[DEBUG] detect_aruco_marker: applying affine stabilization")
                        frame = cv2.warpAffine(frame, M, frame.shape[1::-1])
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        print("[DEBUG] detect_aruco_marker: re-converted stabilized frame to grayscale")

        prev_gray = gray
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=DETECT_PARAMS)
        if ids is not None and TARGET_ID in ids.flatten():
            print(f"[DEBUG] detect_aruco_marker: detected marker IDs {ids.flatten()}")
            idx = list(ids.flatten()).index(TARGET_ID)
            print(f"[DEBUG] detect_aruco_marker: target index in IDs = {idx}")
            _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS
            )
            offset = tvecs[0][0]  # [x, y, z] in meters
            print(f"[DEBUG] detect_aruco_marker: marker offset = {offset}")
            return offset
        await asyncio.sleep(0.05)

    print("[DEBUG] detect_aruco_marker: timeout reached, no marker found")
    return None


async def approach_and_land(drone, initial_offset):
    print("[DEBUG] approach_and_land: initiated")
    # Fetch initial NED & yaw
    async for od in drone.telemetry.position_velocity_ned():
        north0, east0, down0 = od.position.north_m, od.position.east_m, od.position.down_m
        print(f"[DEBUG] approach_and_land: starting NED = ({north0:.2f}, {east0:.2f}, {down0:.2f})")
        break
    async for att in drone.telemetry.attitude_euler():
        yaw = att.yaw_deg
        print(f"[DEBUG] approach_and_land: starting yaw = {yaw:.2f}°")
        break

    print("[DEBUG] approach_and_land: setting initial offboard velocity to zero")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, yaw))
    try:
        await drone.offboard.start()
        print("[DEBUG] approach_and_land: offboard started")
    except OffboardError as e:
        print(f"[ERROR] approach_and_land: Failed to start offboard: {e}")
        return False

    # Compute initial target toward marker
    target_n = north0 + initial_offset[1]
    target_e = east0 + initial_offset[0]
    print(f"[DEBUG] approach_and_land: initial marker-target NED = ({target_n:.2f}, {target_e:.2f}, {down0:.2f})")

    # Approach loop
    while True:
        async for od in drone.telemetry.position_velocity_ned():
            cur_n, cur_e = od.position.north_m, od.position.east_m
            break
        dx = target_n - cur_n
        dy = target_e - cur_e
        dist = math.hypot(dx, dy)
        print(f"[DEBUG] approach_and_land: current NED = ({cur_n:.2f}, {cur_e:.2f}), dx={dx:.3f}, dy={dy:.3f}, dist={dist:.3f}")
        if dist <= TOLERANCE:
            print(f"[DEBUG] approach_and_land: within tolerance ({dist:.3f} ≤ {TOLERANCE}), breaking to send GPS")
            break

        vx = (dx / dist) * VELOCITY_MS
        vy = (dy / dist) * VELOCITY_MS
        print(f"[DEBUG] approach_and_land: commanding velocity vx={vx:.3f}, vy={vy:.3f}, vz=0.0, yaw={yaw:.2f}")
        await drone.offboard.set_velocity_ned(VelocityNedYaw(vx, vy, 0.0, yaw))

        new_offset = await detect_aruco_marker(timeout=0.05)
        if new_offset is not None:
            print(f"[DEBUG] approach_and_land: refining offset, new_offset={new_offset}")
            target_n = cur_n + new_offset[1]
            target_e = cur_e + new_offset[0]
            print(f"[DEBUG] approach_and_land: new marker-target NED = ({target_n:.2f}, {target_e:.2f})")
        await asyncio.sleep(0.1)

    # Fetch final GPS
    print("[DEBUG] approach_and_land: fetching final GPS coordinates")
    latitude, longitude = await fetch_current_gps_coordinates(drone)
    coord_bytes = f"{latitude},{longitude}\n".encode("utf-8")
    print(f"[DEBUG] approach_and_land: final GPS = ({latitude}, {longitude}), sending over serial")

    # Send GPS over serial 100 times
    for i in range(100):
        print(f"[DEBUG] approach_and_land: serial write {i+1}/100: {coord_bytes.decode().strip()}")
        ser.write(coord_bytes)
        await asyncio.sleep(0.05)

    # Move 5 m south (negative north) before landing
    print("[DEBUG] approach_and_land: preparing to move 5 m south before landing")
    async for od in drone.telemetry.position_velocity_ned():
        cur_n, cur_e, cur_d = od.position.north_m, od.position.east_m, od.position.down_m
        print(f"[DEBUG] approach_and_land: current NED before final move = ({cur_n:.2f}, {cur_e:.2f}, {cur_d:.2f})")
        break
    south_target_n = cur_n - 5.0
    print(f"[DEBUG] approach_and_land: final move target = ({south_target_n:.2f}, {cur_e:.2f}, {cur_d:.2f})")
    while True:
        async for od in drone.telemetry.position_velocity_ned():
            cur_n, cur_e = od.position.north_m, od.position.east_m
            break
        dx = south_target_n - cur_n
        dist = abs(dx)
        print(f"[DEBUG] approach_and_land: moving south, current north={cur_n:.2f}, target north={south_target_n:.2f}, dx={dx:.3f}, dist={dist:.3f}")
        if dist <= TOLERANCE:
            print(f"[DEBUG] approach_and_land: reached 5 m south target (dist={dist:.3f}), breaking")
            break
        vy = 0.0
        vx = (dx / dist) * VELOCITY_MS
        print(f"[DEBUG] approach_and_land: commanding final south velocity vx={vx:.3f}, vy={vy:.3f}, vz=0.0, yaw={yaw:.2f}")
        await drone.offboard.set_velocity_ned(VelocityNedYaw(vx, vy, 0.0, yaw))
        await asyncio.sleep(0.1)

    # Stop offboard and land
    print("[DEBUG] approach_and_land: stopping offboard and landing")
    try:
        await drone.offboard.stop()
        print("[DEBUG] approach_and_land: offboard stopped successfully")
    except OffboardError as e:
        print(f"[ERROR] approach_and_land: offboard stop failed: {e}")
    await drone.action.land()
    print("[DEBUG] approach_and_land: landing commanded")
    return True


def gps_to_ned_meters(lat_ref, lon_ref, lat, lon):
    print("[DEBUG] gps_to_ned_meters: converting latitude/longitude to NED")
    dlat = lat - lat_ref
    dlon = lon - lon_ref
    meters_per_deg_lat = 111139.0
    meters_per_deg_lon = 111139.0 * math.cos(math.radians(lat_ref))
    north = dlat * meters_per_deg_lat
    east = dlon * meters_per_deg_lon
    print(f"[DEBUG] gps_to_ned_meters: dlat={dlat:.6f}, dlon={dlon:.6f}, north={north:.2f}, east={east:.2f}")
    return north, east


async def execute_mission():
    print("[DEBUG] execute_mission: starting mission sequence")
    drone = None
    try:
        drone = await initialize_drone_and_takeoff()

        # Fly to first GPS waypoint
        print("[DEBUG] execute_mission: commanding goto_location to first GPS waypoint")
        async for hp in drone.telemetry.home():
            home_abs = hp.absolute_altitude_m
            print(f"[DEBUG] execute_mission: home AMSL altitude = {home_abs:.2f} m")
            break
        target_amsl = home_abs + ALTITUDE
        print(f"[DEBUG] execute_mission: goto_location(lat={FIRST_WP_LAT}, lon={FIRST_WP_LON}, alt={target_amsl:.2f}, yaw=0)")
        await drone.action.goto_location(FIRST_WP_LAT, FIRST_WP_LON, target_amsl, 0.0)
        await asyncio.sleep(7)
        print("[DEBUG] execute_mission: arrived at first GPS waypoint")

        # Fetch NED origin & yaw
        print("[DEBUG] execute_mission: fetching NED origin & yaw at first waypoint")
        async for od in drone.telemetry.position_velocity_ned():
            north0, east0, down0 = od.position.north_m, od.position.east_m, od.position.down_m
            print(f"[DEBUG] execute_mission: NED origin = ({north0:.2f}, {east0:.2f}, {down0:.2f})")
            break
        async for att in drone.telemetry.attitude_euler():
            yaw = att.yaw_deg
            print(f"[DEBUG] execute_mission: yaw at first waypoint = {yaw:.2f}°")
            break

        # Start offboard in velocity mode
        print("[DEBUG] execute_mission: setting initial offboard velocity to zero")
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, yaw))
        try:
            await drone.offboard.start()
            print("[DEBUG] execute_mission: offboard started")
        except OffboardError as e:
            print(f"[ERROR] execute_mission: offboard start failed: {e}")
            await drone.action.return_to_launch()
            return

        # Square sweep: 5 m east/west, 1 m south steps
        print("[DEBUG] execute_mission: beginning offboard sweep (5 m east/west, 1 m south steps)")
        current_n = north0
        current_e = east0
        leg_tolerance = 0.50  # 50 cm tolerance

        for row in range(5):
            print(f"[DEBUG] execute_mission: Starting row {row+1}/5; current origin NED = ({current_n:.2f}, {current_e:.2f}, {down0:.2f})")

            # Leg 1: 5 m east
            target_n = current_n
            target_e = current_e + 5.0
            print(f"[DEBUG] execute_mission: Leg 1 → target NED = ({target_n:.2f}, {target_e:.2f}, {down0:.2f})")
            while True:
                async for od in drone.telemetry.position_velocity_ned():
                    cur_n, cur_e = od.position.north_m, od.position.east_m
                    break
                dx = target_n - cur_n
                dy = target_e - cur_e
                dist = math.hypot(dx, dy)
                print(f"[DEBUG] execute_mission: Leg 1 loop: current NED=({cur_n:.2f}, {cur_e:.2f}), dx={dx:.3f}, dy={dy:.3f}, dist={dist:.3f}")
                if dist <= leg_tolerance:
                    print(f"[DEBUG] execute_mission: Leg 1 reached (dist={dist:.2f} m)")
                    break
                vx = (dx / dist) * VELOCITY_MS
                vy = (dy / dist) * VELOCITY_MS
                print(f"[DEBUG] execute_mission: Leg 1 commanding velocity vx={vx:.3f}, vy={vy:.3f}")
                await drone.offboard.set_velocity_ned(VelocityNedYaw(vx, vy, 0.0, yaw))

                offset = await detect_aruco_marker(timeout=0.05)
                if offset is not None:
                    print(f"[DEBUG] execute_mission: Marker found on Leg 1, offset={offset}")
                    await drone.offboard.stop()
                    print("[DEBUG] execute_mission: stopped offboard to switch to approach_and_land")
                    await approach_and_land(drone, offset)
                    return
                await asyncio.sleep(0.1)

            # Leg 2: 1 m south (negative north)
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
                print(f"[DEBUG] execute_mission: Leg 2 loop: current NED=({cur_n:.2f}, {cur_e:.2f}), dx={dx:.3f}, dy={dy:.3f}, dist={dist:.3f}")
                if dist <= leg_tolerance:
                    print(f"[DEBUG] execute_mission: Leg 2 reached (dist={dist:.2f} m)")
                    break
                vx = (dx / dist) * VELOCITY_MS
                vy = (dy / dist) * VELOCITY_MS
                print(f"[DEBUG] execute_mission: Leg 2 commanding velocity vx={vx:.3f}, vy={vy:.3f}")
                await drone.offboard.set_velocity_ned(VelocityNedYaw(vx, vy, 0.0, yaw))

                offset = await detect_aruco_marker(timeout=0.05)
                if offset is not None:
                    print(f"[DEBUG] execute_mission: Marker found on Leg 2, offset={offset}")
                    await drone.offboard.stop()
                    print("[DEBUG] execute_mission: stopped offboard to switch to approach_and_land")
                    await approach_and_land(drone, offset)
                    return
                await asyncio.sleep(0.1)

            # Leg 3: 5 m west (back to original east)
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
                print(f"[DEBUG] execute_mission: Leg 3 loop: current NED=({cur_n:.2f}, {cur_e:.2f}), dx={dx:.3f}, dy={dy:.3f}, dist={dist:.3f}")
                if dist <= leg_tolerance:
                    print(f"[DEBUG] execute_mission: Leg 3 reached (dist={dist:.2f} m)")
                    break
                vx = (dx / dist) * VELOCITY_MS
                vy = (dy / dist) * VELOCITY_MS
                print(f"[DEBUG] execute_mission: Leg 3 commanding velocity vx={vx:.3f}, vy={vy:.3f}")
                await drone.offboard.set_velocity_ned(VelocityNedYaw(vx, vy, 0.0, yaw))

                offset = await detect_aruco_marker(timeout=0.05)
                if offset is not None:
                    print(f"[DEBUG] execute_mission: Marker found on Leg 3, offset={offset}")
                    await drone.offboard.stop()
                    print("[DEBUG] execute_mission: stopped offboard to switch to approach_and_land")
                    await approach_and_land(drone, offset)
                    return
                await asyncio.sleep(0.1)

            # Prepare next row
            current_n = current_n - 1.0
            print(f"[DEBUG] execute_mission: Completed row {row+1}, moving to next origin NED = ({current_n:.2f}, {current_e:.2f})")

        print("[DEBUG] execute_mission: sweep complete without finding marker")
        try:
            await drone.offboard.stop()
            print("[DEBUG] execute_mission: offboard stopped after sweep")
        except OffboardError as e:
            print(f"[ERROR] execute_mission: offboard stop failed: {e}")
        print("[DEBUG] execute_mission: commanding Return to Launch (RTL)")
        await drone.action.return_to_launch()

    except Exception as e:
        print(f"[ERROR] execute_mission: exception occurred: {e}")
        if drone:
            try:
                await drone.offboard.stop()
                print("[DEBUG] execute_mission: offboard stopped in exception handler")
            except Exception as stop_err:
                print(f"[ERROR] execute_mission: exception stopping offboard: {stop_err}")
            await drone.action.land()
            print("[DEBUG] execute_mission: landing in exception handler")
    finally:
        if drone:
            try:
                await drone.offboard.stop()
                print("[DEBUG] execute_mission: offboard stopped in cleanup")
            except Exception:
                pass
            await drone.action.land()
            print("[DEBUG] execute_mission: cleanup complete, drone disarmed/landed if not already")


if __name__ == '__main__':
    print("[DEBUG] Running execute_mission via asyncio")
    asyncio.run(execute_mission())
    print("[DEBUG] Mission script has exited")

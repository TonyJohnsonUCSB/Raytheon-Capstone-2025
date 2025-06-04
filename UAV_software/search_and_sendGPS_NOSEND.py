import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw
import serial
import math

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
ALTITUDE = 5       # takeoff and waypoint altitude in meters (AGL)
TOLERANCE = 0.10   # 10 cm tolerance (meters)

# ----------------------------
# Waypoints and Geofence
# ----------------------------
#coordinates = [
 #   (34.4189,  -119.85533),
  #  (34.4189,  -119.85530),
   # (34.4189,  -119.85528),
   # (34.4189,  -119.85526),
   # (34.4189,  -119.85524),
   # (34.4189,  -119.85522),
   # (34.4189,  -119.85520),
#]
# coordinates = [
#     (34.4188664, -119.8559220),
#     (34.4188664, -119.855955),
#     (34.4188664, -119.856091),
#     (34.4188664, -119.8561966)
#     ]
coordinates = [
    (34.41870255, -119.85509000),
    (34.41870255, -119.85503363),
    (34.41870255, -119.85497727),
    (34.41866683, -119.85497727),
    (34.41866683, -119.85503363),
    (34.41866683, -119.85509000),
    (34.41863115, -119.85509000),
    (34.41863115, -119.85503363),
    (34.41863115, -119.85497727)
]


# First waypoint (to compute landing point)
FIRST_WP_LAT = 34.4189
FIRST_WP_LON = -119.85533

# Landing point: 5 meters west of first waypoint
LAND_LAT = 34.4189
LAND_LON = -119.8553844479  # computed separately

# Geofence (not modified)
GEOFENCE_POINTS = [
    (34.4186,  -119.85600),
    (34.4186,  -119.85475),
    (34.4192,  -119.85475),
    (34.4192,  -119.85600),
    (34.4186,  -119.85600),
]

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
time.sleep(2)  # allow auto-exposure to stabilize
print("[DEBUG] Camera auto-exposure should be stable now")

# ----------------------------
# Open LoRa communication
# ----------------------------
print("[DEBUG] Opening serial port /dev/ttyUSB0 at 57600 baud")
# ser = serial.Serial(port='/dev/ttyUSB0', baudrate=57600)
# print(f"[DEBUG] Serial port open: {ser.portstr}")
print("[DEBUG] Serial port open: fake serial code")

async def fetch_current_gps_coordinates(drone):
    print("[DEBUG] fetch_current_gps_coordinates: requesting GPS telemetry")
    async for pos in drone.telemetry.position():
        lat = round(pos.latitude_deg, 10)
        lon = round(pos.longitude_deg, 10)
        print(f"[DEBUG] Current GPS: lat={lat}, lon={lon}")
        return lat, lon

async def initialize_drone_and_takeoff():
    print("[DEBUG] Initializing drone system object")
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")
    print("[DEBUG] Sent connection command to drone; awaiting connection state")

    async for state in drone.core.connection_state():
        print(f"[DEBUG] Connection state: {state.is_connected}")
        if state.is_connected:
            print("-- Connected to drone")
            break

    print("[DEBUG] Waiting for global position estimate (health)...")
    async for health in drone.telemetry.health():
        print(f"[DEBUG] Health status: GPS_OK={health.is_global_position_ok}, HOME_OK={health.is_home_position_ok}")
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position OK")
            break

    print("-- Arming")
    await drone.action.arm()
    print("-- Armed")

    print(f"-- Taking off to {ALTITUDE} m AGL")
    await drone.action.set_takeoff_altitude(ALTITUDE)
    await drone.action.takeoff()
    print("[DEBUG] Takeoff command sent; polling altitude")

    async for position in drone.telemetry.position():
        rel_alt = position.relative_altitude_m
        abs_alt = position.absolute_altitude_m
        print(f"[DEBUG] Altitude telemetry: relative={rel_alt:.2f} m, absolute={abs_alt:.2f} m")
        if rel_alt >= ALTITUDE * 0.95:
            print("-- Reached target altitude")
            break
        await asyncio.sleep(0.5)

    async for hp in drone.telemetry.home():
        home_rel = hp.relative_altitude_m
        home_abs = hp.absolute_altitude_m
        print(f"[DEBUG] Home position: relative={home_rel:.2f} m, absolute={home_abs:.2f} m")
        break

    return drone

async def detect_aruco_marker(timeout=5.0):
    print(f"[DEBUG] detect_aruco_marker: searching for marker (timeout={timeout} s)")
    t0 = time.time()
    prev_gray = None
    frame_count = 0

    while (time.time() - t0) < timeout:
        frame = await asyncio.to_thread(picam2.capture_array)
        frame_count += 1
        print(f"[DEBUG] Frame #{frame_count}: captured from camera")
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        print(f"[DEBUG] Converted frame #{frame_count} to grayscale")

        if prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100,
                                          qualityLevel=0.01, minDistance=20)
            num_pts = 0 if pts is None else len(pts)
            print(f"[DEBUG] Frame #{frame_count}: goodFeaturesToTrack found {num_pts} points")
            if pts is not None:
                curr, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                valid = st.reshape(-1) == 1
                valid_count = int(np.count_nonzero(valid))
                print(f"[DEBUG] Frame #{frame_count}: optical flow tracked {valid_count} points")
                if valid_count >= 6:
                    M, inliers = cv2.estimateAffinePartial2D(
                        pts[valid], curr[valid]
                    )
                    if M is not None:
                        print(f"[DEBUG] Frame #{frame_count}: computed affine matrix:\n{M}")
                        frame = cv2.warpAffine(frame, M, frame.shape[1::-1])
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        print(f"[DEBUG] Frame #{frame_count}: warped and re-grayscale conversion")

        prev_gray = gray
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=DETECT_PARAMS)
        num_markers = 0 if ids is None else len(ids)
        print(f"[DEBUG] Frame #{frame_count}: detected {num_markers} markers")
        if ids is not None and TARGET_ID in ids.flatten():
            print(f"[DEBUG] Frame #{frame_count}: Marker ID {TARGET_ID} detected")
            idx = list(ids.flatten()).index(TARGET_ID)
            print(f"[DEBUG] Frame #{frame_count}: using marker index {idx} in corners array")
            _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS
            )
            offset = tvecs[0][0]  # [x, y, z] in meters
            print(f"[DEBUG] Frame #{frame_count}: Pose offset = x={offset[0]:.3f}, y={offset[1]:.3f}, z={offset[2]:.3f}")
            return offset

        print(f"[DEBUG] Frame #{frame_count}: Marker not found yet; time elapsed = {(time.time() - t0):.2f} s")
        await asyncio.sleep(0.05)

    print("[DEBUG] Marker search timed out without detection")
    return None

async def approach_and_land(drone, initial_offset):
    print("[DEBUG] approach_and_land: starting with initial_offset =", initial_offset)

    # Get initial NED and yaw
    print("[DEBUG] Fetching initial NED and yaw")
    async for od in drone.telemetry.position_velocity_ned():
        north0, east0, down0 = od.position.north_m, od.position.east_m, od.position.down_m
        print(f"[DEBUG] Initial NED: north={north0:.3f}, east={east0:.3f}, down={down0:.3f}")
        break

    async for att in drone.telemetry.attitude_euler():
        yaw = att.yaw_deg
        print(f"[DEBUG] Initial yaw: {yaw:.2f}°")
        break

    # Send single offboard setpoint before start
    print("[DEBUG] Sending single offboard setpoint before start")
    await drone.offboard.set_position_ned(PositionNedYaw(north0, east0, down0, yaw))
    try:
        await drone.offboard.start()
        print("[DEBUG] Offboard started")
    except OffboardError as e:
        print(f"[ERROR] Offboard start failed: {e}")
        return False

    # Compute first target N/E based on initial offset
    target_n = north0 + initial_offset[1]
    target_e = east0 + initial_offset[0]
    print(f"[DEBUG] Computed first target: target_n={target_n:.3f}, target_e={target_e:.3f}")

    iteration = 0
    while True:
        iteration += 1
        print(f"[DEBUG] Iteration #{iteration}: sending offboard setpoint to N={target_n:.3f}, E={target_e:.3f}, D={down0:.3f}, Yaw={yaw:.2f}")
        await drone.offboard.set_position_ned(PositionNedYaw(target_n, target_e, down0, yaw))
        await asyncio.sleep(0.5)

        print(f"[DEBUG] Iteration #{iteration}: attempting to re-detect marker")
        new_offset = await detect_aruco_marker(timeout=2.0)
        if new_offset is None:
            print(f"[DEBUG] Iteration #{iteration}: Marker lost—retrying detection")
            continue

        dx, dy = new_offset[0], new_offset[1]
        dist = math.hypot(dy, dx)
        print(f"[DEBUG] Iteration #{iteration}: Detected new_offset: dx={dx:.3f}, dy={dy:.3f}, distance={dist:.3f}")

        if dist <= TOLERANCE:
            print(f"[DEBUG] Iteration #{iteration}: Within tolerance ({dist:.3f} m ≤ {TOLERANCE} m); centering complete")
            break

        target_n += dy
        target_e += dx
        print(f"[DEBUG] Iteration #{iteration}: Updated intermediate target to N={target_n:.3f}, E={target_e:.3f}")

    # Now centered: send GPS 100 times
    latitude, longitude = await fetch_current_gps_coordinates(drone)
    coord_bytes = f"{latitude},{longitude}\n".encode("utf-8")
    print(f"[DEBUG] Centered GPS ready to send: {latitude}, {longitude}")
    for i in range(100):
        # ser.write(coord_bytes)
        print(f"[DEBUG] GPS broadcast #{i+1}: {latitude},{longitude}")
    print("[DEBUG] Finished broadcasting GPS 100 times")

    print("[DEBUG] Commanding Return to Launch")
    await drone.action.return_to_launch()

    try:
        await drone.offboard.stop()
        print("[DEBUG] Offboard stopped")
    except OffboardError as e:
        print(f"[ERROR] Offboard stop error: {e}")

    return True

async def execute_mission():
    drone = None
    try:
        print("[DEBUG] execute_mission: Starting")
        drone = await initialize_drone_and_takeoff()

        for idx, (lat, lon) in enumerate(coordinates, start=1):
            print(f"[DEBUG] Waypoint #{idx}: Heading to ({lat}, {lon}) at {ALTITUDE} m AGL")
            async for hp in drone.telemetry.home():
                home_abs = hp.absolute_altitude_m
                print(f"[DEBUG] Home absolute altitude for AMSL computation: {home_abs:.2f} m")
                break
            target_amsl = home_abs + ALTITUDE
            print(f"[DEBUG] Waypoint #{idx}: goto_location AMSL={target_amsl:.2f} m")
            await drone.action.goto_location(lat, lon, target_amsl, 0.0)

            print("[DEBUG] Sleeping 7s to allow movement toward waypoint")
            await asyncio.sleep(7)

            print(f"[DEBUG] Waypoint #{idx}: Searching and centering")
            initial_offset = await detect_aruco_marker(timeout=5.0)
            if initial_offset is not None:
                print(f"[DEBUG] Waypoint #{idx}: Initial offset found: {initial_offset}")
                success = await approach_and_land(drone, initial_offset)
                if success:
                    print(f"[DEBUG] Waypoint #{idx}: approach_and_land succeeded; mission terminating")
                    return
                else:
                    print(f"[DEBUG] Waypoint #{idx}: approach_and_land reported failure; continuing to next waypoint")
            else:
                print(f"[DEBUG] Waypoint #{idx}: No marker detected at this waypoint")

        print("[DEBUG] No marker found at any waypoint; commanding Return to Launch")
        await drone.action.return_to_launch()

    except Exception as e:
        print(f"[ERROR] Exception in execute_mission: {e}")
    finally:
        if drone:
            print("[DEBUG] Cleanup: stopping offboard if running and landing")
            try:
                await drone.offboard.stop()
                print("[DEBUG] Offboard stopped in cleanup")
            except Exception as e:
                print(f"[ERROR] Offboard stop error in cleanup: {e}")
            await drone.action.land()
            print("[DEBUG] Land command sent")
        print("[DEBUG] Mission complete or aborted")

if __name__ == '__main__':
    print("[DEBUG] Script start")
    asyncio.run(execute_mission())
    print("[DEBUG] Script exit")

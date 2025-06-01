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
TARGET_ID = 1

# ----------------------------
# Flight Parameters
# ----------------------------
ALTITUDE = 5       # takeoff and waypoint altitude in meters (AGL)
AMSL_ALTITUDE = ALTITUDE + 5
TOLERANCE = 0.10   # 10 cm tolerance diagonally (meters)

# ----------------------------
# Waypoints and Geofence
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
cam_cfg = picam2.create_preview_configuration(
    raw={"size": (1640, 1232)},
    main={"format": "RGB888", "size": (write_width, write_height)}
)
picam2.configure(cam_cfg)
picam2.start()
print("[DEBUG] Camera started: RGB888 preview at {}x{}".format(write_width, write_height))
time.sleep(2)  # allow auto-exposure to stabilize

# ----------------------------
# Open LoRa communication
# ----------------------------
ser = serial.Serial(port='/dev/ttyUSB0', baudrate=57600)

async def get_gps_coordinates_from_drone(drone):
    async for pos in drone.telemetry.position():
        return round(pos.latitude_deg, 10), round(pos.longitude_deg, 10)

async def connect_and_arm():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

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
    await drone.action.set_takeoff_altitude(ALTITUDE)
    await drone.action.takeoff()

    await asyncio.sleep(10)
    return drone

async def search_marker(timeout=5.0):
    print(f"[DEBUG] Searching for marker (timeout: {timeout} seconds)")
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
                valid_count = np.count_nonzero(st.reshape(-1) == 1)
                if valid_count >= 6:
                    M, _ = cv2.estimateAffinePartial2D(
                        pts[st.reshape(-1) == 1],
                        curr[st.reshape(-1) == 1]
                    )
                    if M is not None:
                        frame = cv2.warpAffine(frame, M, frame.shape[1::-1])
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        prev_gray = gray
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT,
                                                 parameters=DETECT_PARAMS)
        if ids is not None and TARGET_ID in ids.flatten():
            print(f"[DEBUG] Marker ID {TARGET_ID} detected")
            idx = list(ids.flatten()).index(TARGET_ID)
            _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS
            )
            offset = tvecs[0][0]  # [x, y, z] in meters
            print(f"[DEBUG] Pose offset: {offset}")
            return offset

        print("[DEBUG] Marker not found in current frame")
    print("[DEBUG] Marker search timed out")
    return None

async def center_and_land(drone):
    # Try centering multiple times until within tolerance
    max_attempts = 5
    for attempt in range(max_attempts):
        print(f"[DEBUG] Centering attempt {attempt+1}/{max_attempts}")
        offset = await search_marker(5.0)
        if offset is None:
            print("[DEBUG] No offset returned; abort centering")
            return False

        dx = offset[0]  # East offset (m)
        dy = offset[1]  # North offset (m)
        dist = math.hypot(dx, dy)
        print(f"[DEBUG] Offset distance: {dist:.3f} m")

        if dist <= TOLERANCE:
            print("[DEBUG] Within tolerance; marker centered")
            break

        # Get current NED position
        async for od in drone.telemetry.position_velocity_ned():
            north0 = od.position.north_m
            east0 = od.position.east_m
            down0 = od.position.down_m
            break

        async for att in drone.telemetry.attitude_euler():
            yaw = att.yaw_deg
            break

        # Command movement to reduce offset
        target_n = north0 + dy
        target_e = east0 + dx
        print(f"[DEBUG] Moving to N: {target_n:.2f}, E: {target_e:.2f} to improve centering")
        await drone.offboard.set_position_ned(
            PositionNedYaw(target_n, target_e, down0, yaw)
        )
        await asyncio.sleep(2)  # give time to move

        if attempt == 0:
            try:
                print("[DEBUG] Enabling offboard mode for centering")
                await drone.offboard.start()
            except OffboardError:
                print("[ERROR] Offboard start failed during centering")
                return False

    else:
        print("[DEBUG] Failed to center within tolerance after max attempts")
        return False

    # Once centered: send GPS coordinates 200 times with 1 ms delay
    latitude, longitude = await get_gps_coordinates_from_drone(drone)
    coord_bytes = f"{latitude},{longitude}\n".encode('utf-8')
    print(f"[DEBUG] Centered GPS: {latitude}, {longitude}")
    for i in range(200):
        ser.write(coord_bytes)
        await asyncio.sleep(0.001)  # 1 ms

    print("[DEBUG] GPS location sent 200 times")

    # Fly to landing point 5 meters west of first waypoint
    print("[DEBUG] Heading to landing point")
    await drone.action.goto_location(LAND_LAT, LAND_LON, AMSL_ALTITUDE, 0.0)
    await asyncio.sleep(10)  # allow time to reach

    print("[DEBUG] Initiating landing at landing point")
    await drone.action.land()
    return True

async def run():
    drone = None
    try:
        drone = await connect_and_arm()
        # Enable offboard before any centering movements
        await drone.offboard.set_position_ned(PositionNedYaw(0, 0, 0, 0))

        for lat, lon in coordinates:
            print(f"[DEBUG] Heading to waypoint ({lat}, {lon}) at {ALTITUDE} m AGL")
            await drone.action.goto_location(lat, lon, AMSL_ALTITUDE, 0.0)
            await asyncio.sleep(7)

            print(f"[DEBUG] Searching and centering at ({lat}, {lon})")
            success = await center_and_land(drone)
            if success:
                return

        print("[DEBUG] No marker found at any waypoint; returning to launch")
        await drone.action.return_to_launch()

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


if __name__ == '__main__':
    print("[DEBUG] Script start")
    asyncio.run(run())
    print("[DEBUG] Script exit")
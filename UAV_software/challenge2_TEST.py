import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw
from mavsdk.geofence import Point, Polygon, FenceType, GeofenceData
import math
import serial
import threading

# ----------------------------
# Camera Globals
# ----------------------------
picam2 = Picamera2()
write_width, write_height = 640, 480

# ----------------------------
# Calibration and Distortion
# ----------------------------
INTRINSIC = np.array([
    [653.1070007239106, 0.0,          339.2952147845755],
    [0.0,               650.7753992788821, 258.1165494889447],
    [0.0,               0.0,          1.0]
], dtype=np.float32)
DIST_COEFFS = np.array([
    -0.03887864427953473,
     0.6888798469690414,
     0.00815702400928161,
     0.010438854120041072,
    -1.713270699000528
], dtype=np.float32)

# Geofence polygon
geofence_points = [
    Point(34.4044560, -119.6955893),
    Point(34.4042018, -119.6954699),
    Point(34.4043747, -119.6958679),
    Point(34.4041124, -119.6957519)
]

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
print("[DEBUG] Defining flight parameters")
ALTITUDE = 2       # takeoff and waypoint altitude (AGL, m)
TOLERANCE = 0.10   # 10 cm centering tolerance when approaching marker (m)
VELOCITY_MS = 0.4  # m/s horizontal speed during sweep
=======
ALTITUDE = 1.5       # takeoff and waypoint altitude (AGL, m)
TOLERANCE = 0.10     # 10 cm centering tolerance (m)
VELOCITY_MS = 0.6    # m/s
SERIAL_PORT = '/dev/ttyUSB0'
BAUDRATE = 57600
ser = serial.Serial(port=SERIAL_PORT, baudrate=BAUDRATE)

# ----------------------------
# Initialize Camera
# ----------------------------
cam_cfg = picam2.create_preview_configuration(
    raw={"size": (1640, 1232)},
    main={"format": "RGB888", "size": (write_width, write_height)}
)
picam2.configure(cam_cfg)
picam2.start()
time.sleep(2)

# ----------------------------
# Video Preview Thread
# ----------------------------
video_preview_running = True
def video_preview():
    while video_preview_running:
        frame = picam2.capture_array()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=DETECT_PARAMS)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(bgr, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_SIZE, INTRINSIC, DIST_COEFFS
            )
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.aruco.drawAxis(bgr, INTRINSIC, DIST_COEFFS, rvec, tvec, MARKER_SIZE * 0.5)
        cv2.imshow("Video Preview", bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

threading.Thread(target=video_preview, daemon=True).start()

# ----------------------------
# Telemetry & Control Helpers
# ----------------------------
async def fetch_current_gps_coordinates(drone):
    async for pos in drone.telemetry.position():
        return round(pos.latitude_deg, 10), round(pos.longitude_deg, 10)

async def initialize_drone_and_takeoff():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            break

    print("-- Uploading geofence")
    polygon = Polygon(geofence_points, FenceType.INCLUSION)
    geofence_data = GeofenceData(polygons=[polygon], circles=[]) 
    await drone.geofence.upload_geofence(geofence_data)

    print("[DEBUG] initialize_drone_and_takeoff: commanding arm")
    await drone.action.arm()
    await drone.action.set_takeoff_altitude(ALTITUDE)
    await drone.action.takeoff()
    await asyncio.sleep(5)
    return drone

async def detect_aruco_marker(timeout=0.05):
    t0 = time.time()
    frame = await asyncio.to_thread(picam2.capture_array)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=DETECT_PARAMS)
    if ids is not None and TARGET_ID in ids.flatten():
        idx = list(ids.flatten()).index(TARGET_ID)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS
        )
        return tvecs[0][0]
    return None

async def move_and_detect(drone, vx, vy, duration, yaw):
    t0 = time.time()
    while time.time() - t0 < duration:
        await drone.offboard.set_velocity_ned(VelocityNedYaw(vx, vy, 0.0, yaw))
        offset = await detect_aruco_marker()
        if offset is not None:
            return offset
        await asyncio.sleep(0.05)
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, yaw))
    return None

async def approach_and_land(drone, initial_offset):
    async for od in drone.telemetry.position_velocity_ned():
        north0, east0, down0 = od.position.north_m, od.position.east_m, od.position.down_m
        break
    async for att in drone.telemetry.attitude_euler():
        yaw = att.yaw_deg
        break
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,yaw))
    try:
        await drone.offboard.start()
    except OffboardError:
        return False
    target_n = north0 + initial_offset[1]
    target_e = east0 + initial_offset[0]
    while True:
        async for od in drone.telemetry.position_velocity_ned():
            cur_n, cur_e = od.position.north_m, od.position.east_m
            break
        dx, dy = target_n - cur_n, target_e - cur_e
        dist = math.hypot(dx, dy)
        if dist <= TOLERANCE:
            await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,yaw))
            await asyncio.sleep(1.0)
            break
        vx = (dx / dist) * VELOCITY_MS
        vy = (dy / dist) * VELOCITY_MS
        await drone.offboard.set_velocity_ned(VelocityNedYaw(vx, vy, 0.0, yaw))
        new_offset = await detect_aruco_marker()
        if new_offset is not None:
            target_n = cur_n + new_offset[1]
            target_e = cur_e + new_offset[0]
        await asyncio.sleep(0.1)
    lat, lon = await fetch_current_gps_coordinates(drone)
    coord_bytes = f"{lat},{lon}\n".encode("utf-8")
    for _ in range(100):
        ser.write(coord_bytes)
        await asyncio.sleep(0.05)
    await drone.offboard.set_velocity_ned(VelocityNedYaw(-1,0,0,0))
    await asyncio.sleep(10)
    try:
        await drone.offboard.stop()
    except OffboardError:
        pass
    await drone.action.land()
    return True

def gps_to_ned_meters(lat_ref, lon_ref, lat, lon):
    dlat = lat - lat_ref
    dlon = lon - lon_ref
    meters_per_deg_lat = 111139.0
    meters_per_deg_lon = 111139.0 * math.cos(math.radians(lat_ref))
    return dlat * meters_per_deg_lat, dlon * meters_per_deg_lon

# ----------------------------
# Main Mission Logic
# ----------------------------
async def execute_mission():
    drone = None
    try:
        drone = await initialize_drone_and_takeoff()

<<<<<<< HEAD
        # --- fly to start point ---
        # storke field
        #lat_start, lon_start = 34.4192290, -119.8549169
        
        
        
        # --------------------------- START FIELD COORDINATES ----------
        # East Field
        lat_start, lon_start = 34.4043866, -119.6955729
        # West Field
        # lat_start, lon_start = 34.4042154, -119.6962494
        # --------------------------- START FIELD COORDINATES ----------
        
        
=======
        # goto start
        lat_start, lon_start = 34.4192290, -119.8549169
>>>>>>> f13c7a960900f8b5e5b256ced1ee425d6d36b1e2
        async for hp in drone.telemetry.home():
            home_abs = hp.absolute_altitude_m
            break
        target_amsl = home_abs + ALTITUDE
        await drone.action.goto_location(lat_start, lon_start, target_amsl, 0.0)
        await asyncio.sleep(7)

        # NED origin & yaw
        async for od in drone.telemetry.position_velocity_ned():
            north0, east0, down0 = od.position.north_m, od.position.east_m, od.position.down_m
            break
        async for att in drone.telemetry.attitude_euler():
            yaw = att.yaw_deg
            break

        # start offboard
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,yaw))
        try:
            await drone.offboard.start()
        except OffboardError:
            await drone.action.return_to_launch()
            return

<<<<<<< HEAD
        # --- snake-pattern timing parameters ---
        yard_to_m = 0.9144
        dist1 = 23 * yard_to_m        # long leg in meters
        dist2 = 5 * yard_to_m     # shift in meters
=======
        # snake-pattern timing
        dist1, dist2 = 5, 0.5
>>>>>>> f13c7a960900f8b5e5b256ced1ee425d6d36b1e2
        time_long = dist1 / VELOCITY_MS
        time_shift = dist2 / VELOCITY_MS
        angle1 = math.radians(30)
        angle2 = math.radians(120)
        vx_out  = -VELOCITY_MS * math.cos(angle1)
        vy_out  =  VELOCITY_MS * math.sin(angle1)
        vx_back =  VELOCITY_MS * math.cos(angle1)
        vy_back = -VELOCITY_MS * math.sin(angle1)
        vx_lat  =  VELOCITY_MS * math.cos(angle2)
        vy_lat  = -VELOCITY_MS * math.sin(angle2)

        num_lengths = 10
        for i in range(num_lengths):
            # long leg with continuous detection
            vx, vy = (vx_out, vy_out) if i % 2 == 0 else (vx_back, vy_back)
            offset = await move_and_detect(drone, vx, vy, time_long, yaw)
            if offset is not None:
                await approach_and_land(drone, offset)
                return

            # shift leg
            if i < num_lengths - 1:
                offset = await move_and_detect(drone, vx_lat, vy_lat, time_shift, yaw)
                if offset is not None:
                    await approach_and_land(drone, offset)
                    return

        # complete sweep
        await drone.offboard.stop()
        await drone.action.land()

    except Exception as e:
        if drone:
            try: await drone.offboard.stop()
            except: pass
            await drone.action.land()
    finally:
        global video_preview_running
        video_preview_running = False

if __name__ == '__main__':
    asyncio.run(execute_mission())

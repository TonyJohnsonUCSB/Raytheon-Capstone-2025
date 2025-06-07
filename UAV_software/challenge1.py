import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw
import math
import serial
import threading

# ----------------------------
# Camera Globals
# ----------------------------
print("[DEBUG] Initializing camera")
picam2 = Picamera2()
write_width, write_height = 640, 480

# ----------------------------
# Calibration and Distortion
# ----------------------------
print("[DEBUG] Setting up camera calibration")
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

# ----------------------------
# ArUco Detection Parameters
# ----------------------------
print("[DEBUG] Configuring ArUco detector")
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
DETECT_PARAMS = cv2.aruco.DetectorParameters_create()
DETECT_PARAMS.adaptiveThreshConstant = 7
DETECT_PARAMS.minMarkerPerimeterRate = 0.03
MARKER_SIZE = 0.06611  # meters
TARGET_ID = 3

# ----------------------------
# Flight Parameters
# ----------------------------
print("[DEBUG] Setting flight parameters")
ALTITUDE = 4
TOLERANCE = 0.10
VELOCITY_MS = 1.5
SERIAL_PORT = '/dev/ttyUSB0'
BAUDRATE = 57600
print(f"[DEBUG] Opening serial port {SERIAL_PORT} @ {BAUDRATE}")
ser = serial.Serial(port=SERIAL_PORT, baudrate=BAUDRATE)

# ----------------------------
# Initialize Camera
# ----------------------------
print("[DEBUG] Configuring and starting camera")
cam_cfg = picam2.create_preview_configuration(
    raw={"size": (1640, 1232)},
    main={"format": "RGB888", "size": (write_width, write_height)}
)
picam2.configure(cam_cfg)
picam2.start()
time.sleep(2)
print("[DEBUG] Camera ready")

# ----------------------------
# Video Preview Thread
# ----------------------------
video_preview_running = True
def video_preview():
    print("[DEBUG] Starting video preview thread")
    while video_preview_running:
        frame = picam2.capture_array()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=DETECT_PARAMS)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(bgr, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, INTRINSIC, DIST_COEFFS)
            #for rvec, tvec in zip(rvecs, tvecs):
                #cv2.aruco.drawAxis(bgr, INTRINSIC, DIST_COEFFS, rvec, tvec, MARKER_SIZE * 0.5)
        cv2.imshow("Video Preview", bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    print("[DEBUG] Video preview thread exiting")

threading.Thread(target=video_preview, daemon=True).start()

# ----------------------------
# Telemetry & Control Helpers
# ----------------------------
async def fetch_current_gps_coordinates(drone):
    print("[DEBUG] Fetching current GPS coordinates")
    async for pos in drone.telemetry.position():
        lat, lon = round(pos.latitude_deg, 10), round(pos.longitude_deg, 10)
        print(f"[DEBUG] Current GPS: {lat}, {lon}")
        return lat, lon

async def initialize_drone_and_takeoff():
    print("[DEBUG] Initializing drone connection")
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")
    print("[DEBUG] Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[DEBUG] Drone connected")
            break
    print("[DEBUG] Waiting for health checks")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("[DEBUG] Health checks passed")
            break
    print("[DEBUG] Arming drone")
    await drone.action.arm()
    print(f"[DEBUG] Setting takeoff altitude to {ALTITUDE} m")
    await drone.action.set_takeoff_altitude(ALTITUDE)
    print("[DEBUG] Taking off")
    await drone.action.takeoff()
    await asyncio.sleep(5)
    print("[DEBUG] Takeoff complete")
    return drone

async def detect_aruco_marker(timeout=0.05):
    frame = await asyncio.to_thread(picam2.capture_array)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=DETECT_PARAMS)
    if ids is not None and TARGET_ID not in ids.flatten():
        print(f"WRONG DROP ZONE {ids}")
    if ids is not None and TARGET_ID in ids.flatten():
        idx = list(ids.flatten()).index(TARGET_ID)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers([corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS)
        offset = tvecs[0][0]
        print(f"[DEBUG] detect_aruco_marker: found ID {TARGET_ID}, offset={offset}")
        return offset
    return None

async def move_and_detect(drone, vx, vy, duration, yaw):
    print(f"[DEBUG] move_and_detect: start move vx={vx:.3f}, vy={vy:.3f}, duration={duration:.2f}s")
    t0 = time.time()
    while time.time() - t0 < duration:
        await drone.offboard.set_velocity_ned(VelocityNedYaw(vx, vy, 0.0, yaw))
        offset = await detect_aruco_marker()
        if offset is not None:
            print("[DEBUG] move_and_detect: marker detected during move")
            return offset
        await asyncio.sleep(0.05)
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, yaw))
    print("[DEBUG] move_and_detect: move complete, no marker found")
    return None

async def approach_and_land(drone, initial_offset):
    print("[DEBUG] approach_and_land: starting approach")
    async for od in drone.telemetry.position_velocity_ned():
        north0, east0, down0 = od.position.north_m, od.position.east_m, od.position.down_m
        print(f"[DEBUG] approach_and_land: start NED=({north0:.2f},{east0:.2f},{down0:.2f})")
        break
    async for att in drone.telemetry.attitude_euler():
        yaw = att.yaw_deg
        print(f"[DEBUG] approach_and_land: start yaw={yaw:.2f}°")
        break
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,yaw))
    try:
        print("[DEBUG] approach_and_land: starting offboard")
        await drone.offboard.start()
    except OffboardError as e:
        print(f"[ERROR] approach_and_land: offboard start failed: {e}")
        return False
    target_n = north0 + initial_offset[1]
    target_e = east0 + initial_offset[0]
    print(f"[DEBUG] approach_and_land: target NED=({target_n:.2f},{target_e:.2f})")
    while True:
        async for od in drone.telemetry.position_velocity_ned():
            cur_n, cur_e = od.position.north_m, od.position.east_m
            break
        dx, dy = target_n - cur_n, target_e - cur_e
        dist = math.hypot(dx, dy)
        print(f"[DEBUG] approach_and_land: cur NED=({cur_n:.2f},{cur_e:.2f}), dist={dist:.3f}")
        if dist <= TOLERANCE:
            print("[DEBUG] approach_and_land: within tolerance, hovering")
            await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,yaw))
            await asyncio.sleep(1.0)
            break
        vx = (dx / dist) * VELOCITY_MS
        vy = (dy / dist) * VELOCITY_MS
        print(f"[DEBUG] approach_and_land: commanding vx={vx:.3f}, vy={vy:.3f}")
        await drone.offboard.set_velocity_ned(VelocityNedYaw(vx, vy, 0.0, yaw))
        new_offset = await detect_aruco_marker()
        if new_offset is not None:
            print(f"[DEBUG] approach_and_land: refine offset={new_offset}")
            target_n = cur_n + new_offset[1]
            target_e = cur_e + new_offset[0]
        await asyncio.sleep(0.1)
    lat, lon = await fetch_current_gps_coordinates(drone)
    coord_bytes = f"{lat},{lon}\n".encode("utf-8")
    print(f"[DEBUG] approach_and_land: sending final GPS {lat},{lon} over serial")
    # ~ for i in range(100):
        # ~ ser.write(coord_bytes)
        # ~ await asyncio.sleep(0.05)
    # ~ print("[DEBUG] approach_and_land: backing off")
    # ~ await drone.offboard.set_velocity_ned(VelocityNedYaw(1,0,0,0))
    # ~ await asyncio.sleep(10)
    try:
        print("[DEBUG] approach_and_land: stopping offboard")
        await drone.offboard.stop()
    except OffboardError as e:
        print(f"[ERROR] approach_and_land: stop offboard failed: {e}")
    print("[DEBUG] approach_and_land: landing")
    await drone.action.land()
    return True

def gps_to_ned_meters(lat_ref, lon_ref, lat, lon):
    print("[DEBUG] Converting GPS to NED meters")
    dlat = lat - lat_ref
    dlon = lon - lon_ref
    meters_per_deg_lat = 111139.0
    meters_per_deg_lon = 111139.0 * math.cos(math.radians(lat_ref))
    north = dlat * meters_per_deg_lat
    east = dlon * meters_per_deg_lon
    print(f"[DEBUG] gps_to_ned_meters: north={north:.2f}, east={east:.2f}")
    return north, east

# ----------------------------
# Main Mission Logic
# ----------------------------
async def execute_mission():
    print("[DEBUG] execute_mission: start")
    drone = None
    try:
        drone = await initialize_drone_and_takeoff()

        print("[DEBUG] Going to start point")
        #lat_start, lon_start = 34.4043866, -119.6955729
        lat_start, lon_start = 34.4043988, -119.6955675
        async for hp in drone.telemetry.home():
            home_abs = hp.absolute_altitude_m
            break
        target_amsl = home_abs + ALTITUDE
        print(f"[DEBUG] Goto ({lat_start},{lon_start},{target_amsl:.2f})")
        await drone.action.goto_location(lat_start, lon_start, target_amsl, 0.0)
        await asyncio.sleep(15)

        async for od in drone.telemetry.position_velocity_ned():
            north0, east0, down0 = od.position.north_m, od.position.east_m, od.position.down_m
            print(f"[DEBUG] NED origin=({north0:.2f},{east0:.2f},{down0:.2f})")
            break
        async for att in drone.telemetry.attitude_euler():
            yaw = att.yaw_deg
            print(f"[DEBUG] Start yaw={yaw:.2f}°")
            break

        print("[DEBUG] Starting offboard mode")
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,yaw))
        try:
            await drone.offboard.start()
            print("[DEBUG] Offboard started")
        except OffboardError as e:
            print(f"[ERROR] Offboard start failed: {e}")
            await drone.action.return_to_launch()
            return

        # snake-pattern parameters
        ytm = 0.9144
        dist1 = ytm * 23
        dist2 = ytm * 3
        time_long = dist1 / VELOCITY_MS
        time_shift = dist2 / VELOCITY_MS
        angle1 = math.radians(22)
        angle2 = math.radians(112)
        vx_out, vy_out = -VELOCITY_MS*math.cos(angle1), VELOCITY_MS*math.sin(angle1)
        vx_back, vy_back = VELOCITY_MS*math.cos(angle1), -VELOCITY_MS*math.sin(angle1)
        vx_lat, vy_lat = VELOCITY_MS*math.cos(angle2), -VELOCITY_MS*math.sin(angle2)

        num_lengths = 10
        for i in range(num_lengths):
            leg = "out" if i % 2 == 0 else "back"
            vx, vy = (vx_out, vy_out) if i % 2 == 0 else (vx_back, vy_back)
            print(f"[DEBUG] Leg {i+1} ({leg}): moving for {time_long:.2f}s")
            offset = await move_and_detect(drone, vx, vy, time_long, yaw)
            if offset is not None:
                print(f"[DEBUG] Marker found on leg {i+1}")
                await approach_and_land(drone, offset)
                return

            if i < num_lengths - 1:
                print(f"[DEBUG] Leg {i+1} shift: moving for {time_shift:.2f}s")
                offset = await move_and_detect(drone, vx_lat, vy_lat, time_shift, yaw)
                if offset is not None:
                    print(f"[DEBUG] Marker found during shift on leg {i+1}")
                    await approach_and_land(drone, offset)
                    return

        print("[DEBUG] Snake sweep complete, landing")
        await drone.offboard.stop()
        await drone.action.land()

    except Exception as e:
        print(f"[ERROR] execute_mission: exception {e}")
        if drone:
            try: await drone.offboard.stop()
            except: pass
            await drone.action.land()
    finally:
        global video_preview_running
        video_preview_running = False
        print("[DEBUG] Mission script exiting")

if __name__ == '__main__':
    asyncio.run(execute_mission())

Here’s the complete script with the three fixes applied (renamed detector call, VELOCITY_MS instead of VELOCITY, and fetching/packing GPS before serial send):

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
print("[DEBUG] Initializing camera globals")
picam2 = Picamera2()
write_width, write_height = 640, 480

# ----------------------------
# Calibration and Distortion
# ----------------------------
print("[DEBUG] Setting up intrinsic and distortion coefficients")
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
print("[DEBUG] Configuring ArUco detection parameters")
ARUCO_DICT    = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
DETECT_PARAMS = cv2.aruco.DetectorParameters_create()
DETECT_PARAMS.adaptiveThreshConstant   = 7
DETECT_PARAMS.minMarkerPerimeterRate   = 0.03
MARKER_SIZE   = 0.06611  # meters
TARGET_ID     = 2

# ----------------------------
# Flight Parameters
# ----------------------------
print("[DEBUG] Defining flight parameters")
ALTITUDE    = 4       # AGL, m
TOLERANCE   = 0.10    # m
VELOCITY_MS = 0.6     # m/s
SERIAL_PORT = '/dev/ttyUSB0'
BAUDRATE    = 57600
print(f"[DEBUG] Opening serial port {SERIAL_PORT} at baud {BAUDRATE}")
ser = serial.Serial(port=SERIAL_PORT, baudrate=BAUDRATE)

# ----------------------------
# Initialize Camera
# ----------------------------
print("[DEBUG] Configuring camera preview and main streams")
cam_cfg = picam2.create_preview_configuration(
    raw = {"size": (1640, 1232)},
    main= {"format": "RGB888", "size": (write_width, write_height)}
)
picam2.configure(cam_cfg)
picam2.start()
print(f"[DEBUG] Camera started: RGB888 preview at {write_width}x{write_height}")
time.sleep(2)
print("[DEBUG] Camera auto-exposure should be stable now")

# ----------------------------
# Video Preview Thread
# ----------------------------
video_preview_running = True
def video_preview():
    while video_preview_running:
        frame   = picam2.capture_array()
        bgr     = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

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

preview_thread = threading.Thread(target=video_preview, daemon=True)
preview_thread.start()
print("[DEBUG] Video preview thread started")

# ----------------------------
# Helpers
# ----------------------------
async def fetch_current_gps_coordinates(drone):
    print("[DEBUG] fetch_current_gps_coordinates: awaiting one GPS fix")
    async for pos in drone.telemetry.position():
        lat = round(pos.latitude_deg, 10)
        lon = round(pos.longitude_deg, 10)
        print(f"[DEBUG] GPS: lat={lat}, lon={lon}")
        return lat, lon

async def initialize_drone_and_takeoff():
    print("[DEBUG] Connecting to drone")
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[DEBUG] Drone connected")
            break

    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("[DEBUG] GPS & home OK")
            break

    print("[DEBUG] Arming and taking off")
    await drone.action.arm()
    await drone.action.set_takeoff_altitude(ALTITUDE)
    await drone.action.takeoff()
    await asyncio.sleep(5)
    print("[DEBUG] Takeoff complete")
    return drone

async def detect_aruco_marker(timeout=2.0):
    print(f"[DEBUG] detect_aruco_marker: timeout={timeout}s")
    t0, prev_gray = time.time(), None

    while time.time() - t0 < timeout:
        frame = await asyncio.to_thread(picam2.capture_array)
        gray  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100,
                                          qualityLevel=0.01, minDistance=20)
            if pts is not None:
                curr, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                valid = (st.reshape(-1) == 1)
                if np.count_nonzero(valid) >= 6:
                    M = cv2.estimateAffinePartial2D(pts[valid], curr[valid])[0]
                    if M is not None:
                        frame = cv2.warpAffine(frame, M, frame.shape[1::-1])
                        gray  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        prev_gray = gray
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=DETECT_PARAMS)
        if ids is not None and TARGET_ID in ids.flatten():
            idx = list(ids.flatten()).index(TARGET_ID)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS
            )
            offset = tvecs[0][0]
            print(f"[DEBUG] Marker offset: {offset}")
            return offset

        await asyncio.sleep(0.05)

    print("[DEBUG] No marker within timeout")
    return None

async def approach_and_land(drone):
    print("[DEBUG] Starting approach_and_land")
    # get yaw
    async for att in drone.telemetry.attitude_euler():
        yaw = att.yaw_deg
        print(f"[DEBUG] Initial yaw: {yaw:.1f}")
        break

    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, yaw))
    try:
        await drone.offboard.start()
        print("[DEBUG] Offboard started")
    except OffboardError as e:
        print(f"[ERROR] Offboard start failed: {e}")
        return False

    while True:
        async for od in drone.telemetry.position_velocity_ned():
            north, east, down = od.position.north_m, od.position.east_m, od.position.down_m
            print(f"[DEBUG] NED: N={north:.2f}, E={east:.2f}, D={down:.2f}")
            break

        print("[DEBUG] Detecting marker")
        offset = await detect_aruco_marker(timeout=2.0)
        if offset is None:
            print("[DEBUG] No offset; retrying")
            continue

        dx_e, dx_n = offset[0], offset[1]
        dist = math.hypot(dx_n, dx_e)
        print(f"[DEBUG] Distance to marker: {dist:.3f} m")

        if dist < TOLERANCE:
            lat, lon    = await fetch_current_gps_coordinates(drone)
            coord_str   = f"{lat:.10f},{lon:.10f}\n"
            coord_bytes = coord_str.encode()
            print(f"[DEBUG] Within tolerance; sending GPS {coord_str.strip()}")
            for i in range(100):
                ser.write(coord_bytes)
                await asyncio.sleep(0.05)

            # back up then land
            await drone.offboard.set_velocity_ned(VelocityNedYaw(-1, 0, 0, 0))
            await asyncio.sleep(10)
            try:
                await drone.offboard.stop()
                print("[DEBUG] Offboard stopped")
            except OffboardError as e:
                print(f"[ERROR] Offboard stop failed: {e}")
            await drone.action.land()
            print("[DEBUG] Land commanded")
            return True

        vn = -(dx_n / dist) * VELOCITY_MS * 0.5
        ve =  (dx_e / dist) * VELOCITY_MS * 0.5
        print(f"[DEBUG] Commanding vel N={vn:.2f}, E={ve:.2f}")
        await drone.offboard.set_velocity_ned(VelocityNedYaw(vn, ve, 0, yaw))
        await asyncio.sleep(1.0)

def gps_to_ned_meters(lat_ref, lon_ref, lat, lon):
    dlat = lat - lat_ref
    dlon = lon - lon_ref
    meters_per_deg_lat = 111139.0
    meters_per_deg_lon = 111139.0 * math.cos(math.radians(lat_ref))
    north = dlat * meters_per_deg_lat
    east  = dlon * meters_per_deg_lon
    return north, east

# ----------------------------
# Main Mission
# ----------------------------
async def execute_mission():
    print("[DEBUG] execute_mission: start")
    drone = None
    try:
        drone = await initialize_drone_and_takeoff()

        # fly to start point
        lat_start, lon_start = 34.4192290, -119.8549169
        async for hp in drone.telemetry.home():
            home_abs = hp.absolute_altitude_m
            break
        target_amsl = home_abs + ALTITUDE
        print(f"[DEBUG] Goto ({lat_start}, {lon_start}, {target_amsl:.2f})")
        await drone.action.goto_location(lat_start, lon_start, target_amsl, 0.0)
        await asyncio.sleep(7)

        # record origin NED & yaw
        async for od in drone.telemetry.position_velocity_ned():
            north0, east0, _ = od.position.north_m, od.position.east_m, od.position.down_m
            break
        async for att in drone.telemetry.attitude_euler():
            yaw = att.yaw_deg
            break

        # start offboard
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, yaw))
        try:
            await drone.offboard.start()
        except OffboardError as e:
            print(f"[ERROR] Offboard start failed: {e}")
            await drone.action.return_to_launch()
            return

        # snake‐pattern params
        dist1 = 5.0
        dist2 = 0.5
        angle1 = math.radians(30)
        angle2 = math.radians(120)
        north_long  = -dist1 * math.cos(angle1)
        east_long   =  dist1 * math.sin(angle1)
        north_long_r= -north_long
        east_long_r = -east_long
        north_lat   = dist2 * math.cos(angle2)
        east_lat    = dist2 * math.sin(angle2)
        vx_out      = -VELOCITY_MS * math.cos(angle1)
        vy_out      =  VELOCITY_MS * math.sin(angle1)
        vx_back     =  VELOCITY_MS * math.cos(angle1)
        vy_back     = -VELOCITY_MS * math.sin(angle1)
        vx_lat      = VELOCITY_MS * math.cos(angle2)
        vy_lat      = VELOCITY_MS * math.sin(angle2)

        current_n, current_e = north0, east0
        num_lengths = 10

        for idx in range(num_lengths):
            if idx % 2 == 0:
                target_n, target_e = current_n + north_long, current_e + east_long
                vx, vy = vx_out, vy_out
                leg_type = "out"
            else:
                target_n, target_e = current_n + north_long_r, current_e + east_long_r
                vx, vy = vx_back, vy_back
                leg_type = "back"
            print(f"[DEBUG] Leg {idx+1} ({leg_type}) target = ({target_n:.2f}, {target_e:.2f})")

            # drive long leg
            while True:
                async for od in drone.telemetry.position_velocity_ned():
                    cur_n, cur_e = od.position.north_m, od.position.east_m
                    break
                dx, dy = target_n - cur_n, target_e - cur_e
                dist = math.hypot(dx, dy)
                if dist <= TOLERANCE:
                    await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,yaw))
                    await asyncio.sleep(1)
                    break
                await drone.offboard.set_velocity_ned(VelocityNedYaw(vx, vy, 0, yaw))
                offset = await detect_aruco_marker(timeout=0.05)
                if offset is not None:
                    print(f"[DEBUG] Found on leg {idx+1}")
                    try: await drone.offboard.stop()
                    except: pass
                    await approach_and_land(drone)
                    return
                await asyncio.sleep(0.1)

            current_n, current_e = target_n, target_e

            # lateral shift
            if idx < num_lengths - 1:
                shift_n = current_n + north_lat
                shift_e = current_e + east_lat
                print(f"[DEBUG] Shift {idx+1} target = ({shift_n:.2f}, {shift_e:.2f})")
                while True:
                    async for od in drone.telemetry.position_velocity_ned():
                        cur_n, cur_e = od.position.north_m, od.position.east_m
                        break
                    dx, dy = shift_n - cur_n, shift_e - cur_e
                    dist = math.hypot(dx, dy)
                    if dist <= TOLERANCE:
                        await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,yaw))
                        await asyncio.sleep(1)
                        break
                    await drone.offboard.set_velocity_ned(VelocityNedYaw(vx_lat, vy_lat, 0, yaw))
                    offset = await detect_aruco_marker(timeout=0.05)
                    if offset is not None:
                        print(f"[DEBUG] Found during shift {idx+1}")
                        try: await drone.offboard.stop()
                        except: pass
                        await approach_and_land(drone)
                        return
                    await asyncio.sleep(0.1)
                current_n, current_e = shift_n, shift_e

        # no marker found → land
        try: await drone.offboard.stop()
        except: pass
        await drone.action.land()
        print("[DEBUG] Sweep complete, landing")

    except Exception as e:
        print(f"[ERROR] execute_mission exception: {e}")
        if drone:
            try: await drone.offboard.stop()
            except: pass
            await drone.action.land()
    finally:
        if drone:
            try: await drone.offboard.stop()
            except: pass
            await drone.action.land()
        print("[DEBUG] Cleanup done")

if __name__ == '__main__':
    print("[DEBUG] Launching mission")
    asyncio.run(execute_mission())
    video_preview_running = False
    preview_thread.join()
    print("[DEBUG] Mission exited")

Here’s the full mission script, with the execute_mission() refactored to fly each leg by time at fixed velocity instead of distance-checking loops. All debug prints remain intact.

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
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
DETECT_PARAMS = cv2.aruco.DetectorParameters_create()
DETECT_PARAMS.adaptiveThreshConstant = 7
DETECT_PARAMS.minMarkerPerimeterRate = 0.03
MARKER_SIZE = 0.06611  # meters
TARGET_ID = 1

# ----------------------------
# Flight Parameters
# ----------------------------
print("[DEBUG] Defining flight parameters")
ALTITUDE = 4       # takeoff and waypoint altitude (AGL, m)
TOLERANCE = 0.10   # 10 cm centering tolerance when approaching marker (m)
VELOCITY_MS = 0.6  # m/s horizontal speed during sweep
SERIAL_PORT = '/dev/ttyUSB0'
BAUDRATE = 57600
print(f"[DEBUG] Opening serial port {SERIAL_PORT} at baud {BAUDRATE}")
ser = serial.Serial(port=SERIAL_PORT, baudrate=BAUDRATE)

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
time.sleep(2)  # allow auto-exposure to stabilize
print("[DEBUG] Camera auto-exposure should be stable now")

# ----------------------------
# Video Preview Thread with Pose Overlay
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

preview_thread = threading.Thread(target=video_preview, daemon=True)
preview_thread.start()
print("[DEBUG] Video preview thread started")

# ----------------------------
# Telemetry & Control Helpers
# ----------------------------
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
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS
            )
            offset = tvecs[0][0]
            print(f"[DEBUG] detect_aruco_marker: marker offset = {offset}")
            return offset

        await asyncio.sleep(0.05)

    print("[DEBUG] detect_aruco_marker: timeout reached, no marker found")
    return None

async def approach_and_land(drone, initial_offset):
    print("[DEBUG] approach_and_land: initiated")
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

    target_n = north0 + initial_offset[1]
    target_e = east0 + initial_offset[0]
    print(f"[DEBUG] approach_and_land: initial marker-target NED = ({target_n:.2f}, {target_e:.2f}, {down0:.2f})")

    while True:
        async for od in drone.telemetry.position_velocity_ned():
            cur_n, cur_e = od.position.north_m, od.position.east_m
            break
        dx = target_n - cur_n
        dy = target_e - cur_e
        dist = math.hypot(dx, dy)
        print(f"[DEBUG] approach_and_land: current NED = ({cur_n:.2f}, {cur_e:.2f}), dx={dx:.3f}, dy={dy:.3f}, dist={dist:.3f}")
        if dist <= TOLERANCE:
            print(f"[DEBUG] approach_and_land: within tolerance ({dist:.3f} ≤ {TOLERANCE}), holding position")
            await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, yaw))
            await asyncio.sleep(1.0)
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

    print("[DEBUG] approach_and_land: fetching final GPS coordinates")
    latitude, longitude = await fetch_current_gps_coordinates(drone)
    coord_bytes = f"{latitude},{longitude}\n".encode("utf-8")
    print(f"[DEBUG] approach_and_land: final GPS = ({latitude}, {longitude}), sending over serial")

    for i in range(100):
        print(f"[DEBUG] approach_and_land: serial write {i+1}/100: {coord_bytes.decode().strip()}")
        ser.write(coord_bytes)
        await asyncio.sleep(0.05)

    await drone.offboard.set_velocity_ned(VelocityNedYaw(-1,0,0,0))
    await asyncio.sleep(10)

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

# ----------------------------
# Main Mission Logic with Time-based Snake-Pattern Sweep
# ----------------------------
async def execute_mission():
    print("[DEBUG] execute_mission: starting mission sequence")
    drone = None
    try:
        # --- initialize and take off ---
        drone = await initialize_drone_and_takeoff()

        # --- fly to start point ---
        lat_start, lon_start = 34.4192290, -119.8549169
        async for hp in drone.telemetry.home():
            home_abs = hp.absolute_altitude_m
            print(f"[DEBUG] execute_mission: home AMSL altitude = {home_abs:.2f} m")
            break
        target_amsl = home_abs + ALTITUDE
        print(f"[DEBUG] execute_mission: goto_location → ({lat_start}, {lon_start}, {target_amsl:.2f})")
        await drone.action.goto_location(lat_start, lon_start, target_amsl, 0.0)
        await asyncio.sleep(7)

        # --- get NED origin & yaw ---
        async for od in drone.telemetry.position_velocity_ned():
            north0, east0, down0 = od.position.north_m, od.position.east_m, od.position.down_m
            print(f"[DEBUG] execute_mission: NED origin = ({north0:.2f}, {east0:.2f}, {down0:.2f})")
            break
        async for att in drone.telemetry.attitude_euler():
            yaw = att.yaw_deg
            print(f"[DEBUG] execute_mission: yaw at start = {yaw:.2f}°")
            break

        # --- start offboard ---
        print("[DEBUG] execute_mission: setting initial offboard velocity to zero")
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, yaw))
        try:
            await drone.offboard.start()
            print("[DEBUG] execute_mission: offboard started")
        except OffboardError as e:
            print(f"[ERROR] execute_mission: offboard start failed: {e}")
            await drone.action.return_to_launch()
            return

        # --- snake-pattern timing parameters ---
        yard_to_m = 0.9144
        dist1 = 5        # long leg in meters
        dist2 = 0.5      # shift in meters
        time_long = dist1 / VELOCITY_MS
        time_shift = dist2 / VELOCITY_MS

        angle1 = math.radians(30)
        angle2 = math.radians(120)

        vx_out  = -VELOCITY_MS * math.cos(angle1)
        vy_out  =  VELOCITY_MS * math.sin(angle1)
        vx_back =  VELOCITY_MS * math.cos(angle1)
        vy_back = -VELOCITY_MS * math.sin(angle1)
        vx_lat  =  VELOCITY_MS * math.cos(angle2)
        vy_lat  =  VELOCITY_MS * math.sin(angle2)

        num_lengths = 10
        for i in range(num_lengths):
            # long leg
            if i % 2 == 0:
                vx, vy, leg = vx_out, vy_out, "out"
            else:
                vx, vy, leg = vx_back, vy_back, "back"
            print(f"[DEBUG] Leg {i+1} ({leg}): commanding vx={vx:.3f}, vy={vy:.3f} for {time_long:.2f}s")
            await drone.offboard.set_velocity_ned(VelocityNedYaw(vx, vy, 0, yaw))
            await asyncio.sleep(time_long)
            print(f"[DEBUG] Leg {i+1}: stopping")
            await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, yaw))
            await asyncio.sleep(1)

            offset = await detect_aruco_marker(timeout=0.1)
            if offset is not None:
                print(f"[DEBUG] Marker found on Leg {i+1}, offset={offset}")
                await drone.offboard.stop()
                await approach_and_land(drone, offset)
                return

            # lateral shift
            if i < num_lengths - 1:
                print(f"[DEBUG] Leg {i+1} shift: commanding vx={vx_lat:.3f}, vy={vy_lat:.3f} for {time_shift:.2f}s")
                await drone.offboard.set_velocity_ned(VelocityNedYaw(vx_lat, vy_lat, 0, yaw))
                await asyncio.sleep(time_shift)
                print(f"[DEBUG] Leg {i+1} shift: stopping")
                await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, yaw))
                await asyncio.sleep(1)

                offset = await detect_aruco_marker(timeout=0.1)
                if offset is not None:
                    print(f"[DEBUG] Marker found during shift on Leg {i+1}, offset={offset}")
                    await drone.offboard.stop()
                    await approach_and_land(drone, offset)
                    return

        # completed all legs
        print("[DEBUG] execute_mission: completed snake sweep")
        await drone.offboard.stop()
        print("[DEBUG] execute_mission: offboard stopped after sweep")
        await drone.action.land()

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
                print("[DEBUG] execute_mission: offboard stopped in cleanup")
            except:
                pass
            await drone.action.land()
            print("[DEBUG] execute_mission: cleanup complete")

if __name__ == '__main__':
    print("[DEBUG] Running execute_mission via asyncio")
    asyncio.run(execute_mission())
    video_preview_running = False
    preview_thread.join()
    print("[DEBUG] Mission script has exited and video preview closed")

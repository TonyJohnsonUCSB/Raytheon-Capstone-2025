import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw
import math
import serial

print("[DEBUG] Initializing camera globals")
picam2 = Picamera2()
write_width, write_height = 640, 480

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

print("[DEBUG] Configuring ArUco detection parameters")
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
DETECT_PARAMS = cv2.aruco.DetectorParameters_create()
DETECT_PARAMS.adaptiveThreshConstant = 7
DETECT_PARAMS.minMarkerPerimeterRate = 0.03
MARKER_SIZE = 0.06611
TARGET_ID = 2

print("[DEBUG] Defining flight parameters")
ALTITUDE = 4
TOLERANCE = 0.10
VELOCITY_MS = 0.2
SERIAL_PORT = '/dev/ttyUSB0'
BAUDRATE = 57600
print(f"[DEBUG] Opening serial port {SERIAL_PORT} at baud {BAUDRATE}")
ser = serial.Serial(port=SERIAL_PORT, baudrate=BAUDRATE)

print("[DEBUG] Setting first GPS waypoint")
FIRST_WP_LAT = 34.41870255
FIRST_WP_LON = -119.85509000

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

latest_frame = None

async def show_video_feed():
    global latest_frame
    print("[DEBUG] Starting video feed display loop")
    while True:
        if latest_frame is not None:
            cv2.imshow("Drone Camera Feed", latest_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[DEBUG] 'q' pressed, exiting video feed loop")
                break
        await asyncio.sleep(0.03)
    cv2.destroyAllWindows()

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
    global latest_frame
    print(f"[DEBUG] detect_aruco_marker: start detection with timeout={timeout}s")
    t0 = time.time()
    prev_gray = None

    while (time.time() - t0) < timeout:
        frame = await asyncio.to_thread(picam2.capture_array)
        latest_frame = frame.copy()
        print("[DEBUG] detect_aruco_marker: captured frame")
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        print("[DEBUG] detect_aruco_marker: converted to grayscale")

        if prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=20)
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
                        latest_frame = frame.copy()
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
            offset = tvecs[0][0]
            print(f"[DEBUG] detect_aruco_marker: marker offset = {offset}")
            return offset
        await asyncio.sleep(0.05)

    print("[DEBUG] detect_aruco_marker: timeout reached, no marker found")
    return None

# ... (rest of the script remains unchanged, only cut for brevity) ...

async def execute_mission():
    print("[DEBUG] execute_mission: starting mission sequence")
    asyncio.create_task(show_video_feed())
    drone = None
    try:
        drone = await initialize_drone_and_takeoff()
        # (rest of mission logic unchanged) ...
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

import asyncio
import time
import math
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw

#-----------TASK-------------#
# Using a test configuration, place an the  ArUCo marker arbitrarily designated 
# as "DropZone" 15ft perpendicular to the camera lens. Validate that the ID Subsystem 
# can properly identify as "DropZone". Repeat for each Non-DropZone marker in the 
# region to demonstrate selecting only the designated Drop Zone.

#-----INSTRUCTIONS-----------#
# This script should be used for Qual Tests 1 & 2
# You will have to land the drone manually
# Run the script: Drone ascends to 15ft
# Place TARGET_ID Marker (1) under drone and await "Drop Zone" validation
# Place all other markers under drone and await "Non-Drop Zone" validation 

# ----------------------------
# Flight Parameters
# ----------------------------
ALTITUDE = 4.6       # 4.6 m (15 ft) (AGL) height
AMSL_ALTITUDE = ALTITUDE + 5

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
# Init Camera
# ----------------------------
cam_cfg = picam2.create_preview_configuration(
    raw={'size': (1640, 1232)},
    main={'format': 'RGB888', 'size': (write_width, write_height)}
)
picam2.configure(cam_cfg)
picam2.start()
print('-- Camera started')
time.sleep(2)
print('-- Camera exposure stabilized')

async def connect_and_arm():
    print('-- Connecting to drone...')
    drone = System()
    await drone.connect(system_address='serial:///dev/ttyAMA0:57600')

    print('-- Waiting for connection...')
    async for state in drone.core.connection_state():
        if state.is_connected:
            print('-- Connected')
            break

    print('-- Waiting for global position...')
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print('-- Global position OK')
            break

    print('-- Arming')
    await drone.action.arm()

    print(f'-- Setting takeoff altitude to {ALTITUDE}m')
    await drone.action.set_takeoff_altitude(float(ALTITUDE))

    print('-- Taking off')
    await drone.action.takeoff()
    await asyncio.sleep(10)
    print('-- Takeoff complete')

    return drone

async def search_marker(timeout=10.0):
    print('Searching for marker')
    t0 = time.time()
    prev_gray = None
    frame_cnt = 0

    while time.time() - t0 < timeout:
        frame = await asyncio.to_thread(picam2.capture_array)
        frame_cnt += 1
        print(f'-- Frame {frame_cnt} captured')
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100,
                                          qualityLevel=0.01, minDistance=20)
            if pts is not None:
                curr, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                valid = int(np.count_nonzero(st.reshape(-1) == 1))
                if valid >= 6:
                    M, _ = cv2.estimateAffinePartial2D(
                        pts[st.reshape(-1)==1], curr[st.reshape(-1)==1]
                    )
                    if M is not None:
                        frame = cv2.warpAffine(frame, M, frame.shape[1::-1])
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        print('-- Frame stabilized')

        prev_gray = gray

        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=DETECT_PARAMS)
        if ids is not None:
            flat_ids = ids.flatten()
            print(f'-- Detected marker ID(s): {flat_ids}')
            if TARGET_ID in flat_ids:
                print('DropZone marker detected')
                return 
            else:
                print('Non-DropZone marker detected')

    print('-- search_marker timed out]')
    return None

async def run():
    drone = await connect_and_arm()
    await search_marker(timeout=10.0)

if __name__ == '__main__':
    asyncio.run(run())
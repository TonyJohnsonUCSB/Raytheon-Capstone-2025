import os
# suppress libpng incorrect sRGB profile warnings
os.environ['OPENCV_IO_ENABLE_PNG_WARNINGS'] = '0'

import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw

# --- PID Controller Parameters for Vibration-Resistant Tracking ---
Kp_x, Ki_x, Kd_x = 1.0, 0.0, 0.4  # increased derivative
Kp_y, Ki_y, Kd_y = 1.0, 0.0, 0.4
prev_error_x = prev_error_y = 0.0
integral_x = integral_y = 0.0
last_time = time.time()

# velocity limits & slow-zone
MAX_VEL = 0.5       # m/s
SLOW_DIST = 0.10    # m (10 cm)

# --- Frame stabilization state ---
prev_gray = None

# --- ArUco & Camera Calibration setup ---
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()
parameters.adaptiveThreshConstant = 7
parameters.minMarkerPerimeterRate = 0.03

camera_matrix = np.array([[933.15867, 0, 657.59],
                          [0, 933.1586, 400.36993],
                          [0, 0, 1]])
dist_coeffs   = np.array([-0.43948, 0.18514, 0, 0])
marker_size   = 0.06611  # meters
drop_zone_id  = 1        # ArUco ID to track

# --- Video Recording Setup ---
VIDEO_NAME = '/home/rtxcapstone/Desktop/testVideo.avi'
FRAME_SIZE = (640, 480)
CAPTURE_FPS = 30.0
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(VIDEO_NAME, fourcc, CAPTURE_FPS, FRAME_SIZE)
if not out.isOpened():
    raise RuntimeError(f"Cannot open VideoWriter at {VIDEO_NAME}")

# --- Picamera2 setup ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": 'RGB888', "size": FRAME_SIZE}
)
picam2.configure(config)

dur = int(1e6 / CAPTURE_FPS)
picam2.set_controls({"FrameDurationLimits": (dur, dur)})

print("-- Camera starting...")
picam2.start()
time.sleep(2)
print("-- Camera started")

# --- PID functions ---
def pid_east(error_x: float, dt: float) -> float:
    global prev_error_x, integral_x
    integral_x += error_x * dt
    derivative = (error_x - prev_error_x) / dt if dt > 0 else 0.0
    outp = Kp_x * error_x + Ki_x * integral_x + Kd_x * derivative
    prev_error_x = error_x
    return -outp  # x>0 -> move west (negative east)

def pid_north(error_y: float, dt: float) -> float:
    global prev_error_y, integral_y
    integral_y += error_y * dt
    derivative = (error_y - prev_error_y) / dt if dt > 0 else 0.0
    outp = Kp_y * error_y + Ki_y * integral_y + Kd_y * derivative
    prev_error_y = error_y
    return -outp  # y>0 -> move south (negative north)

# --- Drone connection and takeoff ---
async def connect_and_arm() -> System:
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            break
    await drone.action.arm()
    await drone.action.set_takeoff_altitude(6)
    await drone.action.takeoff()
    await asyncio.sleep(6)
    return drone

# --- Main offboard tracking loop ---
async def offboard_loop(drone: System):
    global prev_gray, last_time, integral_x, integral_y, prev_error_x, prev_error_y
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
    try:
        await drone.offboard.start()
    except OffboardError:
        return

    while True:
        now = time.time()
        dt = now - last_time if last_time else 0.01
        last_time = now

        frame = await asyncio.to_thread(picam2.capture_array)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # stabilization
        if prev_gray is None:
            stabilized = frame.copy()
        else:
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200,
                                               qualityLevel=0.01, minDistance=30)
            M = None
            if prev_pts is not None:
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
                valid = status.reshape(-1)==1
                if np.count_nonzero(valid) >= 6:
                    src = prev_pts[valid]
                    dst = curr_pts[valid]
                    M, _ = cv2.estimateAffinePartial2D(src, dst)
            if M is not None:
                h, w = frame.shape[:2]
                stabilized = cv2.warpAffine(frame, M, (w, h))
            else:
                stabilized = frame.copy()
        prev_gray = gray

        # ArUco detection
        gray_stab = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray_stab, ARUCO_DICT, parameters=parameters)

        vel_east = vel_north = 0.0
        x_cam = y_cam = z_cam = None

        if ids is not None:
            for idx, mid in enumerate(ids.flatten()):
                if int(mid) != drop_zone_id:
                    continue
                _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[idx]], marker_size, camera_matrix, dist_coeffs)
                x_cam, y_cam, z_cam = tvecs[0][0]

                # slow-down zone
                if abs(x_cam) < SLOW_DIST and abs(y_cam) < SLOW_DIST:
                    vel_east = vel_north = 0.0
                else:
                    ve = pid_east(x_cam, dt)
                    vn = pid_north(y_cam, dt)
                    # clamp
                    vel_east  = np.clip(ve, -MAX_VEL, MAX_VEL)
                    vel_north = np.clip(vn, -MAX_VEL, MAX_VEL)
                break
        else:
            integral_x = integral_y = 0.0
            prev_error_x = prev_error_y = 0.0

        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(vel_north, vel_east, 0.0, 0.0)
        )

        # overlay telemetry
        text = f"x={x_cam:.3f} y={y_cam:.3f} z={z_cam:.3f}" if x_cam is not None else "No marker detected"
        cv2.putText(stabilized, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(stabilized, f"vx={vel_east:.2f} vy={vel_north:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        out.write(stabilized)
        cv2.imshow("Preview", stabilized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        await asyncio.sleep(0.01)

    try:
        await drone.offboard.stop()
    except OffboardError:
        pass
    await drone.action.land()
    await asyncio.sleep(5)
    await drone.action.disarm()
    cv2.destroyAllWindows()

async def main():
    drone = await connect_and_arm()
    loop_task = asyncio.create_task(offboard_loop(drone))
    try:
        await loop_task
    except KeyboardInterrupt:
        loop_task.cancel()
        picam2.stop()
        picam2.close()
        out.release()
        cv2.destroyAllWindows()
        await loop_task

if __name__ == "__main__":
    asyncio.run(main())

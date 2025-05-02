import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

# --- Vibration Mitigation Parameters ---
ALPHA = 0.5                       # pose smoothing factor
prev_tvecs = {}                  # store previous tvecs per marker
prev_gray = None                 # for frame-to-frame stabilization

# --- ArUco & camera calibration setup ---
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()
# Detector tweaks for robustness
parameters.adaptiveThreshConstant = 7
parameters.minMarkerPerimeterRate = 0.03

camera_matrix = np.array([[933.15867, 0, 657.59],
                          [0, 933.1586, 400.36993],
                          [0, 0, 1]])
dist_coeffs = np.array([-0.43948, 0.18514, 0, 0])

marker_size = 0.06611
drop_zone_id = 1

# Helper functions
def compute_vel_east(pos):
    if abs(pos) < 0.01:
        return 0.0
    elif abs(pos) < 0.3:
        return -np.sign(pos)
    else:
        return -pos

def compute_vel_north(pos):
    if abs(pos) < 0.01:
        return 0.0
    elif abs(pos) < 0.3:
        return np.sign(pos)
    else:
        return pos

def smooth_tvec(marker_id, new_tvec):
    global prev_tvecs
    if marker_id in prev_tvecs:
        tvec = ALPHA * new_tvec + (1 - ALPHA) * prev_tvecs[marker_id]
    else:
        tvec = new_tvec
    prev_tvecs[marker_id] = tvec
    return tvec

# Initialize PiCamera2 + window
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    raw={"size": (1640, 1232)},
    main={"format": 'RGB888', "size": (640, 480)}
)
picam2.configure(config)
# manual exposure controls for less motion blur
time.sleep(0.1)
picam2.start()
picam2.set_controls({
    "ExposureTime": 20000,    # 20ms shutter
    "AnalogueGain": 4.0       # boost ISO
})
time.sleep(2)

# Video writer
frame_width, frame_height = 640, 480
fps = 20.0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('output2.avi', fourcc, fps, (frame_width, frame_height))

cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Preview", frame_width, frame_height)

async def connect_and_arm():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    print("Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected")
            break

    print("Waiting for GPS and home position...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position OK")
            break

    print("-- Arming")
    await drone.action.arm()
    print("-- Taking off")
    await drone.action.set_takeoff_altitude(6)
    await drone.action.takeoff()
    await asyncio.sleep(6)
    return drone

async def offboard_loop(drone):
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"Offboard start failed: {e._result.result}")
        return

    print("-- Entering tracking loop --")
    global prev_gray
    try:
        while True:
            frame = await asyncio.to_thread(picam2.capture_array)
            # Frame stabilization
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                stabilized = frame.copy()
            else:
                prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200,
                                                   qualityLevel=0.01, minDistance=30)
                M = None
                if prev_pts is not None:
                    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
                    idx = status.reshape(-1) == 1
                    if np.count_nonzero(idx) >= 6:
                        src = prev_pts[idx]
                        dst = curr_pts[idx]
                        M, _ = cv2.estimateAffinePartial2D(src, dst)
                if M is not None:
                    h, w = frame.shape[:2]
                    stabilized = cv2.warpAffine(frame, M, (w, h))
                else:
                    stabilized = frame.copy()
            prev_gray = gray

            # ArUco detection on stabilized frame
            gray_stab = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray_stab, ARUCO_DICT, parameters=parameters)

            vel_east = vel_north = 0.0
            x_cam = y_cam = z_cam = None

            if ids is not None:
                for idx, mid in enumerate(ids.flatten()):
                    if int(mid) != drop_zone_id:
                        continue
                    cv2.aruco.drawDetectedMarkers(stabilized, [corners[idx]])
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[idx]], marker_size, camera_matrix, dist_coeffs)
                    new_tvec = tvecs[0][0]
                    x_cam, y_cam, z_cam = smooth_tvec(int(mid), new_tvec)
                    vel_east = compute_vel_east(x_cam)
                    vel_north = compute_vel_north(y_cam)
                    break

            # Overlay text
            font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            if x_cam is not None:
                cv2.putText(stabilized, f"x={x_cam:.3f}m y={y_cam:.3f}m z={z_cam:.3f}m",
                            (10, 30), font, fs, (0,255,0), th)
                cv2.putText(stabilized, f"vx={vel_east:.3f}m/s vy={vel_north:.3f}m/s",
                            (10, 60), font, fs, (255,0,0), th)
            else:
                cv2.putText(stabilized, "No marker detected",
                            (10, 60), font, fs, (0,0,255), th)

            # Record & command
            video_writer.write(stabilized)
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(vel_north, vel_east, 0.0, 0.0)
            )

            cv2.imshow("Preview", stabilized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(0.05)

    except asyncio.CancelledError:
        pass
    finally:
        print("-- Stopping offboard, landing")
        try:
            await drone.offboard.stop()
        except OffboardError as e:
            print(f"Offboard stop failed: {e._result.result}")
        await drone.action.land()
        await asyncio.sleep(5)
        await drone.action.disarm()
        video_writer.release()
        cv2.destroyAllWindows()

async def main():
    drone = await connect_and_arm()
    task = asyncio.create_task(offboard_loop(drone))
    try:
        await task
    except KeyboardInterrupt:
        task.cancel()
        await task

if __name__ == "__main__":
    asyncio.run(main())

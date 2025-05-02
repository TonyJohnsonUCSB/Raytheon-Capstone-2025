import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

# --- ArUco & camera calibration setup ---
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

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

# Initialize PiCamera2 + window
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    raw={"size": (1640, 1232)},
    main={"format": 'RGB888', "size": (640, 480)}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

# --- set up video writer
frame_width, frame_height = 640, 480
fps = 20.0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('output2.avi', fourcc, fps, (frame_width, frame_height))
# --------------------------------

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
    try:
        while True:
            frame = await asyncio.to_thread(picam2.capture_array)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=parameters)

            vel_east = vel_north = 0.0
            x_cam = y_cam = z_cam = None

            if ids is not None:
                ids = ids.flatten()
                for idx, mid in enumerate(ids):
                    if mid != drop_zone_id:
                        continue
                    cv2.aruco.drawDetectedMarkers(frame, [corners[idx]])
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[idx]], marker_size, camera_matrix, dist_coeffs
                    )
                    x_cam, y_cam, z_cam = tvecs[0][0]
                    vel_east  = compute_vel_east(x_cam)
                    vel_north = compute_vel_north(y_cam)
                    break

            font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            if x_cam is not None:
                texts = [
                    f"x={x_cam:.3f} m, y={y_cam:.3f} m, z={z_cam:.3f} m",
                    f"vel_east={vel_east:.3f} m/s, vel_north={vel_north:.3f} m/s"
                ]
                for i, txt in enumerate(texts):
                    cv2.putText(frame, txt, (10, 30 + i*30), font, fs,
                                (0,255,0) if i==0 else (255,0,0), th)
                print(f"Pose: x={x_cam:.3f}, y={y_cam:.3f}, z={z_cam:.3f} | "
                      f"Setpoints → east: {vel_east:.3f}, north: {vel_north:.3f}")
            else:
                cv2.putText(frame, "No marker detected", (10,60), font, fs, (0,0,255), th)
                print("No marker detected. Setpoints all zero.")

            # record this frame
            video_writer.write(frame)

            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(vel_north, vel_east, 0.0, 0.0)
            )

            cv2.imshow("Preview", frame)
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
        # release the video writer
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

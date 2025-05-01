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

# Helper functions (from your first script)
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

# Initialize PiCamera2 (no preview window)
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    raw={"size": (1640, 1232)},
    main={"format": 'RGB888', "size": (640, 480)}
)
picam2.configure(config)
picam2.start()
# allow camera to warm up
time.sleep(2)

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
    await drone.action.set_takeoff_altitude(3.0)  # Set altitude to 3 meters
    await drone.action.takeoff()
    await asyncio.sleep(6)
    return drone

async def offboard_loop(drone):
    # initialize offboard with zero velocities
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"Offboard start failed: {e._result.result}")
        return

    print("-- Entering tracking loop --")
    try:
        while True:
            # capture frame on separate thread
            frame = await asyncio.to_thread(picam2.capture_array)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=parameters)

            # default setpoints
            vel_east = 0.0
            vel_north = 0.0
            x_cam = y_cam = z_cam = None

            if ids is not None:
                ids = ids.flatten()
                for idx, mid in enumerate(ids):
                    if mid != drop_zone_id:
                        continue

                    # pose estimation
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[idx]], marker_size, camera_matrix, dist_coeffs
                    )
                    tvec = tvecs[0][0]
                    x_cam, y_cam, z_cam = tvec

                    # compute velocities
                    vel_east = compute_vel_east(x_cam)
                    vel_north = compute_vel_north(y_cam)
                    break

            # logging
            if x_cam is not None:
                print(f"Pose: x={x_cam:.3f}, y={y_cam:.3f}, z={z_cam:.3f} | "
                      f"Setpoints → east: {vel_east:.3f}, north: {vel_north:.3f}")
            else:
                print("No marker detected. Setpoints all zero.")

            # send offboard commands (body-x = north, body-y = east)
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(vel_north, vel_east, 0.0, 0.0)
            )

            # brief delay
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

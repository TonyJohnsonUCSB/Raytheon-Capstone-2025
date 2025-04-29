import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

# --- ArUco & camera calibration setup ---
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)  # ArUco dictionary
parameters = cv2.aruco.DetectorParameters_create()  # Detector params

# Camera intrinsic parameters and distortion coefficients
camera_matrix = np.array([[933.15867, 0, 657.59],
                          [0, 933.1586, 400.36993],
                          [0, 0, 1]])
dist_coeffs = np.array([-0.43948, 0.18514, 0, 0])

# Physical marker size (meters) and target ID
marker_size = 0.06611
drop_zone_id = 1

# Control gains for centering the marker
k_x = 1.0    # Gain for horizontal (left/right) control
k_y = 1.0    # Gain for vertical (up/down) control
max_vel = 1.0  # Maximum velocity command (m/s)

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    raw={"size": (1640, 1232)},
    main={"format": 'RGB888', "size": (640, 480)}
)
picam2.configure(config)
picam2.start()
# Allow camera sensor to warm up
time.sleep(2)

async def connect_and_arm():
    """
    Connects to the drone, waits for health checks, arms, and takes off to ~2 m.
    Returns the System instance.
    """
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    # Wait until connected
    print("Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected")
            break

    # Wait for GPS & home position
    print("Waiting for GPS and home position...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position OK")
            break

    # Arm and take off
    print("-- Arming")
    await drone.action.arm()
    print("-- Taking off")
    await drone.action.takeoff()
    # Give time to reach ~2 m altitude
    await asyncio.sleep(6)

    return drone

async def offboard_loop(drone):
    """
    Starts offboard mode and continuously sends velocity commands to center the ArUco marker.
    Exits on cancel, then lands and disarms.
    """
    # Initialize offboard with zero velocities
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"Offboard start failed: {e._result.result}")
        return

    print("-- Entering tracking loop --")
    try:
        while True:
            # Capture a frame (off main thread)
            frame = await asyncio.to_thread(picam2.capture_array)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect markers
            corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=parameters)

            # Default no movement
            vx = vy = vz = 0.0
            if ids is not None:
                ids = ids.flatten()
                for idx, marker_id in enumerate(ids):
                    if marker_id != drop_zone_id:
                        continue
                    # Estimate pose of the detected marker
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[idx]], marker_size, camera_matrix, dist_coeffs
                    )
                    tvec = tvecs[0][0]
                    x_cam, y_cam, z_cam = tvec

                    # Compute velocity setpoints in drone body frame:
                    # Positive x_cam → marker to right → move right (vy positive)
                    vy = np.clip(k_x * x_cam, -max_vel, max_vel)
                    # Positive y_cam → marker above center → move up (vz negative is up)
                    vz = np.clip(-k_y * y_cam, -max_vel, max_vel)
                    break

            # Send velocity command (forward/backward vx = 0, yaw rate = 0)
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, vy, vz, 0.0)
            )
            await asyncio.sleep(0.1)

    except asyncio.CancelledError:
        # Loop cancelled (e.g., on user interrupt)
        pass
    finally:
        # Clean up: stop offboard, land, disarm
        print("-- Stopping offboard, landing")
        try:
            await drone.offboard.stop()
        except OffboardError as e:
            print(f"Offboard stop failed: {e._result.result}")
        await drone.action.land()
        await asyncio.sleep(5)
        await drone.action.disarm()

async def main():
    """
    Main entry: connect, arm, then run the offboard tracking loop until interrupted.
    """
    drone = await connect_and_arm()
    task = asyncio.create_task(offboard_loop(drone))

    try:
        await task
    except KeyboardInterrupt:
        # Cancel tracking loop on Ctrl-C
        task.cancel()
        await task

if __name__ == "__main__":
    asyncio.run(main())

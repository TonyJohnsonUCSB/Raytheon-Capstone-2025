#!/usr/bin/env python3

import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

# ======== Camera Calibration Parameters ========
# These are the intrinsic camera matrix and distortion coefficients
# from a calibration process specific to your PiCamera2 setup.
INTRINSIC = np.array([
    [653.1070007239106,   0.0,               339.2952147845755],
    [0.0,                 650.7753992788821, 258.1165494889447],
    [0.0,                 0.0,               1.0]
], dtype=np.float32)

DIST_COEFFS = np.array([
    -0.03887864427953473,
     0.6888798469690414,
     0.00815702400928161,
     0.010438854120041072,
    -1.713270699000528
], dtype=np.float32)

# ======== Detection & Control Parameters ========
ARUCO_DICT       = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)  # Use 6x6 ArUco markers
parameters       = cv2.aruco.DetectorParameters_create()                      # Detection parameters
MARKER_SIZE      = 0.06611           # Marker size in meters
DROP_ZONE_ID     = 1                 # ID of the ArUco marker to track
CENTER_TOLERANCE = 0.03              # Max offset in meters to consider centered
STEP_HEIGHT      = 1.0               # Meters to descend per step
FINAL_ALTITUDE   = -0.3              # Final downward NED altitude (0 is takeoff level)

# ======== Initialize Pi Camera ========
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    raw={"size": (1640, 1232)},
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(config)
picam2.start()
time.sleep(2)  # Allow time for camera to warm up and stabilize

# ======== Connect to Drone and Take Off ========
async def connect_and_arm():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")  # Connect to the drone via serial port

    #status_text_task = asyncio.ensure_future(print_status_text(drone))

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    print("-- Arming")
    await drone.action.arm()


    print("-- Taking off")
    await drone.action.set_takeoff_altitude(4)  # Set altitude to 5 meters
    await drone.action.takeoff()

    await asyncio.sleep(10)

    return drone

# ======== Center Drone Over Marker ========
async def center_over_marker(drone, curr_down, max_attempts=10):
    """Try to align drone directly over the target ArUco marker."""
    for attempt in range(max_attempts):
        # Capture frame from camera
        frame = await asyncio.to_thread(picam2.capture_array)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers in the frame
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=parameters)

        # Check if our target marker is detected
        if ids is not None and DROP_ZONE_ID in ids:
            idx = list(ids.flatten()).index(DROP_ZONE_ID)

            # Estimate marker pose to get relative position (tvecs)
            _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS
            )
            x, y, z = tvecs[0][0]  # Position of marker in camera frame

            # Get drone's current NED position
            async for odom in drone.telemetry.position_velocity_ned():
                curr_north = odom.position.north_m
                curr_east  = odom.position.east_m
                break

            # Translate camera offsets to NED frame movements
            target_north = curr_north + y  # y-axis in camera = forward = North
            target_east  = curr_east  + x  # x-axis in camera = right = East

            # Command offboard movement to new target position at current altitude
            await drone.offboard.set_position_ned(
                PositionNedYaw(target_north, target_east, curr_down, 0.0)
            )
            await asyncio.sleep(3)  # Give drone time to move

            # Re-check position to see how close we got
            async for od in drone.telemetry.position_velocity_ned():
                err_n = abs(od.position.north_m - target_north)
                err_e = abs(od.position.east_m  - target_east)
                break

            if err_n < CENTER_TOLERANCE and err_e < CENTER_TOLERANCE:
                return True  # Drone is centered well enough

        else:
            print("Marker not found.")

        await asyncio.sleep(1)  # Wait before retrying

    return False  # Gave up after all attempts

# ======== Descend Slowly and Land Precisely ========
async def descend_and_land(drone):
    await drone.telemetry.set_rate_position_velocity_ned(10)  # Update rate for NED telemetry

    # Get starting downward position
    async for pos in drone.telemetry.position_velocity_ned():
        curr_down = pos.position.down_m
        break

    # Hold current position as offboard start point
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, curr_down, 0.0))

    # Try to start offboard mode
    try:
        await drone.offboard.start()
    except OffboardError:
        print("Failed to start offboard.")
        await drone.action.disarm()
        return

    print("Beginning descent sequence.")
    while curr_down > FINAL_ALTITUDE:
        print(f"Centering at down = {curr_down:.2f} m")
        centered = await center_over_marker(drone, curr_down)  # Try to center

        if not centered:
            print("Failed to center. Aborting descent.")
            break

        # Move one meter down
        curr_down -= STEP_HEIGHT
        print(f"Descending to {curr_down:.2f} m")
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, curr_down, 0.0))
        await asyncio.sleep(5)  # Allow time to descend

    # Final landing sequence
    print("Landing now.")
    await drone.offboard.stop()
    await drone.action.land()
    await asyncio.sleep(5)
    await drone.action.disarm()

# ======== Main Entry Point ========
async def main():
    drone = await connect_and_arm()  # Connect and takeoff
    await center_over_marker(drone, 4)
    await descend_and_land(drone)    # Begin slow precision descent

if __name__ == "__main__":
    asyncio.run(main())

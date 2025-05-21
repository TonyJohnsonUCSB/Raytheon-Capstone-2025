#!/usr/bin/env python3

import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

# ----------------------------
# Camera setup
# ----------------------------
print("[DEBUG] Initializing camera")
camera = Picamera2()
image_width, image_height = 640, 480
camera_configuration = camera.create_preview_configuration(
    raw={"size": (1640, 1232)},
    main={"format": "RGB888", "size": (image_width, image_height)}
)
camera.configure(camera_configuration)
camera.start()
print(f"[DEBUG] Camera started with resolution {image_width}x{image_height}")
time.sleep(2)  # allow auto-exposure to stabilize

# ----------------------------
# Calibration parameters
# ----------------------------
print("[DEBUG] Loading intrinsic and distortion parameters")
intrinsic_matrix = np.array([
    [653.1070007239106, 0.0,               339.2952147845755],
    [0.0,               650.7753992788821, 258.1165494889447],
    [0.0,               0.0,               1.0]
], dtype=np.float32)

distortion_coefficients = np.array([
    -0.03887864427953473,
     0.6888798469690414,
     0.00815702400928161,
     0.010438854120041072,
    -1.713270699000528
], dtype=np.float32)

# ----------------------------
# ArUco marker detection
# ----------------------------
print("[DEBUG] Setting up ArUco detector")
aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
marker_detector_parameters = cv2.aruco.DetectorParameters_create()
marker_detector_parameters.adaptiveThreshConstant = 7
marker_detector_parameters.minMarkerPerimeterRate = 0.03
marker_physical_size_m = 0.06611  # meters
target_marker_id = 1

# ----------------------------
# Flight parameters
# ----------------------------
print("[DEBUG] Defining flight parameters")
takeoff_altitude_m = 5
absolute_altitude_msl = takeoff_altitude_m + 9
position_tolerance_m = 0.01

# ----------------------------
# Number of halving passes before landing
# ----------------------------
number_of_halving_steps = 2  # positive integer

# ----------------------------
# GPS waypoints
# ----------------------------
waypoint_coordinates = [
    (34.4189, -119.85533),
    (34.4189, -119.85530),
    (34.4189, -119.85528),
    (34.4189, -119.85526),
    (34.4189, -119.85524),
    (34.4189, -119.85522),
    (34.4189, -119.85520),
]

async def connect_and_arm_drone():
    print("[DEBUG] Connecting to drone via serial")
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    print("[DEBUG] Waiting for connection state")
    async for connection_state in drone.core.connection_state():
        if connection_state.is_connected:
            print("[DEBUG] Drone connected")
            break

    print("[DEBUG] Waiting for GPS fix and home position")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("[DEBUG] Global position and home position OK")
            break

    print("[DEBUG] Arming drone")
    await drone.action.arm()
    print(f"[DEBUG] Setting takeoff altitude to {takeoff_altitude_m} m")
    await drone.action.set_takeoff_altitude(takeoff_altitude_m)
    print("[DEBUG] Taking off")
    await drone.action.takeoff()
    await asyncio.sleep(10)
    print("[DEBUG] Takeoff complete")
    return drone

async def search_for_marker(timeout_seconds=5.0):
    print(f"[DEBUG] Starting marker search with timeout {timeout_seconds} s")
    start_time = time.time()
    previous_gray = None

    while time.time() - start_time < timeout_seconds:
        frame = await asyncio.to_thread(camera.capture_array)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if previous_gray is not None:
            corners_to_track = cv2.goodFeaturesToTrack(previous_gray,
                                                       maxCorners=100,
                                                       qualityLevel=0.01,
                                                       minDistance=20)
            if corners_to_track is not None:
                current_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    previous_gray, gray, corners_to_track, None
                )
                valid_count = np.count_nonzero(status)
                print(f"[DEBUG] Optical flow valid corners: {valid_count}")
                if valid_count >= 6:
                    transformation_matrix, _ = cv2.estimateAffinePartial2D(
                        corners_to_track[status.flatten() == 1],
                        current_pts[status.flatten() == 1]
                    )
                    if transformation_matrix is not None:
                        frame = cv2.warpAffine(frame,
                                               transformation_matrix,
                                               frame.shape[1::-1])
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        print("[DEBUG] Applied frame stabilization warp")

        previous_gray = gray
        corners, ids, _ = cv2.aruco.detectMarkers(gray,
                                                  aruco_dictionary,
                                                  parameters=marker_detector_parameters)
        if ids is not None and target_marker_id in ids.flatten():
            print(f"[DEBUG] Detected marker ID {target_marker_id}")
            index = list(ids.flatten()).index(target_marker_id)
            _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[index]],
                marker_physical_size_m,
                intrinsic_matrix,
                distortion_coefficients
            )
            offset_vector = tvecs[0][0]
            print(f"[DEBUG] Marker pose offset (x, y, z): {offset_vector}")
            return offset_vector

        print("[DEBUG] Marker not found in this frame")

    print("[DEBUG] Marker search timed out")
    return None

async def approach_and_land_on_marker(drone, initial_offset):
    print("[DEBUG] Beginning approach_and_land_on_marker sequence")
    async for od in drone.telemetry.position_velocity_ned():
        north_initial, east_initial, down_initial = (
            od.position.north_m,
            od.position.east_m,
            od.position.down_m,
        )
        break
    async for attitude in drone.telemetry.attitude_euler():
        yaw_degrees = attitude.yaw_deg
        break

    print("[DEBUG] Starting offboard control")
    await drone.offboard.set_position_ned(
        PositionNedYaw(north_initial,
                       east_initial,
                       down_initial,
                       yaw_degrees)
    )
    try:
        await drone.offboard.start()
        print("[DEBUG] Offboard started")
    except OffboardError:
        print("[ERROR] Offboard start failed")
        return

    async def wait_until_position_reached(target_north, target_east, target_down=None):
        while True:
            await asyncio.sleep(0.2)
            async for od in drone.telemetry.position_velocity_ned():
                err_n = abs(od.position.north_m - target_north)
                err_e = abs(od.position.east_m - target_east)
                if target_down is None:
                    print(f"[DEBUG] Pos error -> N:{err_n:.3f}, E:{err_e:.3f}")
                    if err_n < position_tolerance_m and err_e < position_tolerance_m:
                        print("[DEBUG] Horizontal position tolerance reached")
                        return
                else:
                    err_d = abs(od.position.down_m - target_down)
                    print(f"[DEBUG] Pos error -> N:{err_n:.3f}, E:{err_e:.3f}, D:{err_d:.3f}")
                    if (err_n < position_tolerance_m and
                        err_e < position_tolerance_m and
                        err_d < 0.05):
                        print("[DEBUG] Full 3D position tolerance reached")
                        return
                break

    # Step 1: center above marker at current altitude
    target_north = north_initial + initial_offset[1]
    target_east  = east_initial  + initial_offset[0]
    print(f"[DEBUG] Step 1: Centering horizontally to N:{target_north:.3f}, E:{target_east:.3f}")
    await drone.offboard.set_position_ned(
        PositionNedYaw(target_north, target_east, down_initial, yaw_degrees)
    )
    await wait_until_position_reached(target_north, target_east)

    # Halving passes
    for step_index in range(number_of_halving_steps):
        target_down = down_initial / (2 ** (step_index + 1))
        print(f"[DEBUG] Step 2.{step_index+1}: Descend to down={target_down:.3f}")
        await drone.offboard.set_position_ned(
            PositionNedYaw(target_north, target_east, target_down, yaw_degrees)
        )
        await wait_until_position_reached(target_north, target_east, target_down)

        print(f"[DEBUG] Searching for marker refinement (pass {step_index+1})")
        new_offset = await search_for_marker(timeout_seconds=5.0)
        if new_offset is None:
            print("[WARN] Marker not found for refinement; proceeding to landing")
            break

        async for od in drone.telemetry.position_velocity_ned():
            current_north, current_east = (
                od.position.north_m,
                od.position.east_m,
            )
            break

        target_north = current_north + new_offset[1]
        target_east  = current_east  + new_offset[0]
        print(f"[DEBUG] Refinement {step_index+1}: new center N:{target_north:.3f}, E:{target_east:.3f}")
        await drone.offboard.set_position_ned(
            PositionNedYaw(target_north, target_east, target_down, yaw_degrees)
        )
        await wait_until_position_reached(target_north, target_east, target_down)

    # Final landing
    print("[DEBUG] Stopping offboard and landing")
    await drone.offboard.stop()
    await drone.action.land()
    print("[DEBUG] Land command issued")

async def main_mission():
    drone = None
    try:
        drone = await connect_and_arm_drone()
        for idx, (latitude, longitude) in enumerate(waypoint_coordinates, start=1):
            print(f"[DEBUG] Heading to waypoint {idx}: lat={latitude}, lon={longitude}")
            await drone.action.goto_location(latitude, longitude, absolute_altitude_msl, 0.0)
            await asyncio.sleep(7)
            print(f"[DEBUG] Attempting marker search at waypoint {idx}")
            offset_vector = await search_for_marker(timeout_seconds=10.0)
            if offset_vector is not None:
                print(f"[DEBUG] Marker found at waypoint {idx}, commencing landing sequence")
                await approach_and_land_on_marker(drone, offset_vector)
                return
        print("[DEBUG] Marker not found at any waypoint, returning to launch")
        await drone.action.return_to_launch()
    except Exception as exc:
        print(f"[ERROR] Exception during mission: {exc}")
    finally:
        if drone:
            try:
                await drone.offboard.stop()
            except:
                pass
            await drone.action.land()
        print("[DEBUG] Mission complete or aborted")

if __name__ == "__main__":
    print("[DEBUG] Mission script start")
    asyncio.run(main_mission())
    print("[DEBUG] Mission script exit")
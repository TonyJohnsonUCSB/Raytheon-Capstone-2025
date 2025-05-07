import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw

# --- PID Controller Parameters for Vibration-Resistant Tracking ---
# East (X) axis gains
Kp_x = 1.0       # Proportional gain for eastward error
Ki_x = 0.0       # Integral gain for eastward error
Kd_x = 0.2       # Derivative gain for eastward error
# North (Y) axis gains
Kp_y = 1.0       # Proportional gain for northward error
Ki_y = 0.0       # Integral gain for northward error
Kd_y = 0.2       # Derivative gain for northward error

# Internal PID state
prev_error_x = 0.0
prev_error_y = 0.0
integral_x = 0.0
integral_y = 0.0
last_time = time.time()

# Frame stabilization previous gray frame
prev_gray = None

# --- ArUco & Camera Calibration setup ---
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()
# Adjust detector for blur and lighting
parameters.adaptiveThreshConstant = 7
parameters.minMarkerPerimeterRate = 0.03

camera_matrix = np.array([[933.15867, 0, 657.59],
                          [0, 933.1586, 400.36993],
                          [0, 0, 1]])
dist_coeffs = np.array([-0.43948, 0.18514, 0, 0])

marker_size = 0.06611  # meters
drop_zone_id = 1      # ArUco ID to track

picam2 = Picamera2()
# configure with manual exposure for faster shutter
camera_config = picam2.create_preview_configuration(
    raw={"size": (1640, 1232)},
    main={"format": 'RGB888', "size": (640, 480)}
)
picam2.configure(camera_config)

print("-- Camera starting...")
picam2.start()
# allow sensor to adjust
time.sleep(2)
print("-- Camera started")



# --- PID controller functions ---
def pid_east(error_x: float, dt: float) -> float:
    """
    Compute PID output for east (X) axis based on positional error.
    :param error_x: Current error in meters (positive if right of target).
    :param dt: Time elapsed since last update in seconds.
    :return: PID output velocity (m/s) for eastward motion.
    """
    global prev_error_x, integral_x
    integral_x += error_x * dt
    derivative = (error_x - prev_error_x) / dt if dt > 0 else 0.0
    output = Kp_x * error_x + Ki_x * integral_x + Kd_x * derivative
    prev_error_x = error_x
    return -output  # negative sign to correct direction


def pid_north(error_y: float, dt: float) -> float:
    """
    Compute PID output for north (Y) axis based on positional error.
    :param error_y: Current error in meters (positive if ahead of target).
    :param dt: Time elapsed since last update in seconds.
    :return: PID output velocity (m/s) for northward motion.
    """
    global prev_error_y, integral_y
    integral_y += error_y * dt
    derivative = (error_y - prev_error_y) / dt if dt > 0 else 0.0
    output = Kp_y * error_y + Ki_y * integral_y + Kd_y * derivative
    prev_error_y = error_y
    return -output  # swapped sign for north axis control

# --- Drone connection and takeoff ---
async def connect_and_arm() -> System:
    """
    Connects to the drone, waits for health, arms, and takes off to 6m.
    """
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
    print("-- Taking off to 6m")
    #await drone.action.set_takeoff_altitude(6)
    await drone.action.takeoff()
    await asyncio.sleep(6)
    return drone

# --- Main offboard tracking loop ---
async def offboard_loop(drone: System):
    global prev_gray, last_time, integral_x, integral_y, prev_error_x, prev_error_y
    # Initialize offboard with heading fixed to north (0° yaw)
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"Offboard start failed: {e._result.result}")
        return

    print("-- Entering PID tracking loop --")
    try:
        while True:
            # Calculate elapsed time
            now = time.time()
            dt = now - last_time if last_time else 0.01
            last_time = now

            # Capture and stabilize frame
            frame = await asyncio.to_thread(picam2.capture_array)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                stabilized = frame.copy()
            else:
                # Estimate global motion via optical flow
                prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200,
                                                   qualityLevel=0.01, minDistance=30)
                M = None
                if prev_pts is not None:
                    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
                    valid = status.reshape(-1) == 1
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

            # ArUco detection on stabilized frame
            gray_stab = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray_stab, ARUCO_DICT, parameters=parameters)

            # Default motion commands
            vel_east = vel_north = 0.0
            x_cam = y_cam = z_cam = None

            if ids is not None:
                for idx, mid in enumerate(ids.flatten()):
                    if int(mid) != drop_zone_id:
                        continue
                    # Draw and estimate pose
                    cv2.aruco.drawDetectedMarkers(stabilized, [corners[idx]])
                    _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[idx]], marker_size, camera_matrix, dist_coeffs)
                    x_cam, y_cam, z_cam = tvecs[0][0]

                    # Compute PID velocities
                    vel_east = pid_east(x_cam, dt)
                    vel_north = pid_north(y_cam, dt)
                    break
            else:
                # Reset integrals and errors when no marker detected
                integral_x = integral_y = 0.0
                prev_error_x = prev_error_y = 0.0

            # Send velocity command with yaw locked to north
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(vel_north, vel_east, 0.0, 0.0)
            )

            # Overlay telemetry
            if x_cam is not None:
                cv2.putText(stabilized, f"x={x_cam:.3f} y={y_cam:.3f} z={z_cam:.3f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(stabilized, f"vx={vel_east:.2f} vy={vel_north:.2f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                cv2.putText(stabilized, "No marker detected",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Record and display
            cv2.imshow("Preview", stabilized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        pass
    finally:
        # Stop offboard and land
        print("-- Stopping offboard, landing")
        try:
            await drone.offboard.stop()
        except OffboardError as e:
            print(f"Offboard stop failed: {e._result.result}")
        await drone.action.land()
        await asyncio.sleep(5)
        await drone.action.disarm()
        cv2.destroyAllWindows()

# --- Entry point ---
async def main():
    drone = await connect_and_arm()
    task = asyncio.create_task(offboard_loop(drone))
    try:
        await task
    except KeyboardInterrupt:
        task.cancel()
        print("-- Cleaning up...")
        picam2.stop()
        picam2.close()
        out.release()
        cv2.destroyAllWindows()
        await task

if __name__ == "__main__":
    asyncio.run(main())

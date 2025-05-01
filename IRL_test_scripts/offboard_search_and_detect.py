import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

# List of (latitude, longitude) coordinates to form a square path
coordinates = [
    (34.4189167, -119.8553056),
    (34.4189722, -119.8553056),
    (34.4189722, -119.8551667),
    (34.4189167, -119.8551667)
]

# --- ArUco & camera calibration setup ---
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

camera_matrix = np.array([[933.15867, 0, 657.59],
                          [0, 933.1586, 400.36993],
                          [0, 0, 1]])
dist_coeffs = np.array([-0.43948, 0.18514, 0, 0])

marker_size = 0.06611
drop_zone_id = 1

# Control gains & limits (unused here; logic is hard-coded per your original)
max_vel = 1.0

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

# Initialize PiCamera2 + window
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    raw={"size": (1640, 1232)},
    main={"format": 'RGB888', "size": (640, 480)}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Preview", 640, 480)

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
    await drone.action.set_takeoff_altitude(3.0)  # Set altitude to 5 meters
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

    marker_found = False

    try:
        while True:
            # capture & detect in a thread
            frame = await asyncio.to_thread(picam2.capture_array)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=parameters)

            # default
            vel_east  = 0.0
            vel_north = 0.0
            x_cam = y_cam = z_cam = None

            if ids is not None:
                ids = ids.flatten()
                for idx, mid in enumerate(ids):
                    if mid != drop_zone_id:
                        continue

                    # draw & pose-estimate
                    cv2.aruco.drawDetectedMarkers(frame, [corners[idx]])
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[idx]], marker_size, camera_matrix, dist_coeffs
                    )
                    tvec = tvecs[0][0]
                    x_cam, y_cam, z_cam = tvec

                    # piecewise velocities
                    vel_east  = compute_vel_east(x_cam)
                    vel_north = compute_vel_north(y_cam)

                    marker_found = True
                    break

            # overlay
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

            # send offboard (body-x = north, body-y = east, body-z = 0, yawspeed=0)
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(vel_north, vel_east, 0.0, 0.0)
            )

            cv2.imshow("Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(0.05)

    except asyncio.CancelledError:
        pass

    return marker_found
    # finally:
    #     print("-- Stopping offboard, landing")
    #     try:
    #         await drone.offboard.stop()
    #     except OffboardError as e:
    #         print(f"Offboard stop failed: {e._result.result}")
    #     await drone.action.land()
    #     await asyncio.sleep(5)
    #     await drone.action.disarm()

async def main():
    drone = await connect_and_arm()
    task = asyncio.create_task(offboard_loop(drone))
    # waypoint loop
    print("-- Starting waypoint loop")
    for idx, (lat, lon) in enumerate(coordinates):
        print(f"-- Going to waypoint {idx + 1}: ({lat}, {lon})")
        await drone.action.goto_location(lat, lon, ALTITUDE, 0.0)
        await asyncio.sleep(5)
        try:
            marker_found = await task
        except KeyboardInterrupt:
            task.cancel()
            marker_found = await task
        if marker_found:
            print("-- Marker found! Landing")
            await drone.action.land()
            await asyncio.sleep(10)
            return
    print("-- Marker not found at any waypoint :( Landing")
    await drone.action.land()
    await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())

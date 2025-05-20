import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

# --- ArUco & Camera Calibration setup (unchanged) ---
ARUCO_DICT       = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters       = cv2.aruco.DetectorParameters_create()
parameters.adaptiveThreshConstant = 7
parameters.minMarkerPerimeterRate = 0.03

camera_matrix = np.array([[933.15867, 0, 657.59],
                          [0, 933.1586, 400.36993],
                          [0, 0, 1]])
dist_coeffs   = np.array([-0.43948, 0.18514, 0, 0])

marker_size   = 0.06611  # m
drop_zone_id  = 1

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    raw={"size": (1640, 1232)},
    main={"format": 'RGB888', "size": (640, 480)}
)
picam2.configure(config)
picam2.start()
time.sleep(2)  # let exposure settle

# --- Drone connect & takeoff (unchanged) ---
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
    await drone.action.set_takeoff_altitude(6.0)
    await drone.action.takeoff()
    await asyncio.sleep(6)
    return drone

# --- Offboard + ArUco tracking ---
async def offboard_loop(drone: System):
    # initialize offboard in POSITION mode at current alt=6 m, yaw=0°
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -6.0, 0.0))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"Offboard start failed: {e._result}")
        return

    print("-- Tracking loop --")
    try:
        while True:
            frame = await asyncio.to_thread(picam2.capture_array)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=parameters)

            if ids is not None:
                for i, mid in enumerate(ids.flatten()):
                    if int(mid) != drop_zone_id:
                        continue

                    cv2.aruco.drawDetectedMarkers(frame, [corners[i]])
                    _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[i]], marker_size, camera_matrix, dist_coeffs
                    )
                    x_cam, y_cam, z_cam = tvecs[0][0]

                    # map camera→NED: east = +x_cam, north = +z_cam
                    north_m =  z_cam
                    east_m  =  x_cam
                    down_m  = -6.0    # hold 6 m altitude
                    yaw_deg =  0.0

                    # fly to marker
                    await drone.offboard.set_position_ned(
                        PositionNedYaw(north_m, east_m, down_m, yaw_deg)
                    )
                    break
            # show feedback
            if ids is not None and any(int(m)==drop_zone_id for m in ids.flatten()):
                cv2.putText(frame, f"Δ east={x_cam:.2f}m north={z_cam:.2f}m",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
            else:
                cv2.putText(frame, "No marker", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)

            cv2.imshow("Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(0.05)

    finally:
        print("-- Stopping offboard & landing")
        try:
            await drone.offboard.stop()
        except OffboardError:
            pass
        await drone.action.land()
        await asyncio.sleep(5)
        await drone.action.disarm()
        cv2.destroyAllWindows()

# --- Entry point ---
async def main():
    drone = await connect_and_arm()
    await offboard_loop(drone)
    picam2.stop()

if __name__ == "__main__":
    asyncio.run(main())

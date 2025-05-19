#!/usr/bin/env python3

import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

# -- Calibration (hard‑coded) --
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

# -- Params --
ARUCO_DICT   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters   = cv2.aruco.DetectorParameters_create()
parameters.adaptiveThreshConstant = 7
parameters.minMarkerPerimeterRate  = 0.03

MARKER_SIZE  = 0.06611  # meters
DROP_ZONE_ID = 1        # ArUco ID to track
TOLERANCE    = 0.1     # meters

# -- Camera setup & recording --
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    raw  = {"size": (1640, 1232)},
    main = {"format": "RGB888", "size": (640, 480)}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

fourcc    = cv2.VideoWriter_fourcc(*"XVID")
out       = cv2.VideoWriter("/home/rtxcapstone/Desktop/gameTime_full.avi",
                            fourcc, 20.0, (640, 480))

_recording = True

async def _record_task():
    """Continuously grab & write frames until _recording is cleared."""
    global _recording
    while _recording:
        frame = await asyncio.to_thread(picam2.capture_array)
        out.write(frame)
        await asyncio.sleep(0.05)

async def connect_and_arm():
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
    # give it a few seconds to reach altitude
    await asyncio.sleep(10)
    return drone

async def offboard_position_loop(drone: System):
    await drone.telemetry.set_rate_position_velocity_ned(10)

    # init NED setpoint
    async for odom in drone.telemetry.position_velocity_ned():
        init_n = odom.position.north_m / 12
        init_e = odom.position.east_m  / 12
        init_d = odom.position.down_m  / 12
        break

    await drone.offboard.set_position_ned(PositionNedYaw(init_n, init_e, init_d, 0.0))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"Offboard start failed: {e._result.result}")
        await drone.action.disarm()
        return

    prev_gray = None
    try:
        while True:
            frame = await asyncio.to_thread(picam2.capture_array)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # stabilize
            stab = frame.copy()
            if prev_gray is not None:
                pts = cv2.goodFeaturesToTrack(prev_gray, 200, 0.01, 30)
                if pts is not None:
                    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                    valid = status.reshape(-1) == 1
                    if np.count_nonzero(valid) >= 6:
                        M, _ = cv2.estimateAffinePartial2D(pts[valid], curr_pts[valid])
                        stab = cv2.warpAffine(frame, M, frame.shape[1::-1])
            prev_gray = gray

            # detect ArUco
            gray_s = cv2.cvtColor(stab, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray_s, ARUCO_DICT, parameters=parameters)

            if ids is not None and DROP_ZONE_ID in ids:
                idx = list(ids.flatten()).index(DROP_ZONE_ID)
                cv2.aruco.drawDetectedMarkers(stab, [corners[idx]])
                _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS
                )
                x_cam, y_cam, z_cam = tvecs[0][0]

                # current NED
                async for odom in drone.telemetry.position_velocity_ned():
                    cn, ce, cd = odom.position.north_m, odom.position.east_m, odom.position.down_m
                    break

                target_north = cn + y_cam
                target_east = ce + x_cam

                await drone.offboard.set_position_ned(PositionNedYaw(target_north, target_east, cd, 0.0))
                await asyncio.sleep(5)

                async for odom in drone.telemetry.position_velocity_ned():
                    err_n = abs(odom.position.north_m - target_north)
                    err_e = abs(odom.position.east_m  - target_east)
                    break

                if err_n < TOLERANCE and err_e < TOLERANCE:
                    print("Reached drop zone → landing")
                    await drone.offboard.stop()
                    await drone.action.land()
                    break

                cv2.putText(stab,
                            f"dN={y_cam:.2f} dE={x_cam:.2f} dD={z_cam:.2f}",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                cv2.putText(stab, "No marker", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            cv2.imshow("Preview", stab)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                # manual quit → land
                await drone.offboard.stop()
                await drone.action.land()
                break
            await asyncio.sleep(0.1)

    finally:
        # ensure we land & disarm even on errors
        try:
            await drone.offboard.stop()
        except OffboardError:
            pass
        await drone.action.land()
        await asyncio.sleep(5)
        await drone.action.disarm()

async def main():
    global _recording
    # start background record
    recorder = asyncio.create_task(_record_task())

    # take off & track
    drone = await connect_and_arm()
    await offboard_position_loop(drone)

    # stop & finalize AVI
    _recording = False
    await recorder
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
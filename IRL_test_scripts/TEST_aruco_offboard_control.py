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

MARKER_SIZE   = 0.06611  # meters
DROP_ZONE_ID  = 1        # ArUco ID to track

# -- Camera setup & recording --
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    raw  = {"size": (1640, 1232)},
    main = {"format": "RGB888", "size": (640, 480)}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

# set up video writer to record preview
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out    = cv2.VideoWriter(
    "/home/rtxcapstone/Desktop/testVideo.avi",
    fourcc,
    20.0,
    (640, 480)
)

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
    await asyncio.sleep(6)
    return drone

async def offboard_position_loop(drone: System):
    await drone.telemetry.set_rate_position_velocity_ned(10)
    async for odom in drone.telemetry.position_velocity_ned():
        init_north = odom.position.north_m
        init_east  = odom.position.east_m
        init_down  = odom.position.down_m
        break

    await drone.offboard.set_position_ned(
        PositionNedYaw(init_north, init_east, init_down, 0.0)
    )
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
            if prev_gray is not None:
                pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200,
                                              qualityLevel=0.01, minDistance=30)
                M = None
                if pts is not None:
                    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                    valid = status.reshape(-1) == 1
                    if np.count_nonzero(valid) >= 6:
                        M, _ = cv2.estimateAffinePartial2D(pts[valid], curr_pts[valid])
                stab = cv2.warpAffine(frame, M, frame.shape[1::-1]) if M is not None else frame.copy()
            else:
                stab = frame.copy()
            prev_gray = gray

            # detect ArUco
            gray_stab = cv2.cvtColor(stab, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray_stab, ARUCO_DICT, parameters=parameters)

            if ids is not None and DROP_ZONE_ID in ids:
                idx = list(ids.flatten()).index(DROP_ZONE_ID)
                cv2.aruco.drawDetectedMarkers(stab, [corners[idx]])
                _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[idx]], MARKER_SIZE, INTRINSIC, DIST_COEFFS
                )
                x_cam, y_cam, z_cam = tvecs[0][0]

                async for odom in drone.telemetry.position_velocity_ned():
                    curr_north = odom.position.north_m
                    curr_east  = odom.position.east_m
                    curr_down  = odom.position.down_m
                    break

                target_north = curr_north + y_cam
                target_east  = curr_east  + x_cam
                target_down  = curr_down  + z_cam

                await drone.offboard.set_position_ned(
                    PositionNedYaw(target_north, target_east, target_down, 0.0)
                )
                cv2.putText(stab,
                            f"DeltaN={y_cam:.2f} DeltaE={x_cam:.2f} DeltaD={z_cam:.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                cv2.putText(stab, "No marker", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            cv2.imshow("Preview", stab)
            out.write(stab)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            await asyncio.sleep(0.1)

    finally:
        out.release()
        try:
            await drone.offboard.stop()
        except OffboardError:
            pass
        await drone.action.land()
        await asyncio.sleep(5)
        await drone.action.disarm()
        cv2.destroyAllWindows()

async def main():
    drone = await connect_and_arm()
    await offboard_position_loop(drone)

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3

import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

# -- Calibration (hard-coded) --
INTRINSIC = np.array([
    [653.1070007239106, 0.0, 339.2952147845755],
    [0.0, 650.7753992788821, 258.1165494889447],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

DIST_COEFFS = np.array([
    -0.03887864427953473,
     0.6888798469690414,
     0.00815702400928161,
     0.010438854120041072,
    -1.713270699000528
], dtype=np.float32)

# -- ArUco & flight params --
ARUCO_DICT    = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters    = cv2.aruco.DetectorParameters_create()
parameters.adaptiveThreshConstant = 7
parameters.minMarkerPerimeterRate  = 0.03

MARKER_SIZE   = 0.06611   # meters
DROP_ZONE_ID  = 1         # ArUco ID to track
TOLERANCE     = 0.1       # meters

# -- Camera setup --
picam2 = Picamera2()
preview_size = (640, 480)
cam_cfg = picam2.create_preview_configuration(
    raw={"size": (1640, 1232)},
    main={"format": "RGB888", "size": preview_size}
)
picam2.configure(cam_cfg)
picam2.start()
print("Camera initialized, waiting for auto-exposure...")
time.sleep(2)

# -- Recording globals --
fourcc     = cv2.VideoWriter_fourcc(*"XVID")
output_uri = "/home/rtxcapstone/Desktop/gameTime_full.avi"
writer     = None
recording  = False


def start_recording():
    """Begin video recording to AVI."""
    global writer, recording
    writer = cv2.VideoWriter(output_uri, fourcc, 20.0, preview_size)
    recording = True
    print(f"Recording started: {output_uri}")


def stop_recording():
    """Stop recording and release the writer."""
    global writer, recording
    if recording and writer is not None:
        writer.release()
        print("Recording stopped")
    recording = False


async def _record_task():
    """Background task: capture & write frames while recording flag is set."""
    while recording:
        frame = await asyncio.to_thread(picam2.capture_array)
        writer.write(frame)
        await asyncio.sleep(0.05)


async def connect_and_arm():
    """Connect to the drone, arm, set altitude, start recording, and take off."""
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected")
            break

    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("Drone GPS & home position OK")
            break

    await drone.action.arm()
    print("Drone armed")

    await drone.action.set_takeoff_altitude(6)
    print("Takeoff altitude set to 6 meters")

    # start recording at takeoff
    start_recording()
    await drone.action.takeoff()
    print("Taking off...")
    await asyncio.sleep(10)
    print("Hovering at altitude")

    return drone


async def offboard_position_loop(drone: System):
    """Enter offboard mode, detect ArUco, approach drop zone, and land."""
    # prepare offboard hold at current pose
    async for odom in drone.telemetry.position_velocity_ned():
        north0, east0, down0 = (
            odom.position.north_m,
            odom.position.east_m,
            odom.position.down_m
        )
        break
    async for att in drone.telemetry.attitude_euler():
        yaw = att.yaw_deg
        break

    await drone.offboard.set_position_ned(
        PositionNedYaw(north0, east0, down0, yaw)
    )
    try:
        await drone.offboard.start()
        print("Offboard mode started")
    except OffboardError as e:
        print(f"Offboard start failed: {e._result.result}")
        await drone.action.disarm()
        stop_recording()
        return

    prev_gray = None
    try:
        while True:
            # capture frame and convert to gray (RGB input)
            frame = await asyncio.to_thread(picam2.capture_array)
            rgb_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # stabilize via optical flow
            stab = frame.copy()
            if prev_gray is not None:
                pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200,
                                              qualityLevel=0.01,
                                              minDistance=30)
                if pts is not None:
                    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray, rgb_gray, pts, None
                    )
                    valid = status.reshape(-1) == 1
                    if np.count_nonzero(valid) >= 6:
                        M, _ = cv2.estimateAffinePartial2D(
                            pts[valid], curr_pts[valid]
                        )
                        if M is not None:
                            stab = cv2.warpAffine(
                                frame, M, frame.shape[1::-1]
                            )
            prev_gray = rgb_gray

            # detect on stabilized RGB frame
            stab_gray = cv2.cvtColor(stab, cv2.COLOR_RGB2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(
                stab_gray, ARUCO_DICT, parameters=parameters
            )

            if ids is not None and DROP_ZONE_ID in ids.flatten():
                idx = list(ids.flatten()).index(DROP_ZONE_ID)
                cv2.aruco.drawDetectedMarkers(stab, [corners[idx]])
                _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[idx]], MARKER_SIZE,
                    INTRINSIC, DIST_COEFFS
                )
                x_cam, y_cam, z_cam = tvecs[0][0]

                # get current NED
                async for odom in drone.telemetry.position_velocity_ned():
                    cn, ce, cd = (
                        odom.position.north_m,
                        odom.position.east_m,
                        odom.position.down_m
                    )
                    break

                # compute and send target
                target_n = cn + y_cam
                target_e = ce + x_cam
                await drone.offboard.set_position_ned(
                    PositionNedYaw(target_n, target_e, cd, yaw)
                )
                await asyncio.sleep(5)

                # check position error
                async for odom in drone.telemetry.position_velocity_ned():
                    err_n = abs(odom.position.north_m - target_n)
                    err_e = abs(odom.position.east_m - target_e)
                    break
                if err_n < TOLERANCE and err_e < TOLERANCE:
                    print("Reached drop zone, landing...")
                    await drone.offboard.stop()
                    await drone.action.land()
                    break

                cv2.putText(
                    stab,
                    f"dN={y_cam:.2f} dE={x_cam:.2f} dD={z_cam:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )
            else:
                cv2.putText(
                    stab, "No marker",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2
                )

            cv2.imshow("Preview", stab)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                await drone.offboard.stop()
                await drone.action.land()
                break

            await asyncio.sleep(0.1)

    finally:
        # cleanup on exit or error
        try:
            await drone.offboard.stop()
        except OffboardError:
            pass
        await drone.action.land()
        await asyncio.sleep(5)
        await drone.action.disarm()
        stop_recording()


async def main():
    # take off & start recording
    drone = await connect_and_arm()
    recorder = asyncio.create_task(_record_task())

    # perform offboard detection and landing
    await offboard_position_loop(drone)

    # stop recording and close windows
    global recording
    recording = False
    await recorder
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())

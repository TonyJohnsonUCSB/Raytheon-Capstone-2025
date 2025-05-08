#!/usr/bin/env python3

import numpy as np
import cv2
import asyncio
import time
from math import cos, radians
from picamera2 import Picamera2
from mavsdk import System

# === CAMERA + ARUCO SETUP ===
ARUCO_DICT = {
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250  
}
aruco_type = "DICT_6X6_250"
intrinsic_camera = np.array(((933.15867, 0, 657.59), (0, 933.1586, 400.36993), (0, 0, 1)))
distortion = np.array((-0.43948, 0.18514, 0, 0))
drop_zoneID = 1
marker_size = 0.06611  # in meters

# === HELPER FUNCTIONS ===
def draw_axis(img, rvec, tvec, camera_matrix, dist_coeffs, length):
    axis = np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
    img_pts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    img_pts = np.int32(img_pts).reshape(-1,2)
    img = cv2.line(img, tuple(img_pts[0]), tuple(img_pts[1]), (0,0,255), 2)
    img = cv2.line(img, tuple(img_pts[0]), tuple(img_pts[2]), (0,255,0), 2)
    img = cv2.line(img, tuple(img_pts[0]), tuple(img_pts[3]), (255,0,0), 2)
    return img

def aruco_display(corners, ids, rejected, image, drop_zoneID):  
    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = map(lambda x: (int(x[0]), int(x[1])), corners)
            cv2.polylines(image, [np.array([topLeft, topRight, bottomRight, bottomLeft])], True, (0, 255, 0), 2)
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            status = "Drop-Off" if markerID == drop_zoneID else "Non-Drop-Off"
            color = (0, 0, 255) if markerID == drop_zoneID else (255, 0, 0)
            label = f"ID: {markerID} - {status}"
            cv2.putText(image, label, (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def local_offset_to_gps(lat, lon, north_m, east_m):
    """Convert local offset in meters to global GPS coordinates."""
    d_lat = north_m / 111111  # meters per degree latitude
    d_lon = east_m / (111111 * cos(radians(lat)))
    return lat + d_lat, lon + d_lon

async def pose_estimation_and_gps(drone, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    frame = aruco_display(corners, ids, None, frame, drop_zoneID)

    if ids is not None:
        for i, marker_id in enumerate(ids):
            if marker_id != drop_zoneID:
                continue

            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size, intrinsic_camera, distortion)
            draw_axis(frame, rvec, tvec, intrinsic_camera, distortion, 0.1)

            distance = np.linalg.norm(tvec)
            tvec_flat = np.squeeze(tvec)
            x_offset = tvec_flat[0]
            y_offset = tvec_flat[1]

            cv2.putText(frame, f"Distance: {distance:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Get current drone position
            position = await drone.telemetry.position().__anext__()
            lat, lon, abs_alt = position.latitude_deg, position.longitude_deg, position.absolute_altitude_m

            # Assuming z is vertical and drone is nadir (camera pointing straight down)
            # tvec is in camera coordinate frame: x (right), y (down), z (forward)
            # We'll treat x->East, y->North since camera is downward
            marker_lat, marker_lon = local_offset_to_gps(lat, lon, y_offset, x_offset)

            cv2.putText(frame, f"Marker GPS: {marker_lat:.6f}, {marker_lon:.6f}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            print(f"[INFO] Marker GPS Estimation: lat={marker_lat:.6f}, lon={marker_lon:.6f}")

    return frame

# === MAIN EXECUTION ===
async def run():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone")
            break

    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    print("-- Arming")
    await drone.action.arm()
    print("-- Taking off")
    await drone.action.takeoff()
    await asyncio.sleep(5)

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(raw={"size": (1640, 1232)},
                                                         main={"format": 'RGB888', "size": (640, 480)}))
    picam2.start()
    time.sleep(2)

    try:
        start = time.time()
        while time.time() - start < 20:  # 20 sec runtime
            img = picam2.capture_array()
            frame = await pose_estimation_and_gps(drone, img)
            cv2.imshow("Aruco Pose + GPS", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()
        print("-- Landing")
        await drone.action.land()
        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(run())

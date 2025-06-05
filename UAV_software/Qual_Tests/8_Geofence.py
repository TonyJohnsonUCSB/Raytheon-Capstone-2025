import asyncio
from mavsdk import System
from mavsdk.geofence import Geofence, Point
from mavsdk.mission import MissionItem

# ----------------------------
# Geofence Configuration
# ----------------------------
# Define a polygonal fence using latitude/longitude pairs
geofence_points = [
    Point(34.418606, -119.855929),
    Point(34.418600, -119.855196),
    Point(34.419221, -119.855198),
    Point(34.419228, -119.855931),
]

# ----------------------------
# Mission Waypoints
# ----------------------------
# We'll create a simple mission that first flies to a point inside the geofence
# and then attempts to fly to a point outside the fenced‐in area.
# The geofence should intercept and override any attempt to breach.
#
# Interior waypoint (inside the polygon)
INTERIOR_LAT = 34.418900
INTERIOR_LON = -119.855600
# Exterior waypoint (outside the polygon)
EXTERIOR_LAT = 34.418907
EXTERIOR_LON = -119.854805

# Altitude (above takeoff location) in meters
REL_ALTITUDE = 10.0
# Horizontal speed for mission in m/s
MISSION_SPEED = 3.0

async def run():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!")
            break

    print("Waiting for global position estimate and home position...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    # ----------------------------
    # Upload Geofence
    # ----------------------------
    print("-- Uploading geofence polygon")
    # Build a single‐polygon geofence; fences_empty=False enforces the polygon
    fence = Geofence(polygons=[geofence_points], fences_empty=False)
    try:
        await drone.geofence.upload_geofence(fence)
        print("-- Geofence upload successful")
    except Exception as e:
        print(f"Geofence upload failed: {e}")
        return

    # Monitor for any geofence breach events (for logging/debugging)
    async def monitor_fence_breach():
        async for breach in drone.geofence.fence_breach():
            print(f"[WARNING] Fence breach detected at: {breach.latitude_deg}, {breach.longitude_deg}")

    # Start the fence‐breach monitor in the background
    asyncio.ensure_future(monitor_fence_breach())

    # ----------------------------
    # Build Mission Items
    # ----------------------------
    print("-- Creating mission plan (one interior waypoint, then one exterior waypoint)")
    mission_items = []

    # First waypoint: fly to an interior point inside the geofence
    mission_items.append(
        MissionItem(
            INTERIOR_LAT,
            INTERIOR_LON,
            REL_ALTITUDE,
            MISSION_SPEED,
            is_fly_through=True,
            gimbal_pitch_deg=0.0,
            gimbal_yaw_deg=0.0,
            loiter_time_s=0.0,
            camera_action=MissionItem.CameraAction.NONE,
        )
    )

    # Second waypoint: attempt to fly to a point outside the polygon (should be blocked)
    mission_items.append(
        MissionItem(
            EXTERIOR_LAT,
            EXTERIOR_LON,
            REL_ALTITUDE,
            MISSION_SPEED,
            is_fly_through=True,
            gimbal_pitch_deg=0.0,
            gimbal_yaw_deg=0.0,
            loiter_time_s=0.0,
            camera_action=MissionItem.CameraAction.NONE,
        )
    )

    # Clear any existing mission, then upload new one
    await drone.mission.clear_mission()
    try:
        await drone.mission.upload_mission(mission_items)
        print("-- Mission uploaded successfully")
    except Exception as e:
        print(f"Mission upload failed: {e}")
        return

    # ----------------------------
    # Arm and Start Mission
    # ----------------------------
    print("-- Arming")
    await drone.action.arm()

    print("-- Taking off to {:.1f} m".format(REL_ALTITUDE))
    await drone.action.takeoff()
    # Wait until the vehicle reaches takeoff altitude
    await asyncio.sleep(8)

    print("-- Starting mission")
    try:
        await drone.mission.start_mission()
    except Exception as e:
        print(f"Failed to start mission: {e}")
        print("-- Landing")
        await drone.action.land()
        return

    # Monitor mission progress; end when mission is complete
    async for progress in drone.mission.mission_progress():
        print(f"Mission progress: {progress.current}/{progress.total}")
        if progress.current == progress.total:
            print("-- Mission complete")
            break

    # ----------------------------
    # Landing
    # ----------------------------
    print("-- Landing after mission")
    await drone.action.land()
    return

if __name__ == "__main__":
    asyncio.run(run())

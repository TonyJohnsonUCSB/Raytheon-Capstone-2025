import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw
from mavsdk.geofence import Point, Polygon, FenceType

# List of (latitude, longitude) coordinates to form a square path
coordinates = [
    (34.4189167, -119.8553056),
    (34.4189722, -119.8553056),
    (34.4189722, -119.8551667),
    (34.4189167, -119.8551667)
]

# Geofence polygon
geofence_points = [
    Point(34.418606, -119.855929),
    Point(34.418600, -119.855196),
    Point(34.419221, -119.855198),
    Point(34.419228, -119.855931)
]

ALTITUDE = 10.0

async def run():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!")
            break

    print("Waiting for global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    print("-- Uploading geofence")
    polygon = Polygon(geofence_points, FenceType.INCLUSION)
    await drone.geofence.upload_geofence([polygon])

    print("-- Arming")
    await drone.action.arm()

    print("-- Taking off")
    await drone.action.takeoff()
    await asyncio.sleep(10)

    print("-- Flying to waypoints with pauses")
    for idx, (lat, lon) in enumerate(coordinates):
        print(f"-- Going to waypoint {idx + 1}: ({lat}, {lon})")
        await drone.action.goto_location(lat, lon, ALTITUDE, 0.0)
        await asyncio.sleep(15) ### This can be changed depending on how long it takes the drone to get to each waypoint

    first_lat, first_lon = coordinates[0]
    print("-- Returning to first waypoint")
    await drone.action.goto_location(first_lat, first_lon, ALTITUDE, 0.0)
    await asyncio.sleep(15)

    print("-- Landing")
    await drone.action.land()

if __name__ == "__main__":
    asyncio.run(run())

import asyncio  # For asynchronous programming
from mavsdk import System  # Main MAVSDK interface for the drone
from mavsdk.offboard import OffboardError, PositionNedYaw  # For possible offboard control and error handling

# List of (latitude, longitude) coordinates to form a square path
coordinates = [
    (34.4189167, -119.8553056),
    (34.4189722, -119.8553056),
    (34.4189722, -119.8551667),
    (34.4189167, -119.8551667)
]

ALTITUDE = 10.0  # Target flight altitude in meters

async def run():
    drone = System()  # Create a drone system instance
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")  # Connect to the drone via serial port

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():  # Monitor connection state
        if state.is_connected:
            print("-- Connected to drone!")
            break

    print("Waiting for global position estimate...")
    async for health in drone.telemetry.health():  # Wait until global position is OK
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    print("-- Arming")
    await drone.action.arm()  # Arm the drone

    print("-- Taking off")
    await drone.action.takeoff()  # Take off
    await asyncio.sleep(10)  # Wait 10 seconds to reach takeoff altitude

    print("-- Flying to waypoints with pauses")
    for idx, (lat, lon) in enumerate(coordinates):  # Iterate through each waypoint
        print(f"-- Going to waypoint {idx + 1}: ({lat}, {lon})")
        await drone.action.goto_location(lat, lon, ALTITUDE, 0.0)  # Navigate to coordinate at set altitude
        await asyncio.sleep(15)  # Wait for drone to reach the waypoint
        print("-- Pausing for 5 seconds")
        await asyncio.sleep(5)  # Pause at the waypoint

    # Return to first coordinate
    first_lat, first_lon = coordinates[0]  # Extract first waypoint
    print("-- Returning to first waypoint")
    await drone.action.goto_location(first_lat, first_lon, ALTITUDE, 0.0)  # Fly back to start
    await asyncio.sleep(15)  # Wait to arrive

    print("-- Landing")
    await drone.action.land()  # Land the drone

if __name__ == "__main__":
    asyncio.run(run())  # Run the async main function

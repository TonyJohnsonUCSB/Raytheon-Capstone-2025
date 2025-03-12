import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw
import math

VEL = 1
VELDIAG = math.sqrt(2)/2

async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")

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

    print("-- Arming")
    await drone.action.arm()

    print("-- Taking off")
    await drone.action.takeoff()
    await asyncio.sleep(12)

    print("-- Starting offboard mode")
    try:
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed: {error._result.result}")
        print("-- Landing and Disarming")
        await drone.action.land()
        await asyncio.sleep(10)
        await drone.action.disarm()
        return

    moves = [
        (0, VEL, 5, "Moving East"),
        (0, 0, 2, "Pausing"),
        (VEL, 0, 2.5, "Moving South"),
        (0, 0, 2, "Pausing"),
        (0, -VEL, 5, "Moving West"),
        (0, 0, 2, "Pausing"),
        (VEL, 0, 2.5, "Moving South"),
        (0, 0, 2, "Pausing"),
        (0, VEL, 5, "Moving East"),
        (0, 0, 2, "Pausing"),
        (-math.sqrt(5/2), -math.sqrt(5/2), 5, "Returning Home"),
        (0, 0, 2, "Pausing")
    ]

    for north, east, duration, action_text in moves:
        print(f"-- {action_text}")
        await drone.offboard.set_velocity_ned(VelocityNedYaw(north, east, 0.0, 0.0))
        await asyncio.sleep(duration)
    
    print("-- Stopping offboard mode")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Stopping offboard mode failed: {error._result.result}")

    print("-- Landing")
    await drone.action.land()

if __name__ == "__main__":
    asyncio.run(run())

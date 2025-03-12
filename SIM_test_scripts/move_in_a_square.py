import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw

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
    await asyncio.sleep(10)

    print("-- Starting offboard mode")
    try:
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))  # Set initial velocity setpoint
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed: {error._result.result}")
        print("-- Landing and Disarming")
        await drone.action.land()
        await asyncio.sleep(10)
        await drone.action.disarm()
        return

    print("-- Moving North at 5 mph")
    velocity_north = 2.2352  # 5 mph converted to m/s
    await drone.offboard.set_velocity_ned(VelocityNedYaw(velocity_north, 0.0, 0.0, 0.0))
    await asyncio.sleep(5)

    print("-- Stopping offboard mode")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Stopping offboard mode failed: {error._result.result}")

    print("-- Landing")
    await drone.action.land()

if __name__ == "__main__":
    asyncio.run(run())

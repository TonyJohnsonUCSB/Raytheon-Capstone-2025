import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw

VEL = 1

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

    print("-- Arming")
    await drone.action.arm()

    print("-- Taking off")
    await drone.action.takeoff()
    await asyncio.sleep(20)

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

    print("-- Moving North at 1 m/s")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(VEL, 0.0, 0.0, 0.0))
    await asyncio.sleep(5)
    print("-- Waiting for 2 seconds")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0.0, 0.0, 0.0))
    await asyncio.sleep(2)

    print("-- Moving East at 1 m/s")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, VEL, 0.0, 0.0))
    await asyncio.sleep(5)
    print("-- Waiting for 2 seconds")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0.0, 0.0, 0.0))
    await asyncio.sleep(2)


    print("-- Moving South at 1 m/s")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(-VEL, 0.0, 0.0, 0.0))
    await asyncio.sleep(5)
    print("-- Waiting for 2 seconds")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0.0, 0.0, 0.0))
    await asyncio.sleep(2)


    print("-- Moving West at 1 m/s")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, -VEL, 0.0, 0.0))
    await asyncio.sleep(5)
    print("-- Waiting for 2 seconds")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0.0, 0.0, 0.0))
    await asyncio.sleep(2)



    print("-- Stopping offboard mode")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Stopping offboard mode failed: {error._result.result}")

    print("-- Landing")
    await drone.action.land()

if __name__ == "__main__":
    asyncio.run(run())

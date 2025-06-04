import asyncio
import time
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw

#-----------TASK-------------#
# Using an autonomous sequence the UAV should take off, 
# fly 50 yds and perform a controlled landing at designated site

#-----INSTRUCTIONS-----------#
# Run the script: Drone flies N @ 1 m/s for 23 s
# Then drone flies E @ 1 m/s for 23 s for total of 50 yds flown
# Drone lands autonomously after flying 50 yds

# ----------------------------
# Flight Parameters
# ----------------------------
ALTITUDE = 5       # 6 m (20 ft) (AGL) height
AMSL_ALTITUDE = ALTITUDE + 5
VELOCITY = 1     # 1 m/s speed

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
    await drone.action.set_takeoff_altitude(AMSL_ALTITUDE)
    await drone.action.takeoff()
    await asyncio.sleep(10)

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

    print("-- Flying 25 yds North")
    for _ in range(23):
        await drone.offboard.set_velocity_ned(VelocityNedYaw(VELOCITY, 0.0, 0.0, 0.0))
        await asyncio.sleep(1)

    print("-- Stabilizing")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(2)

    print("-- Flying 25 yds East")
    for _ in range(23):
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, VELOCITY, 0.0, 0.0))
        await asyncio.sleep(1)

    print("-- Stabilizing")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(2)

    print("-- Stopping offboard mode")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Stopping offboard mode failed: {error._result.result}")

    print("-- Landing drone after 50 yds flight distance covered")
    await drone.action.land()
    return

if __name__ == '__main__':
    asyncio.run(run())
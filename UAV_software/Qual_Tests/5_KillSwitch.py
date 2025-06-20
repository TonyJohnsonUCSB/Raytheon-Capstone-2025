import asyncio
import time
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw

#-----------TASK-------------#
# Using an autonomous sequence the UAV should take off, sustain flight (not to exceed 10 mins) 
# until the kill switch is pressed by an operator. The UAV should preform a controlled landing 
# within 5yds of the location (x and y axis) it had when the kill switch was pressed. 
# Peform task at 20 feet high at a speed of 5 MPH

#-----INSTRUCTIONS-----------#
# Run the script: Drone flies N @ 2.2 m/s for 10 s
# Flip the land switch after a couple seconds

# ----------------------------
# Flight Parameters
# ----------------------------
ALTITUDE = 6       # 6 m (20 ft) (AGL) height
AMSL_ALTITUDE = ALTITUDE + 5
VELOCITY = 2.2     # 2.2 m/s (5 mph) speed

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

    print("-- Fly north for 10 seconds")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(VELOCITY, 0.0, 0.0, 0.0))
    await asyncio.sleep(10)

    print("-- Stopping offboard mode")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Stopping offboard mode failed: {error._result.result}")

    print("-- Landing")
    await drone.action.land()
    return

if __name__ == '__main__':
    asyncio.run(run())
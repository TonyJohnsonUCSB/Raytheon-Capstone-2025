import asyncio
import time
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw

#-----------TASK-------------#
# Using an autonomous sequence the UAV should:
# 1. Take off to 5 m AGL (well under the 30 yd vertical limit).
# 2. Fly a square pattern at 1 m/s for 5 s for each side.
# 3. Perform a controlled landing at the start point.
#
# The square keeps the UAV within a 30 yd × 30 yd horizontal boundary.

# ----------------------------
# Flight Parameters
# ----------------------------
ALTITUDE_AGL = 5                    # meters AGL (well under 30 yd ≈ 27.43 m)
HOME_AMSL_OFFSET = 5                # assume home is ~5 m AMSL; adjust if needed
TAKEOFF_ALTITUDE_AMSL = ALTITUDE_AGL + HOME_AMSL_OFFSET

VELOCITY_MPS = 1                    # forward speed: 1 m/s
LEG_DURATION = 5                    # forward duration: 5 s

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

    print(f"-- Taking off to {ALTITUDE_AGL} m AGL ({TAKEOFF_ALTITUDE_AMSL} m AMSL)")
    await drone.action.set_takeoff_altitude(TAKEOFF_ALTITUDE_AMSL)
    await drone.action.takeoff()
    # Give time to climb to the target altitude
    await asyncio.sleep(10)

    print("-- Starting offboard mode (zero velocity origin)")
    try:
        # Must set an initial zero-velocity command before starting offboard
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed: {error._result.result}")
        print("-- Landing and Disarming")
        await drone.action.land()
        await asyncio.sleep(10)
        await drone.action.disarm()
        return

    # --- Square Pattern ---
    # 1) Fly North for 5 seconds
    print(f"-- Flying North for {LEG_DURATION} s (≈{BOUNDARY_METERS:.1f} m)")
    for _ in range(LEG_DURATION):
        await drone.offboard.set_velocity_ned(VelocityNedYaw(VELOCITY_MPS, 0.0, 0.0, 0.0))
        await asyncio.sleep(1)

    # Brief pause to stabilize
    print("-- Stabilizing after North leg")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(2)

    # 2) Fly East for 5 seconds
    print(f"-- Flying East for {LEG_DURATION} s (≈{BOUNDARY_METERS:.1f} m)")
    for _ in range(LEG_DURATION):
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, VELOCITY_MPS, 0.0, 0.0))
        await asyncio.sleep(1)

    print("-- Stabilizing after East leg")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(2)

    # 3) Fly South for 5 seconds (negative North velocity)
    print(f"-- Flying South for {LEG_DURATION} s (≈{BOUNDARY_METERS:.1f} m)")
    for _ in range(LEG_DURATION):
        await drone.offboard.set_velocity_ned(VelocityNedYaw(-VELOCITY_MPS, 0.0, 0.0, 0.0))
        await asyncio.sleep(1)

    print("-- Stabilizing after South leg")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(2)

    # 4) Fly West for 5 seconds (negative East velocity)
    print(f"-- Flying West for {LEG_DURATION} s (≈{BOUNDARY_METERS:.1f} m)")
    for _ in range(LEG_DURATION):
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, -VELOCITY_MPS, 0.0, 0.0))
        await asyncio.sleep(1)

    print("-- Stabilizing after West leg")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(2)

    # --- End of square pattern, now land ---
    print("-- Stopping offboard mode")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Stopping offboard mode failed: {error._result.result}")

    print("-- Landing drone (square pattern complete)")
    await drone.action.land()
    return
  
if __name__ == "__main__":
    asyncio.run(run())

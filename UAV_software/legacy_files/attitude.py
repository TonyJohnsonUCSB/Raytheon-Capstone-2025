import asyncio
from mavsdk import System
from mavsdk.telemetry import EulerAngle

async def main():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    print("-- Connecting to drone...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone")
            break

    print("-- Waiting for attitude telemetry...")
    try:
        async for attitude in drone.telemetry.attitude_euler():
            print(f"-- Attitude (degrees): Roll: {attitude.roll_deg:.2f}, Pitch: {attitude.pitch_deg:.2f}, Yaw: {attitude.yaw_deg:.2f}")
            await asyncio.sleep(0.2)
    except KeyboardInterrupt:
        print("-- Stopped by user")

if __name__ == "__main__":
    asyncio.run(main())

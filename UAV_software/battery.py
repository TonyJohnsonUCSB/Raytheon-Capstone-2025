import asyncio
from mavsdk import System

async def main():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected")
            break

    async for battery in drone.telemetry.battery():
        print(f"Battery: {battery.remaining_percent:.1f}%")
        break  # Remove this to keep reading continuously

if __name__ == "__main__":
    asyncio.run(main())

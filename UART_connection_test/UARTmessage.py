import asyncio
from mavsdk import System

async def test_mavsdk_connection():
    # Connect to the MAVSDK server through a serial UART connection
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected successfully!")
            break

    # Wait for a heartbeat from the drone
    print("Waiting for a heartbeat message...")
    async for heartbeat in drone.telemetry.heartbeat():
        print(f"Heartbeat received: {heartbeat}")
        break

    # Test arming the drone
    print("Arming the drone...")
    try:
        await drone.action.arm()
        print("Drone armed successfully!")
    except Exception as e:
        print(f"Failed to arm the drone: {e}")
        return

    # Test disarming the drone
    print("Disarming the drone...")
    try:
        await drone.action.disarm()
        print("Drone disarmed successfully!")
    except Exception as e:
        print(f"Failed to disarm the drone: {e}")

async def main():
    await test_mavsdk_connection()

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
from mavsdk import System

async def test_connection():
    drone = System()
    await drone.connect(system_address="udp://:14550")
    print("Connected to drone successfully!")

asyncio.run(test_connection())

import asyncio
import serial
import random
from mavsdk import System

def coonnection_test_gps():
    latitude = round(random.uniform(-90.0, 90.0), 6)
    longitude = round(random.uniform(-180.0, 180.0), 6)
    return latitude, longitude

async def main():
    # Initialize the drone connection.
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")
    ser = serial.Serial(port='/dev/ttyUSB0',baudrate=57600)
    
    print("Waiting for drone to connect...")
    # Wait until the drone is connected.
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to drone!")
            break

    # Open the serial port for telemetry radio.
    latitude, longitude = coonnection_test_gps()
    coordinates = f"{latitude},{longitude}\n".encode('utf-8')
    ser.write(coordinates)
    
    # Continuously pull GPS data and send it over the serial link.
    async for position in drone.telemetry.position():
        latitude = position.latitude_deg
        longitude = position.longitude_deg
        
        # Format and send the coordinates.
        coordinates = f"{latitude},{longitude}\n".encode('utf-8')
        ser.write(coordinates)
        print(f"Sending coordinates: {latitude},{longitude}")
        
        # Throttle the update rate.
        await asyncio.sleep(0.2)

if __name__ == "__main__":
    asyncio.run(main())

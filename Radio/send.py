import serial
import time
import random

def get_gps_coordinates():
    latitude = round(random.uniform(-90.0, 90.0), 6)
    longitude = round(random.uniform(-180.0, 180.0), 6)
    return latitude, longitude

ser = serial.Serial(port='/dev/ttyUSB0',baudrate=57600)
while True:
    latitude, longitude = get_gps_coordinates()
    coordinates = f"{latitude},{longitude}\n".encode('utf-8')
    ser.write(coordinates)
    print(f"Sending coordinates: {latitude},{longitude}")
    time.sleep(0.2)

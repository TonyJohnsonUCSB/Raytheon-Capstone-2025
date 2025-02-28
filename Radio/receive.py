import serial
import re

ser = serial.Serial(port='/dev/ttyUSB0', baudrate=57600, timeout=1)

while True:
    # Read serial data
    b = ser.read(1000).decode(errors='ignore')
    
    # Find latitude and longitude pairs using regex
    coordinates = re.findall(r'(-?\d+\.\d+),\s*(-?\d+\.\d+)', b)

    # Print extracted coordinates
    for lat, lon in coordinates:
        print(f"GPS coordinates received: {lat}, {lon}")
